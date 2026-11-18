# Wydajność i komunikacja między GPU

## NCCL na kartach GeForce

Najpierw sprawdzić samą komunikację między kartami (kilka sekund, bez danych i modelu):

```bash
torchrun --nproc_per_node 8 tests/check_nccl.py
```

- **Sterownik z odblokowanym P2P** (np. łatka tinygrad na open-gpu-kernel-modules): NCCL domyślnie włącza P2P tylko dla kart
  blisko siebie w topologii PCIe. Gdy ścieżka idzie przez CPU, trzeba to wymusić:
  `NCCL_P2P_LEVEL=SYS torchrun ...`. Sprawdzenie: `python -c "import torch; print(torch.cuda.can_device_access_peer(0, 1))"`.
- **Błąd `Error while attaching to shared memory segment /dev/shm/nccl-...`**: NCCL próbuje transportu przez `/dev/shm`.
  Przy działającym P2P naprawia to `NCCL_P2P_LEVEL=SYS`. Bez P2P trzeba sprawdzić `df -h /dev/shm` (miejsce, uprawnienia
  `drwxrwxrwt`), a w ostateczności `NCCL_SHM_DISABLE=1` (wolniejszy transport przez sockety).
- `NCCL_DEBUG=INFO` pokazuje, którą drogę wybrało NCCL (`via P2P`, `via SHM`, `via NET`).
- Te same zmienne środowiskowe daje się przed `torchrun ... train.py`.

## Profil attention, GEMM i pętli treningu

`tests/profile_dit.py` mierzy osobno SDPA (self/cross/refiner), GEMM forward/backward oraz pełny blok DiT.
`--init` odczytuje rzeczywistą konfigurację checkpointu i wagi pierwszego bloku; nie ładuje Poziomki.
Wejścia są syntetyczne, a pamięć jednego bloku **nie określa**, czy pełny model zmieści się z tym mikrobatchem.

```bash
# Najpierw na jednej wolnej karcie, bez równoległego treningu na tej samej GPU:
CUDA_VISIBLE_DEVICES=0 python tests/profile_dit.py \
  --init runs/poziomka_stage2/merged.pt --depth 12 \
  --parts attention gemm block --mb 8 16 --text-widths 64 128 \
  --out runs/profile_dit

# Strojenie całego bloku; koszt kompilacji/autotuningu poza pomiarem:
CUDA_VISIBLE_DEVICES=0 python tests/profile_dit.py \
  --init runs/poziomka_stage2/merged.pt --parts block \
  --mb 8 16 24 --text-widths 64 128 256 --checkpoints none selective \
  --compile-modes default max-autotune-no-cudagraphs --out runs/profile_blocks
```

Wyniki są w `results.json`, konfiguracja sprzętu/modelu w `metadata.json`. Opcjonalne `--trace` zapisuje
osobny profil po pomiarze każdej konfiguracji: timeline `.trace.json`, tabelę operatorów `.txt` i listę
rzeczywistych kerneli oraz kształtów `.json`. Nieobsługiwane backendy są raportowane jako błędy danego przypadku,
bez cichego powrotu do innego backendu. Attention jest sprawdzane względem SDPA math (wynik i gradienty).
Wariant `varlen` korzysta z `torch.nn.attention.varlen` i obejmuje koszt pakowania K/V; nie zmienia treningu.
`--parts block --block-backends auto cudnn` porównuje backendy cross-attention wewnątrz rzeczywistego
bloku DiT, razem z projekcjami, normami, MLP i checkpointingiem.
Licznik FLOPs używa faktycznej szerokości tekstu, lecz nie dolicza przeliczeń checkpointingu ani optymalizatora.

Do istniejącej komendy `train_pp.py` można dopisać:

```bash
--profile-dir runs/profile_train --profile-start 17 --profile-steps 4 --profile-ranks 0
```

Profiler działa również z `--bench`. Włącza się tylko na wskazanych **rankach treningowych**; dla `pp=2`
warto wskazać `--profile-ranks 0 1`, aby zobaczyć obydwa etapy. Rejestruje m.in. `train/data_loader`,
`train/lm_wait_collect`, `train/lm_request`, `train/forward_backward`, `train/gradient_sync_clip`,
`train/optimizer` i `train/outer_sync_ema`. Suma czasów operatorów nie jest czasem ściennym: operatory
są zagnieżdżone, a kernele mogą się nakładać. Krok z eksportem profilu oraz okno profilowania nie służą
porównywaniu img/s. Ustaw początek po rozgrzaniu używanych szerokości tekstu.
`opt avg` pokazuje średnią z całego okna logowania, również przy okresowym Muonie.

`--masked-attention-backend cudnn` wymusza cuDNN dla cross-attention DiT i attention trainowalnego
refinera tekstu. Domyślnie pozostaje `auto`; self-attention DiT i zamrożony LM zachowują swój backend.
Wybór jest zapisany wewnątrz kompilowanego grafu. Wymaga obsługi danego kształtu przez cuDNN;
nie ma cichego fallbacku. Porównuj rozgrzane przebiegi z tym samym microbatchem i checkpointingiem,
ponieważ przyspieszenie pojedynczego attention nie przekłada się wprost na przyspieszenie całego kroku.

Na 8 GPU legalne konfiguracje do porównania (batch 1008):

| Układ | Opcje | Przykład podziału batcha |
|---|---|---|
| 7 replik, 1 serwer LM | `--pp 1 --lm-rank 1 --microbatches 9` | 7 × 9 × 16 |
| 6 replik, 2 serwery LM | `--pp 1 --lm-rank 2 --microbatches 7` | 6 × 7 × 24 |
| 8 replik z lokalnym LM | `--pp 1 --microbatches 9` (bez `--lm-rank`) | 8 × 9 × 14 |
| 4 pipeline po 2 GPU, LM rozłożony domyślnie | `--pp 2 --microbatches 18` (bez `--lm-rank`) | 4 × 18 × 14 |
| 3 pipeline po 2 GPU, 2 serwery LM | `--pp 2 --lm-rank 2 --microbatches 14` | 3 × 14 × 24 |

To konfiguracje porównawcze, **nie zmierzone rekomendacje**. `--pp 2 --lm-rank 1` jest niepoprawne dla 8 GPU.
Zmiana liczby replik zmienia lokalny batch DiLoCo oraz automatyczne skalowanie LR AdamW.
Zachowaj wymagane przez maszynę `NCCL_P2P_LEVEL=SYS`. Lokalny LM zwiększa użycie VRAM i wykonuje pracę
na tych samych GPU co denoiser; pp=2 dodaje komunikację i bąble pipeline.
