# Trening HomeIMG

Wykonuj komendy z głównego katalogu repozytorium. Pierwsza receptura korzysta z gotowych latentów 256 px i uruchamia trening małego DiT na ośmiu GPU. Domyślny wycinek danych zajmuje około 39 GB; osobno przewidź miejsce na konwersję i checkpointy.

## Środowisko (osobny venv)

Utwórz osobny venv w katalogu repo (`.venv/`, ignorowany przez gita), żeby nie mieszać pakietów z systemowym
Pythonem. Wymagany jest [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128  # najpierw!
uv pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.device_count(), 'GPU')"
```

Bez uv: `python3.11 -m venv .venv && source .venv/bin/activate && pip install torch --index-url
https://download.pytorch.org/whl/cu128 && pip install -r requirements.txt`. Wszystkie dalsze komendy zakładają aktywny venv
(albo wywołanie `.venv/bin/python` / `.venv/bin/torchrun`). Sprawdzone: torch 2.11.0+cu128, diffusers 0.42.1, Python 3.11.

## Text encoder: Slayer149 (zamrożony) + REPA z DINOv2-B/14 (domyślnie)

- `--te-size slayer149` (domyślnie): zamrożony [SlayerLab/Slayer149](https://huggingface.co/SlayerLab/Slayer149) (149M,
  angielski) + trenowany refiner: domyślnie 2 bloki o połowie szerokości denoisera (2B: 1024, ~25M; 750m:
  1280, ~36M; w PP siedzi na stage'u 0, więc większy go spowalnia), `--te-dim` / `--te-depth` zmieniają rozmiar, `--te-dim 768 --te-depth 2` = dawny ~13M
  (potrzebne do wznowienia starszych runów). Wymaga osobnej tokenizacji captionów (obok istniejącej):
  `python prepare_data.py tokenize --data data/aurora_full --encoder slayer149`.
  `--te-size 13m` / `27m` to dawne enkodery trenowane od zera (z MLM).
- **REPA** (`--repa-weight 0.5`, domyślnie): stan DiT po ~30% głębokości jest dopasowywany (cosinus, przez MLP) do cech
  [DINOv2-B/14 z rejestrami](https://huggingface.co/timm/vit_base_patch14_reg4_dinov2.lvd142m) (Apache-2.0) z obrazka
  zdekodowanego z latentu (przeskalowanego do 224 px → siatka 16×16 = tokeny DiT). PE-Spatial-B16:
  `--repa-encoder vit_pe_spatial_base_patch16_512.fb`. `--irepa` = iREPA (projektor conv 3×3 + normalizacja cech po
  tokenach). Dekoder jest drogi, więc REPA liczy się tylko dla 25% każdego micro-batcha (`--repa-frac`), co N-ty krok
  (`--repa-every N`) i przez pierwsze 40% treningu (`--repa-until`). `--repa-weight 0` wyłącza.
- Eksport (`export.py`) pakuje użyte warstwy Slayer149 i jego tokenizer do repo, a `pipeline.py` (`aurora/hf_pipeline.py`)
  obsługuje oba enkodery; `tests/check_export.py` sprawdza zgodność z modułami treningowymi.

## Szybki start

```bash
source .venv/bin/activate
hf download AuroraAI-Research/AuroraIMG2-50M-Preview --local-dir AuroraIMG2-50M-Preview

# dane: pobranie + konwersja do mmap .npy + tokenizacja. Domyślnie 10% datasetu (~39 GB, ~0.57M obrazków);
# całość (~385 GB): dodać --fraction 1 (już pobrane shardy są pomijane)
python prepare_data.py convert  --out data/aurora --download --delete-raw --workers 6
python prepare_data.py tokenize --data data/aurora --encoder slayer149

# trening na 8 GPU. W tym przykładzie model 116m, 10% danych, batch 2048 (8 GPU x 64 x akumulacja 4), 3 098 kroków (~11 epok);
# całość: --fraction 1 (30 976 kroków). OOM: --micro-batch 32 (akumulacja liczy się sama). Kroki, lr i EMA przeskalowują się z batchem względem oryginału (256).
# Wznawia się sam z <out>/latest.pt (co 1/50 runu); trwałe snap_<krok>.pt co 1/10 runu.
# wandb włącza się sam, jeśli jest zalogowany (`wandb login`); projekt trenowanie_generatora_obrazkow, --no-wandb wyłącza.
torchrun --nproc_per_node 8 train.py --data data/aurora --out runs/aurora --size 116m

# WSM: zamiast wygaszać lr, uśrednij checkpointy z końcówki treningu (schemat i okno wybierane na walidacji)
python merge.py runs/aurora                                  # -> runs/aurora/merged.pt
# eksport do formatu diffusers + siatka próbek
python export.py runs/aurora/merged.pt exports/aurora
python sample.py exports/aurora --out grid.png
```

## Duże modele: pipeline parallel (1F1B), `train_pp.py`

Bloki DiT dzielone na `--pp` kart (Schedule1F1B z `torch.distributed.pipelining`). Stage 0 ma też text encoder, projekcję
czasu i `patch_in`, ostatni stage `norm`, `patch_out` i stratę. Między kartami idą stany tokenów, modulacja czasu, tokeny
tekstu i maska (oraz ich gradienty), strata REPA liczy się w stage'u z wybranym blokiem, a cele REPA (dekoder + DINOv2) liczą wszystkie karty po równo
(micro-batch k na karcie k mod pp) i wysyłają do tego stage'u. Presety: `--size 1b / 1.5b / 2b`
(szerokość 2048, 15/23/31 bloków) oraz `--size 1.6b / 2.4b` (szerokość 2560, head dim 128, 15/23 bloki; najlepsze TFLOPS na 3090 Ti wg `tests/bench_shapes.py`). Balans: zamrożony LM liczą wszystkie karty po równo (micro-batch k na karcie
k mod pp, stany wysyłane do stage'u 0), a stage 0, który ma jeszcze trenowany refiner (~1 blok pracy, `--te-cost`),
dostaje o 1 blok mniej; głębokość 8k−1 sprawia, że przy pp 8/4/2 wychodzi równo (2B na 8 kartach: [3, 4, 4, …]). Checkpointy są zbierane na rank 0 w tym samym formacie co z `train.py`, więc `export.py`,
`merge.py` i `tests/check_export.py` działają bez zmian; stan optymalizatora per stage w `<out>/opt_stage<s>.pt`.

```bash
NCCL_P2P_LEVEL=SYS torchrun --nproc_per_node 8 train_pp.py --data data/aurora_full --fraction 1 --size 1.5b \
  --pp 8 --batch 512 --microbatches 32 --out runs/1p5b
# OOM: --act-checkpoint selective (trzyma wyniki matmuli i attention, przelicza tylko elementwise: ~2x mniej
#      aktywacji, bez straty prędkości) albo --act-checkpoint (pełny: prawie zero aktywacji, ~25% wolniej);
#      2 potoki po 4 karty: --pp 4 (+ --diloco H)
```

`tests/check_pp.py` (CPU, gloo, `torchrun --nproc_per_node 1/2/4`) sprawdza, że straty i gradienty (z REPA) są identyczne
jak w niepodzielonym modelu.

## Dalszy trening opublikowanych wag

Dla dwujęzycznego modelu z Poziomką pobierz eksport i utwórz checkpoint początkowy:

```bash
source .venv/bin/activate
hf download cpral/HomeIMG-1.2B-PL-EN-epoch49.2-DanceOPD2048 --local-dir exports/homeimg
python import_homeimg.py exports/homeimg runs/homeimg-init.pt
```

Użyj go w `train_pp.py` przez `--init runs/homeimg-init.pt` z nowym katalogiem `--out`. Import zachowuje wagi denoisera i refinera, ustawia krok na zero i rozpoczyna nową historię optymalizatora. Dane muszą zawierać tokenizację dla Poziomki. Pełna konfiguracja dla tego rozmiaru modelu jest w [recepturze OPD](destylacja.md).

Aby wznowić przerwany trening, uruchom tę samą komendę z dotychczasowym katalogiem wyjściowym. Zachowaj razem `latest.pt` i pliki optymalizatora `opt_stage*.pt`. Po zakończeniu zapisz EMA w formacie Diffusers:

```bash
source .venv/bin/activate
python export.py runs/homeimg/latest.pt exports/homeimg-ema --time-shift 6 --time-shift-ref 0
```
