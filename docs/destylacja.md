# Destylacja HomeIMG

[English](distillation.md)

## DanceOPD / DiffusionOPD w wielu rozdzielczościach

W `train_pp.py` ustaw `--res 1024 --t-shift 6`,
`--opd-resolutions 256 512 1024` i `--opd-res-shifts 1.5 3 6`.
To osobna ścieżka od `--multi-res`, które miesza obrazy w jednym kroku zwykłego treningu FM.
W OPD jedna rozdzielczość jest losowana na świeży batch trajektorii i pozostaje
niezmienna przez wszystkie aktualizacje `--opd-reuse`. Batch i microbatch mają
stałą liczbę obrazów, dobraną do największej rozdzielczości.

Opcjonalne `--opd-token-batch` zwiększa batch niższych rozdzielczości przez
akumulację dodatkowych microbatchy: mnożnik to `round((res_max / res)^2)`.
Przy `--batch 50` i rozdzielczościach 256/512/1024 daje to 800/200/50 obrazów
globalnie, czyli po 204800 tokenów obrazu na aktualizację (dla jednego stanu
na trajektorię). Liczba obrazów w pojedynczym microbatchu i LR pozostają takie
jak dla bazowego batcha 1024 px. Większa liczba tekstów zwiększa koszt LM i refinera,
więc równa liczba tokenów obrazu nie oznacza identycznego czasu kroku.
Checkpoint zapisuje zużytą pozycję strumienia danych osobno od liczby rolloutów;
przy wznowieniu odrzucany lookahead nie powoduje pomijania captionów.

`--opd-res-weights 1 1 1` (domyślne) oznacza równe prawdopodobieństwa batchów,
**nie** bezpośrednio udziały tokenów: przy stałym batchu wynoszą one średnio
1:4:16, a z `--opd-token-batch` w tym przykładzie 1:1:1.
Można podać inne dodatnie wagi. Bez `--opd-res-shifts` przesunięcie wynosi
`res / --t-shift-ref`, jeśli ref ustawiono, w przeciwnym razie
`--t-shift * res / --res`.

Wszystkie repliki wybierają tę samą rozdzielczość na podstawie seedu i kroku
docelowego batcha. Działają zarówno lokalne rollouty, `--opd-prefetch`, jak
i osobny `--opd-actor`. Cache przenosi rozdzielczość i time shift; po wznowieniu
cache jest generowany na nowo, tak jak w dotychczasowym OPD.
Teacher otrzymuje rozmiar latentów w każdym żądaniu; RoPE i micro-conditioning
ucznia wynikają z rozmiaru wejścia. Log pokazuje bieżące `opd/resolution`
i `opd/time_shift`, a TFLOPS uwzględniają rozdzielczości i liczbę stanów
trenowanych w oknie logowania (bez kosztu rolloutu i teachera).
Wymagane `--pp 1`; nie łączyć z `--multi-res`, `--crop-res` ani REPA.
Pierwsze wystąpienie każdej rozdzielczości może wymagać kompilacji nowych grafów.

Test CPU: `PYTHONPATH=. python tests/check_opd_resolution.py`.

### Kolejka zamrożonych stanów tekstowych

`--opd-text-ahead 4` przygotowuje cztery przyszłe batche na ucznia, niezależnie
od aktora. Jeden wątek czyta captiony w kolejności harmonogramu rozdzielczości,
a osobny GPU koduje zarówno Poziomkę (przed trenowanym refinerem), jak i Qwena
teachera. Stany BF16 czekają w RAM; kolejka nie rośnie z długością treningu.
Limit dotyczy czterech przyszłych batchów, oprócz aktualnie konsumowanego.
Log podaje rozmiar stanów każdego batcha w MiB. Po wznowieniu kolejka jest
odbudowywana od zapisanej, zużytej pozycji danych. Checkpoint przechowuje pozycję danych.

Ten tryb wymaga `--opd-actor`, jednego serwera LM i dedykowanego teachera;
nie łączy się z `--teacher-on-lm`. Przykładowe rozmieszczenie siedmiu ranków:
`--skip-gpus 0 --opd-actor-gpus 2 7 --teacher-ranks 1 --teacher-gpu 4 --lm-rank 1 --lm-gpu 6`.
Uczniowie zajmują wtedy GPU 1/3/5, aktorzy 2/7, teacher obrazu 4, oba encodery
tekstu 6. Przy `--batch 30 --opd-token-batch` globalne batche to 480/120/30.
Na GPU teachera nie jest ładowany Qwen. Rozmiary cache i wydajność należy
sprawdzić benchmarkiem; gotowość tekstu nie oznacza gotowości rolloutów.
Wersja wag aktora jest przypisywana dopiero przy wysłaniu zadania, więc dłuższa
kolejka tekstu nie zwiększa dopuszczalnego wieku polityki.

Test protokołu (CPU, cztery procesy Gloo):
`PYTHONPATH=.:tests python tests/check_opd_text_queue.py`.


## Receptura HomeIMG 1.2B na siedem GPU

Uruchom z katalogu repo po przygotowaniu checkpointu Poziomki zgodnie z [instrukcją importu](trening.md). Do odtworzenia eksperymentu 5120 użyłem wag ArceeFusion jako inicjalizacji. Poniższy przykład oczekuje checkpointu pod nazwą `runs/homeimg-init.pt`, lokalnego katalogu Klein i przygotowanych shardów FLUX-Reason 1024 z podpisami PL/EN oraz tokenami Poziomki. Instrukcja dla gotowych latentów 256 px w przewodniku treningowym dotyczy osobnego zestawu danych.

W konfiguracji z ośmioma fizycznymi kartami pomijam GPU 0. Na GPU 1, 3 i 5 trenuję ucznia, na 2 i 7 generuję rollouty, na 4 działa teacher, a na 6 enkodery tekstu. Globalne batche 480/120/30 dla 256/512/1024 px mają po 122 880 tokenów obrazu. Jeden batch trajektorii służy do 12 aktualizacji.

```bash
source .venv/bin/activate
env -u CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID \
torchrun --nproc_per_node 7 train_pp.py \
  --init runs/homeimg-init.pt --out runs/homeimg \
  --data data/flux_reason_1024 --img-dir data/flux_reason_1024 --fraction 1 \
  --size 750m --depth 12 --res 1024 --rope-ref 64 --micro-cond 256 --t-shift 6 \
  --opd-resolutions 256 512 1024 --opd-res-shifts 1.5 3 6 --opd-res-weights 1 1 1 \
  --opd-token-batch --opd-res-mb 16 8 2 --batch 30 --microbatches 5 \
  --dance-opd --dance-beta 1 1 --opd-states 1 --opd-rollout-steps 20 \
  --opd-parts 5 --opd-reuse 12 --opd-state-weighting --opd-cfg 5 \
  --skip-gpus 0 --opd-actor --opd-actor-gpus 2 7 --opd-text-ahead 4 \
  --lm-rank 1 --lm-gpu 6 --teacher-ranks 1 --teacher-gpu 4 \
  --klein models/FLUX.2-klein-base-4B --klein-quant int8 \
  --te-size poziomka --te-quant int8 --pl-prob 0.5 --repa-weight 0 \
  --diloco 32 --outer-lr 1 --outer-momentum 0 --lr-muon 1e-3 --cmuon \
  --muon-period 1 --muon-momentum-dtype bf16 --ema-cpu --seed 87462 \
  --steps 5120 --log-every 1 --eval-every 0 --ckpt-every 256 --merge-every 0 \
  --pp 1 --act-checkpoint selective --enc-chunk 2 --workers 1
```
