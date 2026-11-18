# Ewaluacja jakości obrazów

## Polski test porównawczy HomeIMG (22 prompty)

Treść promptów: [`tests/homeimg22_pl.json`](../tests/homeimg22_pl.json).
Ustawienia: [`tests/homeimg22_settings.json`](../tests/homeimg22_settings.json).
Każdy prompt generuję cztery razy z seedami **0, 1, 2, 3**: **1024 × 1024 px,
CFG 3, time shift 6, 50 kroków**. To 88 obrazów na model, 264 dla trzech modeli.
Nie używam dodatkowego negative promptu; pozostałe ustawienia pochodzą z pipeline'u eksportu.

Porównywane modele: `cpral/HomeIMG-1.2B-PL-EN-epoch49.2`,
`cpral/HomeIMG-1.2B-PL-EN-epoch49.2-DanceOPD2048` oraz
`cpral/HomeIMG-1.2B-PL-EN-epoch49.2-DiffusionOPD3072`.

Skrypt obsługuje samodzielne eksporty Diffusers (lokalne lub HF). Z katalogu repo na serwerze:

```bash
source .venv/bin/activate
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python tests/homeimg22.py cpral/HomeIMG-1.2B-PL-EN-epoch49.2 --out-dir exports/test22_epoch49.2
```

Dla kolejnego modelu zmień model i katalog wynikowy. Każdy obraz powstaje osobno
z generatorem CUDA ustawionym na podany seed, jednakowo dla wszystkich modeli.
`--part K --parts N` dzieli obrazy między procesy/GPU bez zmiany seedów ani batcha.
`--dry-run` wypisuje zadania bez ładowania modelu i generowania obrazów.
Skrypt zapisuje PNG oraz manifest z promptami, seedami i parametrami;
odmawia nadpisania istniejącego wyniku. Identyczne seedy nie gwarantują identycznych
wyników między różnymi wersjami bibliotek lub sprzętem.

Plansze do późniejszego blind testu powinny zawierać tylko obraz i pełny prompt
bezpośrednio pod nim. Nazwy modeli, seedy i parametry pozostają poza planszami;
manifestów nie udostępniam oceniającym. Po wygenerowaniu obrazów złóż plansze poleceniem `python tests/homeimg22_grid.py exports/test22_epoch49.2`.
