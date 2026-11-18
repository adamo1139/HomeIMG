# HomeIMG

**Polsko-angielskie modele do generowania obrazów i kod do ich treningu.** HomeIMG rozwijam samodzielnie, od pierwszych eksperymentów z małym transformatorem dyfuzyjnym po dwujęzyczny model z denoiserem 1,24 mld parametrów. Tutaj udostępniam kod przygotowania danych, treningu, destylacji i oceny. Gotowe wagi EMA publikuję na Hugging Face.

[Wagi na Hugging Face](https://huggingface.co/cpral?search_models=HomeIMG) · [Wygeneruj obraz](#wygeneruj-obraz) · [Wytrenuj model](docs/trening.md) · [Porównaj checkpointy](docs/ewaluacja.md)

![Osiem generacji HomeIMG z polskich promptów](https://huggingface.co/cpral/HomeIMG-1.2B-PL-EN-epoch49.2-DanceOPD2048/resolve/main/grids_cfg3/grid_pl_1024px_ts6.png)

*DanceOPD 2048, osiem stałych promptów PL, 1024 × 1024 px, 50 kroków Eulera, CFG 3, time shift 6. [Plansze dla obu języków i ustawienia eksperymentu](model_cards/HomeIMG-1.2B-PL-EN-epoch49.2-DanceOPD2048/README.md).*

## Modele i etapy treningu

| Wagi | Co zmieniłem na tym etapie |
|---|---|
| [HomeIMG 641M · epoch19](https://huggingface.co/cpral/HomeIMG-641M-PL-EN-epoch19) | Wczesny dwujęzyczny checkpoint. Opisuje przejście od treningu po angielsku do podpisów PL/EN. |
| [HomeIMG 1.2B · epoch49](https://huggingface.co/cpral/HomeIMG-1.2B-PL-EN-epoch49) | Po treningu progresywnym połączyłem rozdzielczości 256, 512 i 1024 px. |
| [HomeIMG 1.2B · epoch49.2](https://huggingface.co/cpral/HomeIMG-1.2B-PL-EN-epoch49.2) | Wspólny punkt startowy dwóch gałęzi destylacji: DanceOPD i DiffusionOPD. |
| [HomeIMG 1.2B · DanceOPD 2048](https://huggingface.co/cpral/HomeIMG-1.2B-PL-EN-epoch49.2-DanceOPD2048) | 2048 aktualizacji przy 1024 px z FLUX.2 [klein] base 4B jako teacherem. Z tego wydania pochodzi plansza powyżej. |

W późniejszym eksperymencie połączyłem denoisery z DanceOPD 2048 i DiffusionOPD 3072 metodą ArceeFusion. Po destylacji przy 1024 px zaobserwowałem regresję w niższych rozdzielczościach. Z połączonych wag rozpocząłem więc kolejny trening DanceOPD, losując dla każdego batcha 256, 512 lub 1024 px i wyrównując liczbę tokenów obrazu. Zakończyłem go po 5120 aktualizacjach. [Konfiguracja eksperymentu](docs/destylacja.md).

## Wygeneruj obraz

Na maszynie z GPU NVIDIA, sterownikiem obsługującym CUDA 12.8 i zainstalowanym `uv`:

```bash
git clone https://github.com/adamo1139/HomeIMG.git
cd HomeIMG
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt

hf download cpral/HomeIMG-1.2B-PL-EN-epoch49.2-DanceOPD2048 --local-dir exports/homeimg
python sample_ckpt.py exports/homeimg --out-dir samples --batch 1 \
  --res 1024 --steps 50 --cfg 3 --t-shift 6 --seed 0 \
  --prompt "Akwarela przedstawiająca Zamek Królewski na Wawelu widziany znad Wisły."
```

Obraz znajdziesz w `samples/00.png`. Pominięcie `--prompt` generuje dwie plansze z ośmioma promptami, po polsku i po angielsku. Interfejs przeglądarkowy jest w [app.py](app.py).

## Jak działa HomeIMG

W dwujęzycznym HomeIMG zamrożona Poziomka koduje podpis, a trenowany refiner przekształca jej stany w reprezentację tekstu dla denoisera. Transformer dyfuzyjny uczy się pola prędkości w przestrzeni latentnej FLUX.2 metodą flow matching. Podczas generacji solver Eulera prowadzi latent od szumu do obrazu, który dekoduje VAE.

W OPD kopia ucznia generuje trajektorie odszumiania. Zamrożony teacher przewiduje prędkość na wybranych stanach tych trajektorii, a uczeń dopasowuje do niej własną predykcję. Kod pozwala rozdzielić trening, generowanie trajektorii i obliczenia teachera między GPU.

| Chcę… | Gdzie zacząć |
|---|---|
| Przygotować dane, wytrenować mały model albo kontynuować trening z wag HF | [Przewodnik treningowy](docs/trening.md) |
| Odtworzyć wielorozdzielczy DanceOPD lub zmienić strategię próbkowania stanów | [Destylacja i podział pracy między GPU](docs/destylacja.md) |
| Sprawdzić komunikację kart i zmierzyć koszt kroku | [NCCL i profilowanie](docs/wydajnosc.md) |
| Prześledzić implementację | [Architektura modelu](aurora/model.py) · [trening DDP](train.py) · [trening rozproszony i OPD](train_pp.py) |

## Jak porównuję checkpointy

Przygotowałem [22 polskie prompty](tests/homeimg22_pl.json). Każdy generuję z czterema ustalonymi seedami, co daje **88 obrazów na checkpoint**. W zestawie są m.in. trzy przedstawienia tej samej sceny wigilijnej: fotografia, akwarela i kadr CGI. Pozwala to ocenić, czy model potrafi zmienić medium, zachowując opis sceny. [Ustawienia i instrukcja odtworzenia plansz](docs/ewaluacja.md).

W próbkach DanceOPD 2048 CFG 3 poprawiał czytelność kształtów względem CFG 1. Nadal widać problemy z liczeniem obiektów, czytelnym tekstem oraz anatomią postaci w tłumie. To obserwacje wizualne na konkretnych promptach. [Przykłady i ograniczenia wydania](model_cards/HomeIMG-1.2B-PL-EN-epoch49.2-DanceOPD2048/README.md).

## Autorstwo i komponenty

HomeIMG rozwijam jako [adamo1139](https://github.com/adamo1139). Punktem wyjścia implementacji bloków DiT jest [AuroraIMG2-50M-Preview](https://huggingface.co/AuroraAI-Research/AuroraIMG2-50M-Preview) (Apache-2.0). Projekt korzysta również z [dekodera FLUX.2](https://huggingface.co/black-forest-labs/FLUX.2-small-decoder) (Apache-2.0), [Poziomki](https://huggingface.co/cpral/Poziomka-Baza-2026-09-05) i optymalizatora [Muon](https://kellerjordan.github.io/posts/muon/). Warunki korzystania z wag i danych znajdziesz w kartach poszczególnych wydań i zbiorów.
