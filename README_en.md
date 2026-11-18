# HomeIMG

[Polski](README.md)

**Polish–English image generation models and the code I use to train them.** I develop HomeIMG independently. The project grew from a small diffusion transformer into a bilingual model with a 1.24B-parameter denoiser. This repository covers data preparation, training, distillation and evaluation. I publish EMA weights on Hugging Face.

[Model weights](https://huggingface.co/cpral?search_models=HomeIMG) · [Generate an image](#generate-an-image) · [Train a model](docs/training.md) · [Compare checkpoints](docs/evaluation.md)

![Eight DanceOPD generations from Polish prompts](https://huggingface.co/cpral/HomeIMG-1.2B-PL-EN-epoch49.2-DanceOPD2048/resolve/main/grids_cfg3/grid_pl_1024px_ts6.png)

*DanceOPD 2048, eight fixed Polish prompts, 1024 × 1024 pixels, 50 Euler steps, CFG 3 and time shift 6. [Bilingual grids and generation settings](model_cards/HomeIMG-1.2B-PL-EN-epoch49.2-DanceOPD2048/README_en.md).*

## Models and training stages

| Weights | Training stage |
|---|---|
| [641M · epoch19](https://huggingface.co/cpral/HomeIMG-641M-PL-EN-epoch19) | Early bilingual checkpoint. I switched from English-only training to Polish and English captions. |
| [1.2B · epoch49](https://huggingface.co/cpral/HomeIMG-1.2B-PL-EN-epoch49) | After progressive training, I mixed 256, 512 and 1024 px batches. |
| [1.2B · epoch49.2](https://huggingface.co/cpral/HomeIMG-1.2B-PL-EN-epoch49.2) | Common starting point for the DanceOPD and DiffusionOPD branches. |
| [1.2B · DanceOPD 2048](https://huggingface.co/cpral/HomeIMG-1.2B-PL-EN-epoch49.2-DanceOPD2048) | 2048 updates at 1024 px with FLUX.2 [klein] base 4B as the teacher. This checkpoint produced the grid above. |

I later merged the DanceOPD 2048 and DiffusionOPD 3072 denoisers with ArceeFusion. After observing a regression at lower resolutions following 1024 px distillation, I trained the merged weights with DanceOPD for 5120 updates, sampling 256, 512 or 1024 px per rollout batch with an equal image-token budget. [Experiment configuration](docs/distillation.md).

## Generate an image

On a machine with an NVIDIA GPU, a CUDA 12.8-compatible driver and `uv`, run:

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

The example uses a Polish prompt asking for a watercolor of Wawel Castle seen from the Vistula. The output is `samples/00.png`. Omit `--prompt` to generate two eight-prompt grids, one in each language. The browser interface is in [app.py](app.py).

## How HomeIMG works

Frozen Poziomka layers encode the caption. A trainable refiner turns their hidden states into conditioning for the denoiser. The diffusion transformer learns a velocity field in FLUX.2 latent space through flow matching. At inference, an Euler solver follows this field from noise to an image latent, which the VAE decodes.

During on-policy distillation (OPD), a copy of the student generates denoising trajectories. A frozen teacher predicts velocities at selected states and the student learns to match them. Training, rollout generation and teacher inference can run on separate GPUs.

| Task | Entry point |
|---|---|
| Prepare data, train from scratch or start from published weights | [Training guide](docs/training.md) |
| Reproduce multi-resolution DanceOPD and adjust state sampling | [Distillation guide](docs/distillation.md) |
| Measure step time and diagnose communication | [NCCL and profiling](docs/performance.md) |
| Read the implementation | [Model](aurora/model.py) · [DDP training](train.py) · [Distributed training and OPD](train_pp.py) |

## How I compare checkpoints

I generate four fixed seeds for each of [22 Polish prompts](tests/homeimg22_pl.json), for **88 images per checkpoint**. Three prompts describe the same Christmas Eve dinner as a photograph, a watercolor and a CGI film still. They test whether the model preserves the scene while changing the medium. [Settings and grid instructions](docs/evaluation.md).

In my visual review of DanceOPD 2048, CFG 3 produced clearer shapes than CFG 1. Counting, legible text and anatomy in crowds remained difficult. [Samples and limitations](model_cards/HomeIMG-1.2B-PL-EN-epoch49.2-DanceOPD2048/README_en.md).

## Author and components

I develop HomeIMG as [adamo1139](https://github.com/adamo1139). The DiT blocks are based on [AuroraIMG2-50M-Preview](https://huggingface.co/AuroraAI-Research/AuroraIMG2-50M-Preview) (Apache-2.0). HomeIMG also uses the [FLUX.2 decoder](https://huggingface.co/black-forest-labs/FLUX.2-small-decoder) (Apache-2.0), [Poziomka](https://huggingface.co/cpral/Poziomka-Baza-2026-09-05) and the [Muon optimizer](https://kellerjordan.github.io/posts/muon/). Consult each model and dataset card for its terms of use.
