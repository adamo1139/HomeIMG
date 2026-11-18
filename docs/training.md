# Train HomeIMG

[Polski](trening.md)

Run commands from the repository root. The first recipe trains a small DiT on eight GPUs using precomputed 256 px latents. The default 10% subset takes about 39 GB. Allow additional space for conversion and checkpoints.

## Environment

Install [uv](https://docs.astral.sh/uv/) and create a Python 3.11 environment. Install the CUDA wheels before the remaining dependencies.

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.device_count(), 'GPU')"
```

The tested environment uses PyTorch 2.11.0+cu128 and Diffusers 0.42.1. Alternatively use `python3.11 -m venv .venv`, activate it and install the same packages with pip.

## Caption conditioning and REPA

The default `--te-size slayer149` combines frozen [Slayer149](https://huggingface.co/SlayerLab/Slayer149) (149M, English) with a trainable refiner. By default it has two blocks at half the denoiser width. Override with `--te-dim` and `--te-depth`; older runs may require `--te-dim 768 --te-depth 2`. Create Slayer tokens alongside the original token files. The older `13m` and `27m` encoders train from scratch with an auxiliary masked-language loss.

REPA aligns intermediate DiT features around 30% depth with [DINOv2-B/14 register features](https://huggingface.co/timm/vit_base_patch14_reg4_dinov2.lvd142m) from decoded latents resized to 224 px. The default weight is 0.5. To limit decoding cost, it uses 25% of each microbatch during the first 40% of training. Adjust `--repa-frac`, `--repa-every` and `--repa-until`, or disable with `--repa-weight 0`. Other options include `--irepa` for convolutional projection and token-wise normalization, and `--repa-encoder vit_pe_spatial_base_patch16_512.fb` for PE-Spatial-B16. Export packages the Slayer layers and tokenizer; `tests/check_export.py` checks parity with the training modules.

## Train from scratch

Download the reference model, convert the shards to memory-mapped arrays, tokenize captions and train. This example uses the 116M preset, global batch 2048 and approximately 3098 updates on 10% of the dataset. Pass `--fraction 1` to conversion and training for the full run. Reduce `--micro-batch` if memory is tight; accumulation adjusts automatically. Learning rate and EMA scale with batch size.

```bash
source .venv/bin/activate
hf download AuroraAI-Research/AuroraIMG2-50M-Preview --local-dir AuroraIMG2-50M-Preview

python prepare_data.py convert  --out data/aurora --download --delete-raw --workers 6
python prepare_data.py tokenize --data data/aurora --encoder slayer149

torchrun --nproc_per_node 8 train.py --data data/aurora --out runs/aurora --size 116m

python merge.py runs/aurora                                  # -> runs/aurora/merged.pt
python export.py runs/aurora/merged.pt exports/aurora
python sample.py exports/aurora --out grid.png
```

The trainer resumes from the output directory. By default it refreshes `latest.pt` every 1/50 of the run and retains snapshots every 1/10. W&B logging activates when logged in; use `--no-wandb` to disable it. The final commands average late checkpoints, export weights and generate a sample grid.

## Pipeline parallelism

`train_pp.py` partitions DiT blocks across `--pp` GPUs using PyTorch's 1F1B schedule. Stage 0 hosts the text refiner, timestep projection and patch input. The final stage computes the output and loss. Stages exchange tokens, modulation, text, masks and gradients. The stage containing the chosen block computes REPA loss; decoding and DINO target computation are distributed across microbatches. Frozen LM work can also be distributed across stages. The partitioner accounts for the refiner on stage 0 through `--te-cost`.

The 1b, 1.5b and 2b presets use width 2048 with 15, 23 or 31 blocks. The 1.6b and 2.4b presets use width 2560, head dimension 128 and 15 or 23 blocks. For example, the 2b model splits into 3 blocks on stage 0 and 4 on each remaining stage for eight GPUs.

```bash
NCCL_P2P_LEVEL=SYS torchrun --nproc_per_node 8 train_pp.py --data data/aurora_full --fraction 1 --size 1.5b \
  --pp 8 --batch 512 --microbatches 32 --out runs/1p5b
```

Use `--act-checkpoint selective` to recompute elementwise operations or `--act-checkpoint` for full checkpointing. On eight GPUs, `--pp 4` creates two replicated pipelines; `--diloco H` sets their averaging interval. Rank 0 gathers weights in the standard checkpoint format. Optimizer state is saved per stage. The CPU/Gloo test `tests/check_pp.py` compares losses and gradients with the unpartitioned model on one, two or four processes.

## Continue from published weights

For a bilingual Poziomka export:

```bash
source .venv/bin/activate
hf download cpral/HomeIMG-1.2B-PL-EN-epoch49.2-DanceOPD2048 --local-dir exports/homeimg
python import_homeimg.py exports/homeimg runs/homeimg-init.pt
```

Pass `--init runs/homeimg-init.pt` and a fresh `--out` to `train_pp.py`. Import preserves denoiser and refiner weights and starts at step zero with a new optimizer. Prepare Poziomka tokens for the captions. See the [distillation recipe](distillation.md) for a complete launch command.

To resume an interrupted training run, use its original output directory and keep `latest.pt` together with `opt_stage*.pt`. Export its EMA weights with:

```bash
source .venv/bin/activate
python export.py runs/homeimg/latest.pt exports/homeimg-ema --time-shift 6 --time-shift-ref 0
```
