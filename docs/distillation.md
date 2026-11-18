# Distill HomeIMG

[Polski](destylacja.md)

## Multi-resolution DanceOPD and DiffusionOPD

In `train_pp.py`, set `--res 1024 --t-shift 6`, `--opd-resolutions 256 512 1024` and `--opd-res-shifts 1.5 3 6`. Each fresh trajectory batch draws one resolution and retains it across all replay updates. Ordinary flow-matching training has a separate `--multi-res` mode. By default, image batch size stays fixed across resolutions.

With `--opd-token-batch`, the image count scales by `round((res_max / res)^2)`. A base batch of 50 gives 800/200/50 images at 256/512/1024 px, or 204,800 image tokens per update with one state per trajectory. Extra microbatches cover the larger batches. The base learning rate stays fixed. More captions increase encoder and refiner work, so equal image-token counts can yield different step times. The checkpoint tracks consumed data separately from rollout count so speculative prefetch can be rebuilt on resume.

Resolution weights of 1/1/1 assign equal probabilities to fresh batches. At a fixed image count, expected token shares are 1:4:16; token balancing gives 1:1:1. Any positive weights are supported. Without explicit per-resolution shifts, the schedule uses `res / --t-shift-ref` if a reference is set, otherwise `--t-shift * res / --res`.

All replicas use the seed and target batch step to select the same resolution. Local rollouts, `--opd-prefetch` and separate actors support this schedule. Cached batches carry their resolution and time shift. The teacher receives the latent shape and the student derives RoPE and micro-conditioning from its input. Logs report `opd/resolution` and `opd/time_shift`. The FLOP estimate covers training work at the actual resolutions and state counts. Use `--pp 1` and disable REPA; omit `--multi-res` and `--crop-res`. Each new shape may trigger compilation. CPU check: `PYTHONPATH=. python tests/check_opd_resolution.py`.

## Prefetch frozen text states

`--opd-text-ahead 4` prepares four future text batches per learner in addition to the one being consumed. A worker reads captions in schedule order. A dedicated GPU runs frozen Poziomka layers before the trainable refiner and the teacher's Qwen encoder. BF16 states wait in a bounded CPU queue. Logs report MiB per batch. After a restart, the queue is rebuilt from the last consumed data position.

This mode requires actors, one LM server and a dedicated teacher. Use a separate teacher GPU instead of `--teacher-on-lm`. With the mapping below, Qwen runs on GPU 6 alongside Poziomka. The actor receives a weight version only when its rollout job is submitted, so text lookahead can be increased independently of policy age. Measure RAM and rollout readiness separately. CPU protocol check with four Gloo processes: `PYTHONPATH=.:tests python tests/check_opd_text_queue.py`.

## Seven-GPU recipe for HomeIMG 1.2B

Run from the repository root after [importing a Poziomka checkpoint](training.md). For the 5120-update experiment I initialized from ArceeFusion weights. The command expects the initial checkpoint at `runs/homeimg-init.pt`, a local Klein model and FLUX-Reason 1024 shards with PL/EN captions and Poziomka tokens. The small-model recipe uses a separate 256 px latent dataset.

On an eight-card machine I exclude GPU 0. Learners use GPUs 1, 3 and 5, actors use 2 and 7, the image teacher uses 4 and both text encoders use 6. Global batches of 480/120/30 contain 122,880 image tokens each. I reuse each trajectory batch for twelve updates.

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
