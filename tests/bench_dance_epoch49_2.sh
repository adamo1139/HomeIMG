#!/usr/bin/env bash
# Run on the training server, with its Python venv active.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
unset CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_LEVEL=SYS
bench_tag="$(date +%Y%m%d_%H%M%S)_$$"
torchrun --nproc_per_node 7 train_pp.py \
  --data data/flux_reason_1024 --img-dir data/flux_reason_1024 \
  --fraction 1 --size 750m --depth 12 \
  --res 1024 --rope-ref 64 --micro-cond 256 --t-shift 6 \
  --lm-rank 1 --lm-gpu 1 --skip-gpus 0 \
  --dance-opd --dance-beta 1 1 --opd-states 1 --opd-rollout-steps 16 \
  --teacher-ranks 2 --teacher-on-lm --klein-quant int8 \
  --klein /media/nvme_2tb/maked/FLUX.2-klein-base-4B --opd-cfg 4 \
  --muon-momentum-dtype bf16 --te-size poziomka --te-quant int8 --pl-prob 0.5 \
  --init runs/poziomka_multires2/latest.pt \
  --batch 48 --microbatches 6 \
  --diloco 64 --outer-lr 1 --outer-momentum 0 \
  --lr-muon 1e-3 --cmuon --muon-period 1 \
  --repa-weight 0 --ema-cpu --seed 7 --log-every 1 \
  --steps 12 --bench --mem-debug --out "runs/bench_dance49_2_mb2_p1_${bench_tag}" \
  --pp 1 --act-checkpoint selective --enc-chunk 2 --workers 1 \
  2>&1 | tee "bench_dance49_2_mb2_p1_${bench_tag}.log"
