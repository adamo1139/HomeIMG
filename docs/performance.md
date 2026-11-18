# Performance and GPU communication

[Polski](wydajnosc.md)

## NCCL on GeForce cards

Check communication independently of model loading and data preparation:

```bash
torchrun --nproc_per_node 8 tests/check_nccl.py
```

On a driver that supports GPU peer-to-peer access (for example, patched open GPU kernel modules), NCCL may limit P2P to nearby PCIe devices. Use `NCCL_P2P_LEVEL=SYS` if the path crosses the host. Check support with `python -c "import torch; print(torch.cuda.can_device_access_peer(0, 1))"`.

For errors attaching to `/dev/shm/nccl-...`, inspect the selected transport with `NCCL_DEBUG=INFO` (P2P, SHM or NET). If P2P works, try the SYS setting above. Otherwise check space with `df -h /dev/shm` and permissions (`drwxrwxrwt`). As a fallback, `NCCL_SHM_DISABLE=1` enables slower socket transport when P2P is unavailable. Set these variables before launching torchrun.

## Profile attention, GEMM and the training loop

`tests/profile_dit.py` measures self-, cross- and refiner attention, forward/backward GEMMs and the full DiT block. With `--init`, it reads checkpoint dimensions and first-block weights without loading Poziomka. Inputs are synthetic. Single-block memory excludes the rest of the model. Run on an idle GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python tests/profile_dit.py \
  --init runs/poziomka_stage2/merged.pt --depth 12 \
  --parts attention gemm block --mb 8 16 --text-widths 64 128 \
  --out runs/profile_dit

CUDA_VISIBLE_DEVICES=0 python tests/profile_dit.py \
  --init runs/poziomka_stage2/merged.pt --parts block \
  --mb 8 16 24 --text-widths 64 128 256 --checkpoints none selective \
  --compile-modes default max-autotune-no-cudagraphs --out runs/profile_blocks
```

The second command tunes the entire block, excluding compilation and autotuning from measurements. See `results.json` and hardware/model `metadata.json`. Add `--trace` for a timeline, operator table and kernel shapes per case. Unsupported backends appear as errors. Attention outputs and gradients are compared with math SDPA. The varlen benchmark includes K/V packing with `torch.nn.attention.varlen`. Use `--parts block --block-backends auto cudnn` to compare cross-attention within the actual block including MLP, projections, norms and checkpointing. FLOPs use actual text width and exclude optimizer and checkpoint recomputation.

Append these options to a training command:

```bash
--profile-dir runs/profile_train --profile-start 17 --profile-steps 4 --profile-ranks 0
```

The profiler also supports `--bench`. For two pipeline stages select `--profile-ranks 0 1`. Training-rank traces cover data loading, LM request and collection, forward/backward, gradient synchronization and clipping, optimizer, outer sync and EMA. Nested operators and overlapping kernels make summed operator time different from elapsed time. Warm up all text widths before profiling and exclude trace capture/export when comparing images per second. `opt avg` averages the full logging window, including periodic Muon steps.

`--masked-attention-backend cudnn` selects cuDNN for DiT cross-attention and trainable refiner attention. The default is `auto`. DiT self-attention and the frozen LM use their own backends. The selection is part of the compiled graph; unsupported shapes fail explicitly. Compare whole steps at the same microbatch size and checkpoint settings.

## Eight-GPU configurations for a global batch of 1008

| Layout | Options | Batch decomposition |
|---|---|---|
| Seven replicas, one LM server | `--pp 1 --lm-rank 1 --microbatches 9` | 7 × 9 × 16 |
| Six replicas, two LM servers | `--pp 1 --lm-rank 2 --microbatches 7` | 6 × 7 × 24 |
| Eight replicas, local LM | `--pp 1 --microbatches 9` | 8 × 9 × 14 |
| Four two-stage pipelines, distributed LM | `--pp 2 --microbatches 18` | 4 × 18 × 14 |
| Three two-stage pipelines, two LM servers | `--pp 2 --lm-rank 2 --microbatches 14` | 3 × 14 × 24 |

These are valid layouts to benchmark. On eight GPUs, one LM server leaves seven training ranks, which cannot form two-stage pipelines. Replica count affects the local DiLoCo batch and automatic AdamW learning-rate scaling. Keep machine-specific NCCL settings consistent. A local LM shares GPU memory and compute with the denoiser; pipeline parallelism introduces communication and idle slots.
