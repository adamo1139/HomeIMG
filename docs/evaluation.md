# Evaluate image quality

[Polski](ewaluacja.md)

## Polish 22-prompt comparison

I generate each of the [22 Polish prompts](../tests/homeimg22_pl.json) with seeds **0, 1, 2 and 3** at **1024 × 1024, CFG 3, time shift 6 and 50 steps**. This gives 88 images per model or 264 across three checkpoints. I use the exported pipeline defaults for the remaining options and leave the negative prompt empty. [Generation settings](../tests/homeimg22_settings.json).

The original comparison uses the `cpral/HomeIMG-1.2B-PL-EN-epoch49.2` base model and its `-DanceOPD2048` and `-DiffusionOPD3072` releases. Run from the repository root with either a local Diffusers export or a Hugging Face model ID:

```bash
source .venv/bin/activate
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python tests/homeimg22.py cpral/HomeIMG-1.2B-PL-EN-epoch49.2 --out-dir exports/test22_epoch49.2
```

Change both the model and output directory for each checkpoint. The script generates images individually with a seeded CUDA generator. Split jobs with `--part K --parts N` to preserve seeds and batch size across GPUs. Use `--dry-run` to print jobs without loading the model. Each run writes PNGs and a manifest with prompts, seeds and parameters, and protects existing outputs. Library and hardware differences can affect results even with identical seeds.

For blind evaluation, display each image with its full Polish prompt directly underneath. Keep model names, seeds and manifests hidden from reviewers. Once generation is complete, assemble the grids:

```bash
source .venv/bin/activate
python tests/homeimg22_grid.py exports/test22_epoch49.2
```
