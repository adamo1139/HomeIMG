"""Convert a Poziomka HomeIMG export to a fresh train_pp.py --init checkpoint.

No optimizer history is transferred. Frozen LM weights stay in their source
repository; only the trained refiner belongs in a training checkpoint.
"""
import argparse
from dataclasses import asdict, fields, replace
import json
from pathlib import Path

import torch
from safetensors import safe_open

from aurora.model import Denoiser, DenoiserConfig, TE_SIZES
from aurora.slayer import LMTextEncoder


def read_weights(path, exclude_lm=False):
    with safe_open(path, framework="pt", device="cpu") as f:
        return {k: f.get_tensor(k).float() for k in f.keys()
                if not (exclude_lm and k.startswith("lm."))}


def convert(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    dc = json.loads((root / "transformer/config.json").read_text())
    tc = json.loads((root / "text_encoder/config.json").read_text())
    if tc.get("lm_arch") != "poziomka":
        raise ValueError("This converter supports Poziomka HomeIMG exports only")
    den_cfg = DenoiserConfig(**{f.name: dc[f.name] for f in fields(DenoiserConfig)})
    te_cfg = replace(TE_SIZES["poziomka"],
                     **{k: tc[k] for k in ("lm_dim", "dim", "depth", "heads", "kv_heads",
                                          "head_dim", "mlp_hidden", "max_tokens", "out_dim")},
                     experts=tc["lm_top_k"], quant="int8")
    if tc["lm_blocks"] != 16:
        raise ValueError("Expected all 16 Poziomka LM layers")
    if (te_cfg.out_dim or te_cfg.dim) != den_cfg.text_dim:
        raise ValueError("Text encoder output dimension differs from denoiser text_dim")
    # Validate against the actual training architecture without loading the frozen LM.
    with torch.device("meta"):
        den = Denoiser(den_cfg)
        te = LMTextEncoder(te_cfg, with_lm=False)
    den.load_state_dict(read_weights(root / "transformer/diffusion_pytorch_model.safetensors"),
                        strict=True, assign=True)
    te.load_state_dict(read_weights(root / "text_encoder/diffusion_pytorch_model.safetensors", True),
                       strict=True, assign=True)
    den_sd, te_sd = dict(den.named_parameters()), dict(te.named_parameters())
    den_sd = {k: v.detach() for k, v in den_sd.items()}
    te_sd = {k: v.detach() for k, v in te_sd.items()}
    for name, tensor in list(te_sd.items()) + list(den_sd.items()):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite weights: {name}")
    # Shared tensor storage avoids duplicating several GB while preserving EMA order.
    ck = dict(step=0, opd_batches=0, den_config=asdict(den_cfg), te_config=asdict(te_cfg),
              den=den_sd, te=te_sd, ema=list(te_sd.values()) + list(den_sd.values()), opts=None,
              args={"t_shift": dc.get("time_shift", 1.0),
                    "t_shift_ref": dc.get("time_shift_ref", 0)},
              import_source=str(root), checkpoint_kind="fresh_init_from_export")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("xb") as stream:
        torch.save(ck, stream)
    temporary.replace(output)
    print(f"Saved {output}: {len(den_sd)} denoiser + {len(te_sd)} refiner tensors; step 0, EMA = weights")
    print("Use --init with a NEW --out directory; optimizer state starts fresh.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Local HomeIMG export directory")
    parser.add_argument("output", help="New .pt checkpoint")
    args = parser.parse_args()
    convert(args.model, args.output)
