"""Generate the fixed Polish HomeIMG comparison suite from a Diffusers export.

22 prompts x seeds 0, 1, 2, 3; 1024 px, CFG 3, time shift 6, 50 steps.
Each image is generated separately with a reset CUDA RNG, regardless of part.
Use a separate output directory per model. No generation is started by --dry-run.
"""
import argparse
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from homeimg_loader import load_homeimg_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Self-contained Diffusers export directory or HF repo")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="resume an identical manifest, skipping valid existing PNGs")
    args = parser.parse_args()
    if args.parts < 1 or not 0 <= args.part < args.parts:
        parser.error("require --parts > 0 and 0 <= --part < --parts")

    prompts = json.loads((HERE / "homeimg22_pl.json").read_text())
    settings = json.loads((HERE / "homeimg22_settings.json").read_text())
    seeds = settings.pop("seeds")
    jobs = [dict(prompt_id=i + 1, prompt=prompt, seed=seed,
                 file=f"prompt_{i + 1:02d}_seed_{seed}.png")
            for i, prompt in enumerate(prompts) for seed in seeds]
    selected = jobs[args.part::args.parts]
    manifest = dict(model=args.model, settings=settings, seeds=seeds,
                    part=args.part, parts=args.parts, images=selected)
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Refuse to mix models/configurations or silently overwrite an earlier run.
    manifest_path = args.out_dir / f"manifest_part_{args.part:03d}.json"
    if not args.resume and (manifest_path.exists() or any((args.out_dir / job["file"]).exists() for job in selected)):
        parser.error("output already exists; choose a new output directory")
    if args.resume:
        if not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest:
            parser.error("--resume requires an identical existing manifest")
        from PIL import Image
        pending = []
        for job in selected:
            path = args.out_dir / job["file"]
            if path.exists():
                try:
                    with Image.open(path) as image:
                        if image.size != (settings["resolution"], settings["resolution"]):
                            raise ValueError("wrong image dimensions")
                        image.verify()
                except (OSError, ValueError) as error:
                    parser.error(f"invalid existing image {path}: {error}")
            else:
                pending.append(job)
        selected = pending
    for other in args.out_dir.glob("manifest_part_*.json"):
        previous = json.loads(other.read_text())
        for key in ("model", "settings", "seeds", "parts"):
            if previous[key] != manifest[key]:
                parser.error(f"output directory contains a different {key}")
    if not args.resume:
        with manifest_path.open("x") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
    if not selected:
        print("All images already exist.")
        return

    import torch
    from huggingface_hub import snapshot_download

    root = args.model if Path(args.model).is_dir() else snapshot_download(args.model)
    pipe = load_homeimg_pipeline(root)
    pipe.to("cuda")
    pipe.vae.to(torch.bfloat16)
    with torch.inference_mode():
        for number, job in enumerate(selected, 1):
            generator = torch.Generator(device="cuda").manual_seed(job["seed"])
            image = pipe(job["prompt"], generator=generator, **settings).images[0]
            image.save(args.out_dir / job["file"])
            print(f"{number}/{len(selected)} saved {job['file']}", flush=True)


if __name__ == "__main__":
    main()
