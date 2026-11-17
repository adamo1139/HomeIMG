"""Assemble 88 existing PNGs into a four-column JPEG; no model or GPU needed.

Rows follow prompt order, columns follow seeds 0/1/2/3. Only prompt text is
printed below each image. Source PNGs remain unchanged.
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def wrap(text, font, width):
    lines = []
    line = ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if line and font.getlength(candidate) > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--tile-size", type=int, default=512,
                        help="display size; use 1024 to preserve original pixel dimensions")
    parser.add_argument("--font", default="DejaVuSans.ttf")
    args = parser.parse_args()
    if not 256 <= args.tile_size <= 1024:
        parser.error("--tile-size must be between 256 and 1024")
    manifests = sorted(args.directory.glob("manifest_part_*.json"))
    if not manifests:
        parser.error("no generation manifests found")
    jobs = {}
    reference = json.loads(manifests[0].read_text())
    for path in manifests:
        data = json.loads(path.read_text())
        if any(data[key] != reference[key] for key in ("model", "settings", "seeds", "parts")):
            parser.error(f"inconsistent manifest: {path}")
        for job in data["images"]:
            key = (job["prompt_id"], job["seed"])
            if key in jobs:
                parser.error(f"duplicate image: {key}")
            jobs[key] = job
    seeds = reference["seeds"]
    expected = {(p, s) for p in range(1, 23) for s in (0, 1, 2, 3)}
    if seeds != [0, 1, 2, 3] or set(jobs) != expected:
        parser.error("expected all 22 prompts with seeds 0, 1, 2, 3")
    for job in jobs.values():
        path = args.directory / job["file"]
        if not path.is_file():
            parser.error(f"missing image: {path}")

    size = args.tile_size
    padding = max(12, size // 40)
    font = ImageFont.truetype(args.font, max(18, size // 28))
    ascent, descent = font.getmetrics()
    line_height = ascent + descent + 4
    rows = []
    for prompt_id in range(1, 23):
        prompt = jobs[prompt_id, 0]["prompt"]
        if any(jobs[prompt_id, seed]["prompt"] != prompt for seed in seeds):
            parser.error(f"inconsistent text for prompt {prompt_id}")
        lines = wrap(prompt, font, size - 2 * padding)
        rows.append((prompt_id, lines, size + 2 * padding + len(lines) * line_height))
    grid = Image.new("RGB", (4 * size, sum(row[2] for row in rows)), "white")
    draw = ImageDraw.Draw(grid)
    y = 0
    for prompt_id, lines, height in rows:
        for column, seed in enumerate(seeds):
            path = args.directory / jobs[prompt_id, seed]["file"]
            with Image.open(path) as source:
                if source.size != (1024, 1024):
                    parser.error(f"expected 1024x1024 image: {path}")
                tile = source.convert("RGB")
                if size != 1024:
                    tile = tile.resize((size, size), Image.Resampling.LANCZOS)
                grid.paste(tile, (column * size, y))
            for index, line in enumerate(lines):
                draw.text((column * size + padding, y + size + padding + index * line_height),
                          line, font=font, fill="black", anchor="lt")
        y += height
    output = args.directory / "grid.jpg"
    temporary = args.directory / "grid.jpg.tmp"
    grid.save(temporary, format="JPEG", quality=95, subsampling=0)
    temporary.replace(output)
    print(f"Saved {output}: {grid.width} x {grid.height}; 22 rows x 4 seeds")


if __name__ == "__main__":
    main()
