"""Composite four dashboard panel screenshots into a 2x2 grid for the README.

Usage:
    uv run python scripts/composite_dashboard.py \
        docs/_tab1.png docs/_tab2.png docs/_tab3.png docs/_tab4.png \
        docs/dashboard.png

Adapted from `llm-cost-autopilot/scripts/composite_dashboard.py`. The
crop on the left strips Streamlit's sidebar so the composite focuses on
panel content.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TILE_WIDTH = 1200
GAP = 8
BORDER = 4
LABEL_HEIGHT = 36
# Guardian's Streamlit dashboard has no sidebar, so we do not crop. The
# autopilot version uses CROP_LEFT=290 to strip its sidebar.
CROP_LEFT = 0
BG = (255, 255, 255)
SEP = (220, 220, 220)
LABEL_FG = (40, 40, 40)


def _load_font() -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, 22)
            except OSError:
                continue
    return ImageFont.load_default()


def _tile(img_path: str, label: str, font: ImageFont.ImageFont) -> Image.Image:
    src = Image.open(img_path).convert("RGB")
    if src.width > CROP_LEFT + 100:
        src = src.crop((CROP_LEFT, 0, src.width, src.height))
    scale = TILE_WIDTH / src.width
    panel = src.resize((TILE_WIDTH, int(src.height * scale)), Image.LANCZOS)

    tile = Image.new("RGB", (TILE_WIDTH, LABEL_HEIGHT + panel.height), BG)
    draw = ImageDraw.Draw(tile)
    draw.text((16, 6), label, fill=LABEL_FG, font=font)
    draw.line(
        [(0, LABEL_HEIGHT - 1), (TILE_WIDTH, LABEL_HEIGHT - 1)],
        fill=SEP,
        width=1,
    )
    tile.paste(panel, (0, LABEL_HEIGHT))
    return tile


def main(paths: list[str], labels: list[str], out_path: str) -> None:
    if len(paths) != 4:
        raise SystemExit(f"expected 4 input images, got {len(paths)}")
    font = _load_font()
    tiles = [_tile(p, lbl, font) for p, lbl in zip(paths, labels, strict=True)]

    target_h = max(t.height for t in tiles)
    normed = []
    for t in tiles:
        if t.height == target_h:
            normed.append(t)
            continue
        padded = Image.new("RGB", (TILE_WIDTH, target_h), BG)
        padded.paste(t, (0, 0))
        normed.append(padded)

    final_w = TILE_WIDTH * 2 + GAP + BORDER * 2
    final_h = target_h * 2 + GAP + BORDER * 2
    final = Image.new("RGB", (final_w, final_h), SEP)
    final.paste(normed[0], (BORDER, BORDER))
    final.paste(normed[1], (BORDER + TILE_WIDTH + GAP, BORDER))
    final.paste(normed[2], (BORDER, BORDER + target_h + GAP))
    final.paste(normed[3], (BORDER + TILE_WIDTH + GAP, BORDER + target_h + GAP))

    final.save(out_path, optimize=True)
    print(f"wrote {out_path} ({final_w}x{final_h})")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 5:
        raise SystemExit(
            "usage: composite_dashboard.py panel1.png panel2.png panel3.png panel4.png out.png"
        )
    labels = [
        "PII protection stats",
        "Redaction explorer",
        "Eval results",
        "Redaction detail",
    ]
    main(args[:4], labels, args[4])
