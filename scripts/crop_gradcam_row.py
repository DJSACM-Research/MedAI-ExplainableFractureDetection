"""Extract a single row from the Grad-CAM comparison grid PNG.

Crops the first row (Comminuted fracture case) from the existing
4-row grid to produce a compact single-row figure for the paper.

Usage:
    python scripts/crop_gradcam_row.py
"""

from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IN_PATH = ROOT / "outputs" / "figures" / "gradcam_comparison_grid.png"
OUT_DIR = ROOT / "outputs" / "figures"

img = Image.open(IN_PATH)
w, h = img.size

# The grid has 4 image rows plus a suptitle band at the top.
# We want just the first row (below the suptitle).
# Strategy: detect the title band height, then take 1/4 of the remaining.
# The suptitle + spacing is roughly 5-8% of total height.
# We'll crop from y=0 to y = h/4 + a small margin for the column headers.
row_h = h // 4
# Take the first ~33% of the height to capture column titles + first image row
crop_bottom = int(h * 0.325)

cropped = img.crop((0, 0, w, crop_bottom))
for ext, fmt in [("png", "PNG"), ("pdf", "PDF")]:
    out = OUT_DIR / f"gradcam_single_row.{ext}"
    cropped.save(out, fmt, dpi=(300, 300))
    print(f"Saved: {out}  ({cropped.size[0]}x{cropped.size[1]})")
