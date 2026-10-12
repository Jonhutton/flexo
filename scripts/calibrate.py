#!/usr/bin/env python
"""Tune card detection against a real Arena pick screen. No API key, no cost.

    python scripts/calibrate.py                    # grab the live Arena window
    python scripts/calibrate.py --image shot.png   # use a saved screenshot
    python scripts/calibrate.py --image shot.png --dump-crops

Writes debug/calibration.png with a numbered box per detected card, and prints
a suggested `vision.pack_region` for config.yaml. You want exactly one box per
card on screen.

If the count is wrong:
  too few boxes    lower vision.detection.min_area_frac (try 0.002), or widen
                   aspect_range to [0.55, 0.90]
  extra boxes      raise min_area_frac, or set pack_region so the UI chrome
                   around the pack falls outside the frame
  nothing works    set vision.detection.mode to "grid" and tune detection.grid
                   to your layout

--dump-crops writes each card crop to debug/crops/. Open them: that is exactly
what the vision model is asked to read, so if a name is illegible there, the
fix is detection or resolution, not the prompt.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from PIL import Image

from flexo.detect import crop_cards, crop_fraction, draw_boxes, find_cards, name_banner

ROOT = Path(__file__).resolve().parents[1]


def suggest_pack_region(boxes, frame_w: int, frame_h: int, pad: float = 0.02):
    """Bounding box of every detected card, padded, as [x, y, w, h] fractions.

    Narrowing change detection to this region is what stops a mouse hover over
    the sidebar from reading as a freshly dealt pack.
    """
    if not boxes:
        return None
    x0 = min(b.x for b in boxes) / frame_w - pad
    y0 = min(b.y for b in boxes) / frame_h - pad
    x1 = max(b.x + b.w for b in boxes) / frame_w + pad
    y1 = max(b.y + b.h for b in boxes) / frame_h + pad
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(1.0, x1), min(1.0, y1)
    return [round(x0, 3), round(y0, 3), round(x1 - x0, 3), round(y1 - y0, 3)]


def main() -> int:
    ap = argparse.ArgumentParser(description="Flexo card-detection calibration")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--image", help="screenshot to analyze instead of the live screen")
    ap.add_argument("--dump-crops", action="store_true",
                    help="also write each card crop (and its title bar) to debug/crops/")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    vcfg = cfg.get("vision", {}) or {}

    if args.image:
        img = Image.open(args.image).convert("RGB")
        source = args.image
    else:
        from flexo.capture import ScreenCapture
        cap = ScreenCapture(ROOT / (vcfg.get("captures_dir") or "captures"))
        img = cap._grab()
        if img is None:
            print("Could not grab the screen. Is Arena running and not minimized?")
            return 1
        source = "live screen"

    debug_dir = ROOT / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    img.save(debug_dir / "calibration_raw.png")

    frame = crop_fraction(img, vcfg.get("pack_region"))
    boxes, mode = find_cards(frame, vcfg)

    print(f"source      : {source}  ({img.width}x{img.height})")
    print(f"pack_region : {vcfg.get('pack_region')}  -> frame {frame.width}x{frame.height}")
    print(f"detection   : mode={mode}, {len(boxes)} box(es) found")
    if boxes:
        widths = [b.w for b in boxes]
        heights = [b.h for b in boxes]
        print(f"card size   : {min(widths)}-{max(widths)}w x "
              f"{min(heights)}-{max(heights)}h px "
              f"(aspect {boxes[0].w / boxes[0].h:.3f})")
        if min(widths) < 200:
            print("  NOTE: crops are narrow - card names will be hard to read. "
                  "Raise vision.upscale_min_width, or run Arena at a higher "
                  "resolution.")

    out = debug_dir / "calibration.png"
    draw_boxes(frame, boxes, [str(i + 1) for i in range(len(boxes))]).save(out)
    print(f"\nwrote {out.relative_to(ROOT)} - open it and count the boxes.")

    region = suggest_pack_region(boxes, frame.width, frame.height)
    if region and not vcfg.get("pack_region"):
        print("\nsuggested config.yaml:\n  vision:\n    pack_region: "
              f"{region}")

    if args.dump_crops and boxes:
        crop_dir = debug_dir / "crops"
        crop_dir.mkdir(parents=True, exist_ok=True)
        for i, crop in enumerate(crop_cards(frame, boxes), start=1):
            crop.save(crop_dir / f"card_{i:02d}.png")
            name_banner(crop, float(vcfg.get("name_banner_frac", 0.14))).save(
                crop_dir / f"card_{i:02d}_banner.png")
        print(f"wrote {len(boxes)} crop(s) + title bars to "
              f"{crop_dir.relative_to(ROOT)}")
        print("  Check a *_banner.png: if the card NAME isn't centered in it, "
              "adjust the frac in detect.name_banner().")

    return 0 if boxes else 1


if __name__ == "__main__":
    sys.exit(main())
