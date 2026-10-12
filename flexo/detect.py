"""Find the individual card rectangles on the Arena pick screen.

This is what lets Flexo read cards ONE AT A TIME instead of squinting at a
whole 1080p screenshot. A card name is ~9px tall at 1080p; the same name in a
cropped, upscaled card is comfortably legible. Cropping is the single biggest
accuracy win in the vision path.

Why contours instead of a fixed grid: Arena re-flows the pack as you pick
(14 -> 13 -> ... -> 1) and the layout shifts with resolution and window size.
Card frames are high-contrast rectangles with a known aspect ratio (~0.716),
so edge detection + contour filtering finds them regardless of count or
position. A fixed grid stays as a fallback for odd resolutions where auto
detection comes up empty.

Fail-soft: find_cards() always returns a list. An empty auto pass falls back
to the grid rather than raising into the live loop.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image

# MTG cards are 63x88mm -> 0.716. Arena's frame plus a little detection slop.
DEFAULT_ASPECT_RANGE = (0.60, 0.85)


@dataclass(frozen=True)
class Box:
    """One detected card rectangle, in pixels of the frame it was found in."""
    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    def iou(self, other: "Box") -> float:
        """Intersection-over-union with another box, 0.0..1.0."""
        ix = max(0, min(self.x + self.w, other.x + other.w) - max(self.x, other.x))
        iy = max(0, min(self.y + self.h, other.y + other.h) - max(self.y, other.y))
        inter = ix * iy
        if not inter:
            return 0.0
        return inter / float(self.area + other.area - inter)

    def as_tuple(self) -> tuple[int, int, int, int]:
        """(left, top, right, bottom) — PIL crop order."""
        return (self.x, self.y, self.x + self.w, self.y + self.h)


# --------------------------------------------------------------------------- helpers
def crop_fraction(img: Image.Image, region: Optional[list]) -> Image.Image:
    """Crop by [x, y, w, h] fractions of the image. None/empty -> unchanged.

    Used to narrow both change detection and card search to the pack area, so
    hovering the sidebar doesn't look like a new pack.
    """
    if not region:
        return img
    x, y, w, h = region
    width, height = img.size
    return img.crop((int(x * width), int(y * height),
                     int((x + w) * width), int((y + h) * height)))


def _sort_reading_order(boxes: list[Box]) -> list[Box]:
    """Sort into rows (top to bottom) then left to right, tolerant of y jitter.

    Slot numbers must be stable between the crop list and the model's reply,
    so this ordering is the contract between detect and vision.
    """
    if not boxes:
        return []
    median_h = float(np.median([b.h for b in boxes]))
    rows: list[list[Box]] = []
    for b in sorted(boxes, key=lambda b: b.cy):
        if rows and abs(rows[-1][0].cy - b.cy) < median_h * 0.5:
            rows[-1].append(b)
        else:
            rows.append([b])
    out: list[Box] = []
    for row in rows:
        out.extend(sorted(row, key=lambda b: b.cx))
    return out


def _dedupe(boxes: list[Box], iou_threshold: float = 0.4) -> list[Box]:
    """Drop overlapping detections, keeping the largest of each cluster."""
    kept: list[Box] = []
    for b in sorted(boxes, key=lambda b: -b.area):
        if all(b.iou(k) < iou_threshold for k in kept):
            kept.append(b)
    return kept


# --------------------------------------------------------------------------- detection
def find_cards_auto(img: Image.Image,
                    min_area_frac: float = 0.0035,
                    max_area_frac: float = 0.06,
                    aspect_range: tuple = DEFAULT_ASPECT_RANGE) -> list[Box]:
    """Contour-detect card rectangles. Returns boxes in reading order."""
    arr = np.array(img.convert("RGB"))
    height, width = arr.shape[:2]
    frame_area = float(width * height)

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    # Dilate + close so a card's border survives as one connected contour even
    # where the art washes out against the table background.
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found: list[Box] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h == 0:
            continue
        frac = (w * h) / frame_area
        aspect = w / float(h)
        if not (min_area_frac <= frac <= max_area_frac):
            continue
        if not (aspect_range[0] <= aspect <= aspect_range[1]):
            continue
        # Reject outlines that aren't filled rectangles (L-shaped UI chrome).
        if cv2.contourArea(c) / float(w * h) < 0.6:
            continue
        found.append(Box(x, y, w, h))

    found = _dedupe(found)
    if len(found) >= 2:
        # Every card in a pack renders the same size; drop outliers such as a
        # hover-zoomed card or sidebar art.
        median_area = float(np.median([b.area for b in found]))
        found = [b for b in found if 0.6 * median_area <= b.area <= 1.6 * median_area]

    return _sort_reading_order(found)


def grid_boxes(img: Image.Image, region: list, rows: int, cols: int,
               pad: float = 0.04) -> list[Box]:
    """Fixed-layout fallback: evenly slice a region into rows x cols cells."""
    width, height = img.size
    rx, ry, rw, rh = region
    x0, y0 = rx * width, ry * height
    gw, gh = rw * width, rh * height
    cw, ch = gw / cols, gh / rows
    px, py = cw * pad, ch * pad
    out: list[Box] = []
    for r in range(rows):
        for c in range(cols):
            out.append(Box(int(x0 + c * cw + px), int(y0 + r * ch + py),
                           int(cw - 2 * px), int(ch - 2 * py)))
    return out


def find_cards(img: Image.Image, cfg: dict) -> tuple[list[Box], str]:
    """Dispatch on the vision.detection config. Returns (boxes, mode_used)."""
    d = (cfg or {}).get("detection", {}) or {}
    mode = d.get("mode", "auto")
    if mode == "auto":
        boxes = find_cards_auto(
            img,
            min_area_frac=float(d.get("min_area_frac", 0.0035)),
            max_area_frac=float(d.get("max_area_frac", 0.06)),
            aspect_range=tuple(d.get("aspect_range", DEFAULT_ASPECT_RANGE)),
        )
        if boxes:
            return boxes, "auto"
        print("[detect] auto detection found no cards; falling back to grid.",
              flush=True)
    g = d.get("grid", {}) or {}
    boxes = grid_boxes(img,
                       g.get("region", [0.08, 0.14, 0.84, 0.72]),
                       int(g.get("rows", 2)), int(g.get("cols", 7)))
    return boxes, "grid"


# --------------------------------------------------------------------------- crops
def crop_cards(img: Image.Image, boxes: list[Box]) -> list[Image.Image]:
    return [img.crop(b.as_tuple()) for b in boxes]


def name_banner(crop: Image.Image, frac: float = 0.14) -> Image.Image:
    """Top strip of a card, where the printed name lives.

    Used for the recheck pass: when a full-card read comes back shaky, sending
    just the title bar (upscaled harder) usually settles it. `frac` needs
    tuning against a real screenshot — the detected box often includes a drop
    shadow above the card, which pushes the name further down.
    """
    w, h = crop.size
    return crop.crop((0, 0, w, max(1, int(h * frac))))


def draw_boxes(img: Image.Image, boxes: list[Box],
               labels: Optional[list] = None) -> Image.Image:
    """Debug overlay: numbered boxes plus whatever name each slot resolved to."""
    arr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    for i, b in enumerate(boxes):
        cv2.rectangle(arr, (b.x, b.y), (b.x + b.w, b.y + b.h), (0, 200, 255), 2)
        text = str(i + 1)
        if labels and i < len(labels) and labels[i]:
            text += f" {labels[i]}"
        cv2.putText(arr, text, (b.x + 4, b.y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
