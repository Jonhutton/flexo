"""Standalone tests for card-rectangle detection and the pack-change detector.
No screen, no network, no API keys:

    python tests/test_detect.py

Cards are synthesized as light rounded rectangles on a dark table, at the real
MTG aspect ratio, so the detector is exercised on the same shape it has to find
in Arena. This is the layer that lets Flexo read cards one at a time instead of
hunting 9px text in a full-screen grab.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw

from flexo.capture import ChangeDetector
from flexo.detect import (DEFAULT_SLOTS, crop_cards, crop_fraction, find_cards,
                          find_cards_auto, grid_boxes, name_banner, slot_boxes)

W, H = 1920, 1080
CARD_W, CARD_H = 150, 209          # ~0.718 aspect, close to a real card


def make_screen(n_cards: int, cols: int = 7) -> Image.Image:
    """A dark 1080p frame with `n_cards` card-shaped rectangles laid out in rows."""
    img = Image.new("RGB", (W, H), (18, 20, 24))
    d = ImageDraw.Draw(img)
    x0, y0, gap = 240, 180, 40
    for i in range(n_cards):
        r, c = divmod(i, cols)
        x = x0 + c * (CARD_W + gap)
        y = y0 + r * (CARD_H + gap)
        d.rounded_rectangle([x, y, x + CARD_W, y + CARD_H], radius=8,
                            fill=(226, 221, 205), outline=(30, 28, 26), width=3)
        # A darker title bar across the top, like a card's name plate.
        d.rectangle([x + 6, y + 6, x + CARD_W - 6, y + 28], fill=(198, 192, 176))
    return img


def test_finds_every_card():
    for n in (14, 13, 9, 3, 1):
        boxes = find_cards_auto(make_screen(n))
        assert len(boxes) == n, f"expected {n} boxes, got {len(boxes)}"


def test_reading_order_is_stable():
    """Slot numbers are the contract between detect and vision — row by row,
    left to right. If this drifts, every card gets the wrong name."""
    boxes = find_cards_auto(make_screen(9))
    first_row = boxes[:7]
    assert all(a.cx < b.cx for a, b in zip(first_row, first_row[1:])), \
        "first row is not left-to-right"
    assert boxes[7].cy > boxes[0].cy, "second row did not sort below the first"


def test_crops_match_boxes():
    img = make_screen(5)
    boxes = find_cards_auto(img)
    crops = crop_cards(img, boxes)
    assert len(crops) == 5
    for box, crop in zip(boxes, crops):
        assert crop.size == (box.w, box.h)
    banner = name_banner(crops[0], frac=0.14)
    assert banner.width == crops[0].width and 0 < banner.height < crops[0].height


def test_grid_fallback_shape():
    boxes = grid_boxes(make_screen(1), [0.08, 0.14, 0.84, 0.72], rows=2, cols=7)
    assert len(boxes) == 14
    assert all(b.w > 0 and b.h > 0 for b in boxes)


def test_empty_screen_finds_nothing():
    """A screen with no cards must come back empty, never as a guessed grid:
    phantom slots would send blank crops to the model on camera."""
    blank = Image.new("RGB", (W, H), (18, 20, 24))
    for mode in ("slots", "auto"):
        boxes, used = find_cards(blank, {"detection": {"mode": mode}})
        assert boxes == [] and used == mode, (mode, used, len(boxes))


# --- slots mode: Arena's fixed lattice -------------------------------------
LATTICE = DEFAULT_SLOTS["lattice"]


def make_arena(n_cards: int, frame_rgb=(235, 235, 235), zoom: bool = False) -> Image.Image:
    """Cards on the slot lattice, dark table, bright name bars - including
    pale/white frames, which is what contour detection missed on real pixels."""
    img = Image.new("RGB", (1918, 1198), (40, 22, 18))
    d = ImageDraw.Draw(img)
    for b in slot_boxes(img, LATTICE, 5, 3)[:n_cards]:
        d.rectangle([b.x, b.y, b.x + b.w, b.y + b.h], fill=frame_rgb)
        d.rectangle([b.x + 6, b.y + 6, b.x + b.w - 6, b.y + int(0.1 * b.h)],
                    fill=(200, 200, 190))
    if zoom:
        d.rectangle([641, 93, 641 + 488, 93 + 697], fill=(230, 225, 210),
                    outline=(10, 10, 10), width=6)
    return img


def test_slots_count_every_pack_size():
    for n in (14, 13, 9, 5, 4, 2, 1):
        boxes, mode = find_cards(make_arena(n), {"detection": {"mode": "slots"}})
        assert mode == "slots" and len(boxes) == n, (n, len(boxes))


def test_slots_reading_order():
    boxes, _ = find_cards(make_arena(7), {})          # slots is the default
    assert [b.y for b in boxes[:5]] == [boxes[0].y] * 5
    assert all(a.x < b.x for a, b in zip(boxes[:5], boxes[1:5]))
    assert boxes[5].y > boxes[0].y and boxes[5].x == boxes[0].x


def test_hover_zoom_is_occluded():
    boxes, mode = find_cards(make_arena(14, zoom=True), {})
    assert mode == "occluded" and boxes == []


REAL = Path(__file__).resolve().parents[1] / "Screenshots 101226"
REAL_TRUTH = {"114548": 14, "114710": "occluded", "115016": 9,
              "120021": 4, "120120": 2, "120149": 0}


def test_real_arena_screenshots():
    """The 2026-10-12 Quick Draft captures (local only, not committed)."""
    shots = sorted(REAL.glob("*.png")) if REAL.exists() else []
    if not shots:
        print("  (real screenshots not present; skipped)")
        return
    for f in shots:
        want = REAL_TRUTH.get(f.stem[-6:])
        if want is None:
            continue
        boxes, mode = find_cards(Image.open(f).convert("RGB"), {})
        got = "occluded" if mode == "occluded" else len(boxes)
        assert got == want, (f.name, got, want)


def test_crop_fraction():
    img = Image.new("RGB", (1000, 500))
    assert crop_fraction(img, None).size == (1000, 500)        # None = untouched
    assert crop_fraction(img, [0.1, 0.2, 0.5, 0.5]).size == (500, 250)


def test_change_detector_debounces():
    """One fire per new STABLE screen; animation frames must not trigger."""
    a, b = make_screen(14), make_screen(9)
    cd = ChangeDetector(hash_distance=6, stable_samples=3)

    fires = [i for i, img in enumerate([a, a, a, a, a]) if cd.update(img)]
    assert fires == [2], f"first screen should commit once, got {fires}"

    # Switching screens needs `stable_samples` steady frames before it commits.
    fires = [i for i, img in enumerate([b, b, b, b]) if cd.update(img)]
    assert fires == [2], f"second screen should commit once, got {fires}"

    # Holding still fires nothing further.
    assert not any(cd.update(b) for _ in range(4))


def test_change_detector_ignores_flicker():
    """A single odd frame mid-animation must not commit on its own."""
    a, b = make_screen(14), make_screen(3)
    cd = ChangeDetector(hash_distance=6, stable_samples=3)
    for _ in range(4):
        cd.update(a)
    assert not cd.update(b), "a lone changed frame should not commit"
    assert not cd.update(a), "returning to the committed screen should not fire"


if __name__ == "__main__":
    test_finds_every_card()
    test_reading_order_is_stable()
    test_crops_match_boxes()
    test_grid_fallback_shape()
    test_empty_screen_finds_nothing()
    test_slots_count_every_pack_size()
    test_slots_reading_order()
    test_hover_zoom_is_occluded()
    test_real_arena_screenshots()
    test_crop_fraction()
    test_change_detector_debounces()
    test_change_detector_ignores_flicker()
    print("test_detect: all tests passed.")
