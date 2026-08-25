"""Generate synthetic pick-screen fixtures for the vision dry-run path.

    python scripts/make_vision_fixture.py

Real Arena screenshots are better (press F9 in a live draft and grab them from
captures/), but these synthetic screens use real card names from the cached
17Lands data, so the full pipeline — vision -> validation -> ratings -> Flexo —
can be exercised end to end before the next real draft. Writes:

    tests/fixtures/pick_screen_p1p3.png     a draft pick screen (5 cards + sidebar)
    tests/fixtures/not_a_pick_screen.png    a menu-ish screen Flexo must ignore
"""

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures"

SET_CODE = "ECL"
PACK_CARDS = ["Kulrath Zealot", "Bogslither's Embrace", "Rooftop Percher",
              "Gangly Stompling", "Evolving Wilds"]
SIDEBAR = ["Prideful Feastling", "Kulrath Zealot"]


def font(size: int):
    for name in ("arialbd.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_card(d: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, name: str):
    d.rounded_rectangle([x, y, x + w, y + h], radius=14, fill=(38, 34, 48),
                        outline=(120, 110, 140), width=3)
    # name banner near the top of the card, like a real frame
    d.rounded_rectangle([x + 10, y + 14, x + w - 10, y + 52], radius=8,
                        fill=(214, 205, 180))
    f = font(22)
    d.text((x + 18, y + 22), name, font=f, fill=(20, 18, 14))
    # art box + text box so it reads as a magic card
    d.rectangle([x + 14, y + 62, x + w - 14, y + int(h * 0.55)], fill=(70, 60, 90))
    d.rectangle([x + 14, y + int(h * 0.60), x + w - 14, y + h - 20],
                fill=(200, 192, 170))


def make_pick_screen() -> Path:
    img = Image.new("RGB", (1920, 1080), (16, 14, 22))
    d = ImageDraw.Draw(img)
    d.text((760, 40), "Pack 1, Pick 3", font=font(44), fill=(235, 230, 210))

    card_w, card_h, gap = 300, 430, 40
    total = len(PACK_CARDS) * card_w + (len(PACK_CARDS) - 1) * gap
    x0 = (1500 - total) // 2 + 60
    for i, name in enumerate(PACK_CARDS):
        draw_card(d, x0 + i * (card_w + gap), 200, card_w, card_h, name)

    # drafted-cards sidebar on the right
    d.rectangle([1600, 120, 1900, 1000], fill=(24, 22, 32),
                outline=(90, 85, 110), width=2)
    d.text((1620, 140), "Drafted Cards", font=font(28), fill=(220, 215, 200))
    f = font(22)
    for i, name in enumerate(SIDEBAR):
        d.rounded_rectangle([1615, 200 + i * 54, 1885, 244 + i * 54], radius=6,
                            fill=(52, 48, 66))
        d.text((1628, 210 + i * 54), name, font=f, fill=(228, 222, 205))

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "pick_screen_p1p3.png"
    img.save(path)
    return path


def make_non_pick_screen() -> Path:
    img = Image.new("RGB", (1920, 1080), (12, 16, 26))
    d = ImageDraw.Draw(img)
    d.text((700, 380), "MTG ARENA", font=font(80), fill=(230, 225, 205))
    d.rounded_rectangle([810, 560, 1110, 640], radius=12, fill=(180, 140, 40))
    d.text((900, 580), "PLAY", font=font(40), fill=(25, 20, 10))
    path = OUT / "not_a_pick_screen.png"
    img.save(path)
    return path


if __name__ == "__main__":
    print("wrote", make_pick_screen())
    print("wrote", make_non_pick_screen())
