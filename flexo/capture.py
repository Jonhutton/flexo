"""Screen capture of the MTG Arena window — the vision input layer's eyes.

Finds the Arena client window via the Win32 API (ctypes — no extra dependency)
and grabs exactly that region with mss, so a second monitor full of OBS never
leaks into Flexo's view. Falls back to the primary monitor when the window
can't be found or is minimized.

Every capture is saved as PNG to a per-session folder (debugging / replay /
future fixtures), and a downscaled JPEG is produced for the vision API — a
legible pick screen doesn't need native 1440p, and image tokens scale with
resolution.

Fail-soft: capture() returns None on any failure and never raises into the
live loop.
"""

import ctypes
import io
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import mss
from PIL import Image

# Window titles to try, in order. The Arena client titles itself "MTGA".
_ARENA_TITLES = ("MTGA",)


@dataclass
class Capture:
    """One screenshot, ready for both disk and the vision API."""
    png_path: Path            # full-res PNG saved for debugging/replay
    jpeg_bytes: bytes         # downscaled JPEG for the API
    media_type: str           # always "image/jpeg"
    width: int                # downscaled dimensions actually sent
    height: int
    taken_at: float           # time.time() at grab


# --------------------------------------------------------------------------- win32
def _find_arena_rect() -> Optional[tuple[int, int, int, int]]:
    """Return (left, top, width, height) of the Arena window, or None."""
    if sys.platform != "win32":
        return None
    user32 = ctypes.windll.user32
    for title in _ARENA_TITLES:
        hwnd = user32.FindWindowW(None, title)
        if not hwnd:
            continue
        if user32.IsIconic(hwnd):   # minimized -> rect is garbage
            return None
        class RECT(ctypes.Structure):
            _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                        ("right", ctypes.c_long), ("bottom", ctypes.c_long)]
        r = RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(r)):
            return None
        w, h = r.right - r.left, r.bottom - r.top
        if w < 200 or h < 200:      # sanity: not a real game window
            return None
        return (r.left, r.top, w, h)
    return None


def _set_dpi_aware() -> None:
    """GetWindowRect returns virtual coords unless the process is DPI-aware;
    without this a scaled display (125%/150%) makes the grab region drift."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)   # per-monitor aware
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


# --------------------------------------------------------------------------- capture
class ScreenCapture:
    def __init__(self, captures_dir: Path, max_long_edge: int = 2048,
                 jpeg_quality: int = 85):
        _set_dpi_aware()
        self.max_long_edge = max_long_edge
        self.jpeg_quality = jpeg_quality
        session = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.session_dir = Path(captures_dir) / session
        self._counter = 0

    def _grab(self) -> Optional[Image.Image]:
        """Raw grab of the Arena window (or primary monitor) as a PIL image."""
        try:
            with mss.mss() as sct:
                rect = _find_arena_rect()
                if rect:
                    left, top, w, h = rect
                    region = {"left": left, "top": top, "width": w, "height": h}
                else:
                    region = sct.monitors[1]   # primary monitor fallback
                raw = sct.grab(region)
                return Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
        except Exception as e:
            print(f"[capture] screen grab failed: {e}", flush=True)
            return None

    def grab_gray_thumb(self, size: int = 48) -> Optional[bytes]:
        """Tiny grayscale thumbnail for cheap screen-change detection (auto mode).
        No disk write, no API call — just pixels for diffing."""
        img = self._grab()
        if img is None:
            return None
        return img.convert("L").resize((size, size), Image.BILINEAR).tobytes()

    def capture(self, tag: str = "shot") -> Optional[Capture]:
        """Grab the screen, save PNG, return API-ready JPEG bytes.

        Returns None on any failure — the caller logs and skips, never crashes.
        """
        img = self._grab()
        if img is None:
            return None

        try:
            self.session_dir.mkdir(parents=True, exist_ok=True)
            self._counter += 1
            png_path = self.session_dir / f"{self._counter:03d}_{tag}.png"
            img.save(png_path, "PNG")
        except Exception as e:
            print(f"[capture] could not save capture ({e}); continuing.", flush=True)
            png_path = Path("(unsaved)")

        jpeg, w, h = encode_for_vision(img, self.max_long_edge, self.jpeg_quality)
        return Capture(png_path=png_path, jpeg_bytes=jpeg, media_type="image/jpeg",
                       width=w, height=h, taken_at=time.time())


def encode_for_vision(img: Image.Image, max_long_edge: int = 2048,
                      jpeg_quality: int = 85) -> tuple[bytes, int, int]:
    """Downscale (if needed) and JPEG-encode an image for the vision API."""
    long_edge = max(img.width, img.height)
    if max_long_edge and long_edge > max_long_edge:
        scale = max_long_edge / long_edge
        img = img.resize((round(img.width * scale), round(img.height * scale)),
                         Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=jpeg_quality)
    return buf.getvalue(), img.width, img.height


def thumb_diff(a: Optional[bytes], b: Optional[bytes]) -> float:
    """Mean absolute pixel difference between two grayscale thumbs, 0.0..1.0.
    1.0 when either is missing or sizes differ (treat as 'changed')."""
    if a is None or b is None or len(a) != len(b):
        return 1.0
    total = sum(abs(x - y) for x, y in zip(a, b))
    return total / (len(a) * 255)


def load_image_for_vision(path: Path, max_long_edge: int = 2048,
                          jpeg_quality: int = 85) -> tuple[bytes, str]:
    """Load a screenshot from disk (dry-run path) into API-ready JPEG bytes."""
    img = Image.open(path)
    jpeg, _, _ = encode_for_vision(img, max_long_edge, jpeg_quality)
    return jpeg, "image/jpeg"
