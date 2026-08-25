"""Standalone tests for PushToTalk early-STT segmentation and barge-in callbacks.

No mic, no whisper download, no pytest:

    python tests/test_stt_early.py

Drives PushToTalk's key handlers and audio callback directly with synthetic
blocks (sine = speech, zeros = silence) and a fake whisper model, verifying:
  - on_ptt_down fires immediately on key-down (barge-in hook)
  - segments are cut at silence gaps WHILE holding (early transcription)
  - the joined transcript is delivered in order after release
  - a stray tap aborts via on_ptt_abort (releasing the speech hold)
"""

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from flexo.stt import PushToTalk

SR = 16000
BLOCK = 480  # 30ms blocks, like a real stream


class FakeModel:
    """Stands in for WhisperModel: returns 'word<N>' per transcribe call."""

    def __init__(self):
        self.calls = 0
        self.lock = threading.Lock()

    def transcribe(self, audio, language=None):
        with self.lock:
            self.calls += 1
            n = self.calls
        return iter([SimpleNamespace(text=f"word{n}")]), None


def make_ptt(**kw):
    events = {"down": 0, "abort": 0, "text": None, "done": threading.Event()}

    def on_transcript(text):
        events["text"] = text
        events["done"].set()

    def on_down():
        events["down"] += 1

    def on_abort():
        events["abort"] += 1
        events["done"].set()

    ptt = PushToTalk(
        on_transcript=on_transcript, on_ptt_down=on_down, on_ptt_abort=on_abort,
        sample_rate=SR, early_stt=True, silence_ms=450, silence_rms=0.008,
        min_segment_seconds=0.6, min_seconds=0.3, **kw,
    )
    ptt._model = FakeModel()
    ptt._start_stream = lambda: None   # no real mic in tests
    ptt._stop_stream = lambda: None
    return ptt, events


def feed(ptt, seconds, kind):
    """Push `seconds` of synthetic audio through the callback in 30ms blocks."""
    total = int(seconds * SR)
    t = np.arange(BLOCK) / SR
    for start in range(0, total, BLOCK):
        if kind == "speech":
            block = (0.1 * np.sin(2 * np.pi * 220 * t)).astype("float32")
        else:
            block = np.zeros(BLOCK, dtype="float32")
        ptt._audio_cb(block.reshape(-1, 1), BLOCK, None, None)


# --------------------------------------------------------------------------- tests
def test_early_segmentation_and_ordered_transcript():
    ptt, events = make_ptt()
    ptt._on_press(ptt.hotkey)
    assert events["down"] == 1, "on_ptt_down did not fire on key-down"

    feed(ptt, 1.0, "speech")     # sentence one...
    feed(ptt, 0.6, "silence")    # ...pause: should cut + transcribe segment 1 now
    assert ptt._session["early_segs"] == 1, "no segment cut at the silence gap"
    feed(ptt, 1.0, "speech")     # sentence two (tail)
    ptt._on_release(ptt.hotkey)

    assert events["done"].wait(3.0), "transcript never delivered"
    assert events["text"] == "word1 word2", f"got {events['text']!r}"
    assert events["abort"] == 0
    print(f"  transcript: {events['text']!r} (1 segment pre-transcribed during hold)")


def test_stray_tap_aborts():
    ptt, events = make_ptt()
    ptt._on_press(ptt.hotkey)
    feed(ptt, 0.1, "speech")     # 100ms tap, under min_seconds
    ptt._on_release(ptt.hotkey)
    assert events["done"].wait(3.0), "abort never delivered"
    assert events["abort"] == 1 and events["text"] is None


def test_no_early_cut_before_min_segment():
    ptt, events = make_ptt()
    ptt._on_press(ptt.hotkey)
    feed(ptt, 0.3, "speech")     # under min_segment_seconds
    feed(ptt, 0.6, "silence")
    assert ptt._session["early_segs"] == 0, "cut a segment shorter than the minimum"
    feed(ptt, 0.7, "speech")
    ptt._on_release(ptt.hotkey)
    assert events["done"].wait(3.0)
    assert events["text"] == "word1"   # everything went out as one tail segment


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
