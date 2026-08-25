"""Standalone tests for the cancellable speech queue — the simulated-PTT-interrupt
suite. No audio device, no API keys, no pytest needed:

    python tests/test_speech_queue.py

A FakeTTS yields 50ms PCM chunks; a fake player "plays" them in real time by
sleeping, honoring the same cancel event the real player checks. So barge-in
latency measured here is representative of the real kill granularity.
"""

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from flexo.audio import SpeechQueue


class FakeTTS:
    """Streams `seconds` of silence in 50ms chunks per utterance."""
    name = "fake"

    def __init__(self, seconds: float = 3.0, sample_rate: int = 16000):
        self.seconds = seconds
        self.sample_rate = sample_rate

    def synth(self, text: str) -> bytes:
        raise RuntimeError("fallback path must not run in these tests")

    def stream_pcm(self, text: str):
        n = max(1, int(self.seconds / 0.05))
        chunk = np.zeros(int(self.sample_rate * 0.05), dtype="float32")

        def gen():
            for _ in range(n):
                yield chunk
        return self.sample_rate, gen()


def fake_player(sample_rate, chunks, cancelled, on_playback_start=None):
    """Real-time playback simulation: sleep for each chunk's duration, honor cancel."""
    started = False
    for c in chunks:
        if cancelled.is_set():
            break
        if not started:
            started = True
            if on_playback_start:
                on_playback_start()
        time.sleep(len(c) / sample_rate)
    close = getattr(chunks, "close", None)
    if close:
        close()


def wait_until(cond, timeout=3.0, step=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(step)
    return False


def make_queue(seconds=3.0):
    return SpeechQueue(FakeTTS(seconds=seconds), player=fake_player)


# --------------------------------------------------------------------------- tests
def test_barge_in_kills_current_and_flushes_queue():
    q = make_queue(seconds=3.0)
    utt = q.say("a long line", kind="pack", seq=1)
    assert wait_until(lambda: utt.playing), "utterance never started playing"
    queued = q.say("a queued line", kind="banter")

    t0 = time.perf_counter()
    cut = q.interrupt_and_hold()               # simulated F8 down
    assert cut is True, "should report Flexo was audibly cut off"
    assert wait_until(lambda: q._current is None, timeout=0.5), \
        "playback did not stop promptly after barge-in"
    stop_ms = (time.perf_counter() - t0) * 1000
    assert stop_ms < 300, f"barge-in kill took {stop_ms:.0f}ms (want <300ms)"
    assert len(q._q) == 0, "queue was not flushed on barge-in"
    assert queued.cancelled.is_set(), "queued utterance was not cancelled"

    q.release_hold()
    utt2 = q.say("after the interruption", kind="banter")
    assert wait_until(lambda: utt2.playing), "queue did not recover after barge-in"
    print(f"  barge-in kill latency: {stop_ms:.0f}ms")


def test_hold_blocks_playback_until_release():
    q = make_queue(seconds=0.3)
    q.interrupt_and_hold()                     # F8 held before anything queued
    utt = q.say("queued while holding", kind="banter")
    time.sleep(0.3)
    assert not utt.playing, "spoke while PTT was held"
    q.release_hold()
    assert wait_until(lambda: utt.playing), "did not speak after release"


def test_interrupt_before_audio_reports_not_cut():
    q = make_queue(seconds=0.5)
    q.interrupt_and_hold()
    q.say("never aired", kind="pack", seq=1)
    cut = q.interrupt_and_hold()               # second press: nothing was audible
    assert cut is False, "reported a cut when nothing was audibly playing"
    q.release_hold()


def test_new_pack_replaces_stale_queued_pack_line():
    q = make_queue(seconds=0.5)
    q.interrupt_and_hold()                     # hold so nothing plays yet
    old = q.say("pack 3 pick 4 advice", kind="pack", seq=1)
    banter = q.say("unrelated banter", kind="banter")
    new = q.say("pack 3 pick 5 advice", kind="pack", seq=2)
    assert old.cancelled.is_set(), "older queued pack line survived a newer one"
    assert not banter.cancelled.is_set(), "banter must not be dropped by pack staleness"
    assert not new.cancelled.is_set()
    assert [u.text for u in q._q] == ["unrelated banter", "pack 3 pick 5 advice"]
    q.release_hold()


def test_cancel_stale_packs_on_new_pack_event():
    # A new pack EVENT (before its line is even generated) must drop queued
    # lines about older packs.
    q = make_queue(seconds=0.5)
    q.interrupt_and_hold()
    old = q.say("pack N advice", kind="pack", seq=1)
    q.cancel_stale_packs(2)
    assert old.cancelled.is_set() and len(q._q) == 0
    q.release_hold()


def test_cancel_utterance_drops_unspoken_line():
    # Post-barge-in "fold the pick into the reply" path: the queued line is
    # cancelled and never spoken.
    q = make_queue(seconds=0.5)
    q.interrupt_and_hold()
    utt = q.say("undelivered pick line", kind="pack", seq=1)
    q.cancel_utterance(utt)
    q.release_hold()
    time.sleep(0.3)
    assert not utt.playing and utt.cancelled.is_set()


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
