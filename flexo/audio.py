"""Cancellable speech pipeline: one queue, one playback worker, hard barge-in.

Replaces the old fire-and-forget Speaker. Rules this module enforces:

  - ONE utterance plays at a time (single worker thread owns the output device).
  - Every utterance is cancellable MID-WORD: playback writes small (~50ms) PCM
    blocks and checks a per-utterance cancel event between writes.
  - PTT barge-in (`interrupt_and_hold`): kills current audio, flushes the whole
    queue, and blocks playback until `release_hold` — the drafter always wins.
  - Stale packs never speak: a "pack" utterance carries a monotonically
    increasing seq; enqueuing a newer pack line (or `cancel_stale_packs` when a
    new pack EVENT lands, before its line is even ready) drops queued lines
    about older packs. A line that is already mid-air is left to finish
    (barge-in is the only thing that cuts audible speech).

Providers hand us (sample_rate, iterator of float32 mono PCM chunks) via
TTSProvider.stream_pcm, so streaming and non-streaming TTS play identically.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import numpy as np
import sounddevice as sd

BLOCK_SECONDS = 0.05   # playback write granularity == barge-in kill latency


@dataclass
class Utterance:
    text: str
    kind: str = "system"          # "pack" | "banter" | "system"
    seq: int = 0                  # pack generation (staleness ordering)
    enqueued_at: float = 0.0      # perf_counter stamp for --timings
    playing: bool = False         # True once audio actually reached the device
    cancelled: threading.Event = field(default_factory=threading.Event)


def play_pcm_stream(sample_rate: int, chunks: Iterable[np.ndarray],
                    cancelled: threading.Event,
                    on_playback_start: Optional[Callable[[], None]] = None) -> None:
    """Play float32 mono PCM chunks; checks `cancelled` between ~50ms blocks.

    On cancel the stream is abort()ed, dropping device-buffered audio, so the
    voice dies mid-word instead of draining the buffer.
    """
    block = max(1, int(sample_rate * BLOCK_SECONDS))
    stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32",
                             latency="low")
    stream.start()
    started = False
    it = iter(chunks)
    try:
        while not cancelled.is_set():
            try:
                chunk = next(it)
            except StopIteration:
                break
            data = np.asarray(chunk, dtype="float32").reshape(-1, 1)
            for i in range(0, len(data), block):
                if cancelled.is_set():
                    break
                if not started:
                    started = True
                    if on_playback_start:
                        on_playback_start()
                stream.write(np.ascontiguousarray(data[i:i + block]))
    finally:
        if cancelled.is_set():
            stream.abort()
        else:
            stream.stop()
        stream.close()
        close = getattr(chunks, "close", None)   # stop a streaming HTTP body
        if close:
            close()


class SpeechQueue:
    """Single-owner speech queue. All voice output goes through here."""

    def __init__(self, tts, timings: bool = False, player=play_pcm_stream):
        self.tts = tts
        self.timings = timings
        self._player = player
        self._q: deque = deque()
        self._cond = threading.Condition()
        self._holding = False
        self._current: Optional[Utterance] = None
        threading.Thread(target=self._worker, daemon=True).start()

    # ------------------------------------------------------------------ enqueue
    def say(self, text: str, kind: str = "system", seq: int = 0) -> Optional[Utterance]:
        text = (text or "").strip()
        if not text:
            return None
        utt = Utterance(text=text, kind=kind, seq=seq,
                        enqueued_at=time.perf_counter())
        with self._cond:
            if kind == "pack":
                self._drop_stale_packs_locked(utt.seq)
            self._q.append(utt)
            self._cond.notify()
        return utt

    # ------------------------------------------------------------------ control
    def interrupt_and_hold(self) -> bool:
        """PTT down: kill current audio, flush everything, block playback.

        Returns True if Flexo was audibly cut off mid-line (drives the persona's
        one-clause interruption acknowledgment).
        """
        with self._cond:
            self._holding = True
            cur = self._current
            cut = bool(cur and cur.playing and not cur.cancelled.is_set())
            if cur:
                cur.cancelled.set()
            for u in self._q:
                u.cancelled.set()
            self._q.clear()
            return cut

    def release_hold(self) -> None:
        with self._cond:
            self._holding = False
            self._cond.notify()

    def cancel_stale_packs(self, current_seq: int) -> None:
        """A new pack event exists: drop queued lines about older packs."""
        with self._cond:
            self._drop_stale_packs_locked(current_seq + 1)

    def cancel_utterance(self, utt: Utterance) -> None:
        with self._cond:
            utt.cancelled.set()
            try:
                self._q.remove(utt)
            except ValueError:
                pass   # already playing or already gone

    def wait_idle(self, timeout: float = 30.0) -> bool:
        """Block until nothing is queued or playing (used by replay shutdown)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._cond:
                if not self._q and self._current is None:
                    return True
            time.sleep(0.1)
        return False

    # ------------------------------------------------------------------ internals
    def _drop_stale_packs_locked(self, newer_seq: int) -> None:
        for u in [u for u in self._q if u.kind == "pack" and u.seq < newer_seq]:
            u.cancelled.set()
            self._q.remove(u)

    def _worker(self) -> None:
        while True:
            with self._cond:
                while self._holding or not self._q:
                    self._cond.wait()
                utt = self._q.popleft()
                self._current = utt
            try:
                self._speak(utt)
            except Exception as e:
                print(f"[speech] utterance dropped: {e}", flush=True)
            finally:
                with self._cond:
                    self._current = None

    def _speak(self, utt: Utterance) -> None:
        if utt.cancelled.is_set():
            return
        try:
            sample_rate, chunks = self.tts.stream_pcm(utt.text)
            self._play(utt, sample_rate, chunks)
            return
        except Exception as e:
            print(f"[speech:{getattr(self.tts, 'name', '?')}] streaming failed: {e}",
                  flush=True)
        if utt.cancelled.is_set() or utt.playing:
            return   # cancelled meanwhile, or died mid-play: don't restart the line
        try:
            # Non-streaming retry: full synth + decode through the same player.
            from flexo.tts.base import decode_audio
            data, sample_rate = decode_audio(self.tts.synth(utt.text))
            self._play(utt, sample_rate, iter([data]))
        except Exception as e:
            print(f"[speech] fallback synth failed, line dropped: {e}", flush=True)

    def _play(self, utt: Utterance, sample_rate: int, chunks) -> None:
        def on_start():
            utt.playing = True
            if self.timings:
                dt = (time.perf_counter() - utt.enqueued_at) * 1000
                print(f"[timings] {utt.kind}: playback-start +{dt:.0f}ms after enqueue",
                      flush=True)

        if self.timings:
            chunks = self._stamp_first_chunk(utt, chunks)
        self._player(sample_rate, chunks, utt.cancelled, on_start)

    def _stamp_first_chunk(self, utt: Utterance, chunks):
        first = True
        for c in chunks:
            if first:
                first = False
                dt = (time.perf_counter() - utt.enqueued_at) * 1000
                print(f"[timings] {utt.kind}: tts-first-audio +{dt:.0f}ms after enqueue",
                      flush=True)
            yield c
