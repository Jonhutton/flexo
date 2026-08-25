"""Push-to-talk speech-to-text with barge-in and early transcription.

HOLD the hotkey (default F8) to record the mic; RELEASE to finish. Push-to-talk
ONLY — the mic is never open outside a key hold, so narrating to viewers never
reaches Flexo.

Latency/interruption behaviors (v1.1):
  - on_ptt_down fires the INSTANT the key goes down, before any audio work —
    the orchestrator uses it to kill Flexo's voice (hard barge-in).
  - Early STT: while the key is held, speech is cut into segments at silence
    gaps and transcribed in a background worker AS YOU TALK, so at release only
    the final tail still needs transcribing. faster-whisper has no true
    streaming mode; silence-boundary chunking is the practical equivalent and
    never splits words. Disable with voice_input.early_stt: false.
  - The model is warmed up at start() so the first real hold doesn't pay
    ctranslate2's lazy-init cost.
  - on_ptt_abort fires when a hold produced no usable speech (stray tap,
    transcription failure) so the orchestrator can un-pause Flexo's voice.
"""

import queue
import threading
import time
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
from pynput import keyboard


def _parse_hotkey(name: str):
    name = (name or "f8").strip().lower()
    special = getattr(keyboard.Key, name, None)
    if special is not None:
        return special
    return keyboard.KeyCode.from_char(name[0])


class PushToTalk:
    def __init__(
        self,
        on_transcript: Callable[[str], None],
        hotkey: str = "f8",
        whisper_model: str = "small",
        device: str = "cpu",
        sample_rate: int = 16000,
        min_seconds: float = 0.3,
        on_ptt_down: Optional[Callable[[], None]] = None,
        on_ptt_abort: Optional[Callable[[], None]] = None,
        early_stt: bool = True,
        silence_ms: int = 600,
        silence_rms: float = 0.008,
        min_segment_seconds: float = 1.2,
        timings: bool = False,
    ):
        self.on_transcript = on_transcript
        self.on_ptt_down = on_ptt_down
        self.on_ptt_abort = on_ptt_abort
        self.hotkey = _parse_hotkey(hotkey)
        self.hotkey_name = hotkey
        self.whisper_model_size = whisper_model
        self.device = device
        self.sample_rate = sample_rate
        self.min_seconds = min_seconds
        self.early_stt = early_stt
        self.silence_samples = int(silence_ms / 1000 * sample_rate)
        self.silence_rms = silence_rms
        self.min_segment_samples = int(min_segment_seconds * sample_rate)
        self.timings = timings

        self._model = None
        self._recording = False
        self._session: Optional[dict] = None
        self._stream: Optional[sd.InputStream] = None
        self._listener: Optional[keyboard.Listener] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """Load the model, warm it up, and begin listening for the hotkey."""
        from faster_whisper import WhisperModel  # heavy import; do it lazily

        compute_type = "int8" if self.device == "cpu" else "float16"
        print(f"[stt] loading faster-whisper '{self.whisper_model_size}' "
              f"({self.device})...", flush=True)
        self._model = WhisperModel(
            self.whisper_model_size, device=self.device, compute_type=compute_type
        )
        # Warm-up: ctranslate2 lazily initializes on the first transcription, which
        # would otherwise land on your first real utterance.
        segs, _ = self._model.transcribe(
            np.zeros(self.sample_rate // 2, dtype="float32"), language="en"
        )
        list(segs)
        self._listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self._listener.start()
        mode = "early (transcribe while held)" if self.early_stt else "at release"
        print(f"[stt] push-to-talk ready - hold {self.hotkey_name.upper()} to talk "
              f"(STT {mode}).", flush=True)

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()
        self._stop_stream()

    # ------------------------------------------------------------------ hotkey
    def _on_press(self, key):
        if key != self.hotkey:
            return
        with self._lock:
            if self._recording:
                return
            self._recording = True
            self._session = {
                "q": queue.Queue(),    # ("seg", blocks) | ("end", total) | ("abort",)
                "blocks": [],          # un-committed audio since the last segment cut
                "samples": 0,          # samples in `blocks`
                "silence": 0,          # trailing-silence samples in `blocks`
                "total": 0,            # samples this hold, across all segments
                "early_segs": 0,
            }
            session = self._session
        # Barge-in FIRST: Flexo must shut up before we even open the mic.
        if self.on_ptt_down:
            try:
                self.on_ptt_down()
            except Exception as e:
                print(f"[stt] on_ptt_down failed: {e}", flush=True)
        threading.Thread(target=self._session_worker, args=(session,),
                         daemon=True).start()
        self._start_stream()

    def _on_release(self, key):
        if key != self.hotkey:
            return
        with self._lock:
            if not self._recording:
                return
            self._recording = False
            session = self._session
            self._session = None
        self._stop_stream()
        if session is None:
            return
        session["t_release"] = time.perf_counter()
        if session["total"] < self.min_seconds * self.sample_rate:
            session["q"].put(("abort",))       # a stray tap, not speech
            return
        if session["blocks"]:
            session["q"].put(("seg", session["blocks"]))
            session["blocks"] = []
        session["q"].put(("end", session["total"]))

    # ------------------------------------------------------------------ audio
    def _audio_cb(self, indata, frames, time_info, status):
        session = self._session
        if not self._recording or session is None:
            return
        block = indata.copy().reshape(-1)
        session["blocks"].append(block)
        session["samples"] += len(block)
        session["total"] += len(block)
        if not self.early_stt:
            return
        # Silence-boundary segmentation: once we've heard >= min_segment of
        # speech followed by a silence gap, commit that chunk for transcription
        # now instead of at release. Cutting at silence keeps words whole.
        rms = float(np.sqrt(np.mean(block ** 2)))
        if rms < self.silence_rms:
            session["silence"] += len(block)
        else:
            session["silence"] = 0
        if (session["silence"] >= self.silence_samples
                and session["samples"] - session["silence"] >= self.min_segment_samples):
            seg = session["blocks"]
            session["blocks"] = []
            session["samples"] = 0
            session["silence"] = 0
            session["early_segs"] += 1
            session["q"].put(("seg", seg))

    def _start_stream(self):
        self._stream = sd.InputStream(
            samplerate=self.sample_rate, channels=1, dtype="float32",
            callback=self._audio_cb,
        )
        self._stream.start()

    def _stop_stream(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

    # ------------------------------------------------------------------ transcription
    def _session_worker(self, session: dict):
        """One per hold: transcribes segments in order as they're committed,
        then delivers the joined transcript (or aborts)."""
        texts: list[str] = []
        failed = False
        while True:
            item = session["q"].get()
            if item[0] == "abort":
                self._abort()
                return
            if item[0] == "end":
                break
            audio = np.concatenate(item[1]).astype("float32")
            try:
                segs, _ = self._model.transcribe(audio, language="en")
                text = " ".join(s.text.strip() for s in segs).strip()
                if text:
                    texts.append(text)
            except Exception as e:
                print(f"[stt] transcription failed: {e}", flush=True)
                failed = True
        text = " ".join(texts).strip()
        if self.timings and "t_release" in session:
            dt = (time.perf_counter() - session["t_release"]) * 1000
            audio_s = session["total"] / self.sample_rate
            print(f"[timings] stt: transcript +{dt:.0f}ms after release "
                  f"({audio_s:.1f}s audio, {session['early_segs']} segment(s) "
                  f"pre-transcribed during hold)", flush=True)
        if not text:
            if not failed:
                print("[stt] heard nothing intelligible.", flush=True)
            self._abort()
            return
        try:
            self.on_transcript(text)
        except Exception as e:
            print(f"[stt] on_transcript failed: {e}", flush=True)
            self._abort()

    def _abort(self):
        if self.on_ptt_abort:
            try:
                self.on_ptt_abort()
            except Exception as e:
                print(f"[stt] on_ptt_abort failed: {e}", flush=True)
