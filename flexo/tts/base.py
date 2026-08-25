"""TTS provider interface + PCM decode helper.

Providers produce audio two ways:
  synth(text)      -> full encoded bytes (MP3). Every provider must implement it;
                      it is the non-streaming fallback path.
  stream_pcm(text) -> (sample_rate, iterator of float32 mono PCM chunks). The
                      default wraps synth(), so non-streaming providers
                      (edge-tts) work unchanged; streaming providers
                      (ElevenLabs) override it to yield chunks as they arrive.

Playback and cancellation live in flexo.audio (SpeechQueue) — providers only
produce audio data. speak() remains as a blocking convenience for scripts.
"""

import io
import threading
from abc import ABC, abstractmethod
from typing import Iterator, Tuple

import numpy as np
import soundfile as sf


def decode_audio(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    """Decode encoded audio (MP3) to (float32 mono samples, sample_rate)."""
    data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sample_rate


class TTSProvider(ABC):
    name = "base"

    @abstractmethod
    def synth(self, text: str) -> bytes:
        """Return fully-generated encoded audio (MP3) bytes for `text`."""

    def stream_pcm(self, text: str) -> Tuple[int, Iterator[np.ndarray]]:
        """(sample_rate, PCM chunk iterator). Default: synth fully, then decode."""
        data, sample_rate = decode_audio(self.synth(text))
        return sample_rate, iter([data])

    def speak(self, text: str) -> None:
        """One-off blocking speak (scripts/manual tests). The live path is
        flexo.audio.SpeechQueue, which adds queuing and cancellation."""
        from flexo.audio import play_pcm_stream
        text = (text or "").strip()
        if not text:
            return
        try:
            sample_rate, chunks = self.stream_pcm(text)
            play_pcm_stream(sample_rate, chunks, threading.Event())
        except Exception as e:
            print(f"[tts:{self.name}] speak failed: {e}", flush=True)
