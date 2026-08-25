"""ElevenLabs TTS (used when ELEVENLABS_API_KEY is set).

No SDK dependency — plain REST. Two paths:

  stream_pcm(): POST .../stream with a pcm_* output_format and yield PCM chunks
    as they arrive, so playback starts on the first chunk (~300-500ms) instead
    of after full generation. PCM sidesteps incremental MP3 decoding entirely.
  synth(): full MP3 in one response — the fallback when streaming is disabled,
    the output format isn't PCM, or a streaming request fails.

Model tier comes from config (`tts.model`). eleven_flash_v2_5 is the
low-latency banter tier; eleven_turbo_v2_5 the older, slower default — keep
both around for A/B.
"""

import os
from typing import Iterator, Tuple

import numpy as np
import requests

from .base import TTSProvider

API = "https://api.elevenlabs.io/v1/text-to-speech"


class ElevenLabsTTS(TTSProvider):
    name = "elevenlabs"

    def __init__(self, voice_id: str, model: str = "eleven_flash_v2_5",
                 output_format: str = "pcm_22050", streaming: bool = True):
        self.voice_id = voice_id
        self.model = model
        self.output_format = output_format
        # True streaming needs raw PCM (chunk-decodable); anything else falls
        # back to the full-MP3 path in the base class.
        self.streaming = streaming and output_format.startswith("pcm_")
        self.sample_rate = (int(output_format.rsplit("_", 1)[1])
                            if output_format.startswith("pcm_") else 0)
        self.api_key = os.environ["ELEVENLABS_API_KEY"]  # presence checked by make_tts
        # Keep-alive: reusing the TLS connection saves ~200-400ms on every
        # utterance after the first. Only the SpeechQueue worker thread calls
        # into this provider, so a single Session is safe.
        self.session = requests.Session()

    # ------------------------------------------------------------------ internals
    def _headers(self, accept: str) -> dict:
        return {
            "xi-api-key": self.api_key,
            "accept": accept,
            "content-type": "application/json",
        }

    def _payload(self, text: str) -> dict:
        return {
            "text": text,
            "model_id": self.model,
            "voice_settings": {"stability": 0.4, "similarity_boost": 0.7},
        }

    # ------------------------------------------------------------------ paths
    def synth(self, text: str) -> bytes:
        r = self.session.post(f"{API}/{self.voice_id}",
                              headers=self._headers("audio/mpeg"),
                              json=self._payload(text), timeout=30)
        r.raise_for_status()
        return r.content

    def stream_pcm(self, text: str) -> Tuple[int, Iterator[np.ndarray]]:
        if not self.streaming:
            return super().stream_pcm(text)

        def chunks():
            r = self.session.post(f"{API}/{self.voice_id}/stream",
                                  headers=self._headers("*/*"),
                                  params={"output_format": self.output_format},
                                  json=self._payload(text),
                                  stream=True, timeout=(10, 60))
            r.raise_for_status()
            leftover = b""
            try:
                for raw in r.iter_content(chunk_size=4096):
                    if not raw:
                        continue
                    buf = leftover + raw
                    cut = len(buf) & ~1   # 16-bit samples can split across chunks
                    leftover = buf[cut:]
                    if cut:
                        yield np.frombuffer(buf[:cut], dtype="<i2").astype(np.float32) / 32768.0
            finally:
                r.close()   # abandoning the generator (barge-in) stops the download

        return self.sample_rate, chunks()
