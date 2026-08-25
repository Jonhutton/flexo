"""Swappable text-to-speech.

Provider is chosen at runtime:
  ElevenLabs  if ELEVENLABS_API_KEY is set (or forced via config) — streams PCM
              chunks so playback starts before generation finishes
  edge-tts    (free) otherwise — non-streaming, played through the same
              cancellable pipeline via the base-class stream_pcm fallback
Both expose TTSProvider.stream_pcm / synth, so the SpeechQueue never cares
which one is active.
"""

import os

from .base import TTSProvider
from .edgetts import EdgeTTS
from .elevenlabs import ElevenLabsTTS


def make_tts(cfg: dict) -> TTSProvider:
    tts_cfg = cfg.get("tts", {})
    forced = tts_cfg.get("force_provider")
    have_key = bool(os.environ.get("ELEVENLABS_API_KEY"))

    if forced == "elevenlabs" or (forced is None and have_key):
        if not have_key:
            print("[tts] force_provider=elevenlabs but ELEVENLABS_API_KEY unset; "
                  "falling back to edge-tts.", flush=True)
        else:
            return ElevenLabsTTS(
                voice_id=tts_cfg.get("elevenlabs_voice_id"),
                # `model` is the documented key; `elevenlabs_model` (v1) still honored.
                model=tts_cfg.get("model")
                      or tts_cfg.get("elevenlabs_model", "eleven_flash_v2_5"),
                output_format=tts_cfg.get("output_format", "pcm_22050"),
                streaming=tts_cfg.get("streaming", True),
            )
    return EdgeTTS(voice=tts_cfg.get("edge_voice", "en-GB-RyanNeural"))


__all__ = ["TTSProvider", "EdgeTTS", "ElevenLabsTTS", "make_tts"]
