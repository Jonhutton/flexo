"""Free TTS via Microsoft edge-tts (no API key required)."""

import asyncio

import edge_tts

from .base import TTSProvider


class EdgeTTS(TTSProvider):
    name = "edge"

    def __init__(self, voice: str = "en-GB-RyanNeural"):
        self.voice = voice

    def synth(self, text: str) -> bytes:
        return asyncio.run(self._synth_async(text))

    async def _synth_async(self, text: str) -> bytes:
        communicate = edge_tts.Communicate(text, self.voice)
        chunks = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunks.extend(chunk["data"])
        return bytes(chunks)
