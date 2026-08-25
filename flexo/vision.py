"""Vision extraction — Flexo sees the pick screen instead of parsing the log.

Sends a screenshot to the Claude vision API and gets back structured draft
state (screen type, pack/pick numbers, card names). Structured outputs
guarantee parseable JSON; card names still go through the fuzzy validation
layer (set_cards.py) before anything downstream trusts them.

Fail-soft: extract() retries once, then returns None. A missed extraction is
a skipped commentary beat, never a crash mid-recording.
"""

import base64
import json
from dataclasses import dataclass, field
from typing import Optional

import anthropic

# Strict JSON the model must return — mirrors what's actually visible on the
# Arena pick screen. additionalProperties:false + required for structured outputs.
EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "screen_type": {"type": "string",
                        "enum": ["draft_pick", "deck_build", "other"]},
        "pack_number": {"type": "integer"},
        "pick_number": {"type": "integer"},
        "cards_in_pack": {"type": "array", "items": {"type": "string"}},
        "cards_picked_so_far": {"type": "array", "items": {"type": "string"}},
        "confidence_notes": {"type": "string"},
    },
    "required": ["screen_type", "pack_number", "pick_number", "cards_in_pack",
                 "cards_picked_so_far", "confidence_notes"],
    "additionalProperties": False,
}

_SYSTEM = (
    "You read MTG Arena Quick Draft screenshots and return structured draft "
    "state as JSON. Be literal: transcribe card names exactly as printed on "
    "the cards, one entry per card visible in the current pack. If the screen "
    "is not a draft pick screen, set screen_type accordingly and leave the "
    "card lists empty.\n\n"
    "Rules:\n"
    "- cards_in_pack: every card offered in the pack currently on screen, "
    "left-to-right, top-to-bottom. Read the printed name on each card.\n"
    "- pack_number / pick_number: read the 'Pack X, Pick Y' indicator if "
    "visible; use 0 if unreadable.\n"
    "- cards_picked_so_far: only cards visibly listed in the drafted-cards "
    "sidebar/panel, if one is on screen. Empty list if not visible.\n"
    "- confidence_notes: anything ambiguous — partially covered names, blurry "
    "text, cards you had to guess. Empty string if everything was clear.\n"
    "- Never invent a card name you cannot actually read."
)


@dataclass
class VisionResult:
    screen_type: str
    pack_number: int
    pick_number: int
    cards_in_pack: list[str]
    cards_picked_so_far: list[str] = field(default_factory=list)
    confidence_notes: str = ""

    @property
    def is_pick_screen(self) -> bool:
        return self.screen_type == "draft_pick" and bool(self.cards_in_pack)


class VisionExtractor:
    def __init__(self, model: str, max_tokens: int = 1500,
                 effort: Optional[str] = None):
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort   # thinking/latency control; None for models without it
        self.client = anthropic.Anthropic()   # ANTHROPIC_API_KEY from env

    def extract(self, image_bytes: bytes,
                media_type: str = "image/jpeg") -> Optional[VisionResult]:
        """Screenshot -> VisionResult. One retry on a malformed reply, then None."""
        for attempt in (1, 2):
            data = self._call(image_bytes, media_type)
            if data is not None:
                try:
                    return VisionResult(
                        screen_type=data.get("screen_type", "other"),
                        pack_number=int(data.get("pack_number") or 0),
                        pick_number=int(data.get("pick_number") or 0),
                        cards_in_pack=[c for c in data.get("cards_in_pack", []) if c],
                        cards_picked_so_far=[c for c in data.get("cards_picked_so_far", []) if c],
                        confidence_notes=data.get("confidence_notes", ""),
                    )
                except Exception as e:
                    print(f"[vision] bad payload shape (attempt {attempt}): {e}",
                          flush=True)
            if attempt == 1:
                print("[vision] retrying extraction once...", flush=True)
        print("[vision] extraction failed; skipping this capture.", flush=True)
        return None

    # ------------------------------------------------------------------ internals
    def _call(self, image_bytes: bytes, media_type: str) -> Optional[dict]:
        output_config: dict = {"format": {"type": "json_schema",
                                          "schema": EXTRACT_SCHEMA}}
        if self.effort:
            output_config["effort"] = self.effort
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=[{"type": "text", "text": _SYSTEM,
                         "cache_control": {"type": "ephemeral"}}],
                output_config=output_config,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image",
                         "source": {"type": "base64", "media_type": media_type,
                                    "data": base64.standard_b64encode(image_bytes).decode()}},
                        {"type": "text",
                         "text": "Extract the draft state from this screenshot."},
                    ],
                }],
            )
            if resp.stop_reason == "refusal":
                print("[vision] model declined the request (refusal).", flush=True)
                return None
            text = next((b.text for b in resp.content if b.type == "text"), "")
            return json.loads(text)
        except Exception as e:
            # Network/model hiccups and JSON errors both land here — log & retry/skip.
            print(f"[vision] Claude call failed: {e}", flush=True)
            return None
