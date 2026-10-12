"""Vision extraction — Flexo sees the pick screen instead of parsing the log.

Two ways to read a pack, both ending in validated card names:

  VisionExtractor   whole-screen read. One image, one call; also reports screen
                    type and the pack/pick counter. Cheap, but it has to find
                    ~9px name text in a full 1080p frame.

  CardIdentifier    per-card read (preferred). detect.py finds the card
                    rectangles, each crop is upscaled on its own, and the model
                    is asked ONE question: which of this set's ~290 names is
                    on this card. Names only — never what a card does.

Both funnel through the same fuzzy validation layer (set_cards.CardValidator),
so a name that doesn't exist in the set can't reach Flexo's mouth either way.
Oracle text always comes from Scryfall, never from the model.

Fail-soft: a failed call returns None / empty rather than raising. A missed
read is a skipped commentary beat, never a crash mid-recording.
"""

import base64
import io
import json
from dataclasses import dataclass, field
from typing import Optional

import anthropic
from PIL import Image

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


# =========================================================================== #
# Per-card identification — the accurate path                                 #
# =========================================================================== #

# Names only. A union type would let the model return null, but structured
# outputs are happiest with a plain string, so "" means the name was unreadable.
IDENTIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "cards": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "slot": {"type": "integer"},
                    "name": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["slot", "name", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["cards"],
    "additionalProperties": False,
}

_IDENTIFY_SYSTEM = (
    "You identify Magic: The Gathering cards in crops taken from MTG Arena "
    "draft screenshots. Each image is one card.\n\n"
    "Read the card name from the title bar and match it to the candidate list "
    "below, which contains every card in this set. Use the art, mana cost, and "
    "frame color to disambiguate similar names. Copy the candidate name "
    "EXACTLY as it appears in the list.\n\n"
    "Rules:\n"
    "- Return one entry per image, in the order given, numbered from 1.\n"
    "- If a crop is not a card, or the name is unreadable, return an empty "
    "string for name rather than guessing.\n"
    "- confidence is 0.0-1.0: how sure you are of the NAME, not the card.\n"
    "- Never describe or evaluate a card. Names only."
)


@dataclass
class Identification:
    """One slot's read, with the evidence that produced it."""
    slot: int
    raw_name: Optional[str]            # what the model actually returned
    name: Optional[str] = None         # canonical set name, or None
    confidence: float = 0.0            # the model's own confidence
    score: float = 0.0                 # fuzzy match score against the set list
    status: str = "unknown"            # ok | recheck | unknown


def _b64_png(img: Image.Image, upscale_min_width: int) -> str:
    """PNG-encode a crop, upscaling small ones so the name text survives.

    At 1080p a card name is only ~9px tall. Lanczos upscaling before encoding
    is what makes the difference between a readable title bar and a smear.
    """
    if img.width < upscale_min_width:
        scale = upscale_min_width / img.width
        img = img.resize((int(img.width * scale), int(img.height * scale)),
                         Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class CardIdentifier:
    """Reads card names off per-card crops, constrained to one set's list."""

    def __init__(self, validator, model: str, max_tokens: int = 2000,
                 effort: Optional[str] = "low", upscale_min_width: int = 420):
        self.validator = validator          # set_cards.CardValidator
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.upscale_min_width = upscale_min_width
        self.client = anthropic.Anthropic()     # ANTHROPIC_API_KEY from env

    # ------------------------------------------------------------------ public
    def identify(self, crops: list, banner_fn=None) -> list:
        """Crops -> one Identification per slot, in the order given.

        One call for the whole pack, then at most one follow-up call that
        re-asks only about the slots that came back shaky.
        """
        if not crops:
            return []

        rows = self._ask(crops, self.validator.names)
        by_slot = {}
        for i, r in enumerate(rows):
            if isinstance(r, dict):
                by_slot[int(r.get("slot", i + 1) or i + 1)] = r

        ids: list = []
        for slot in range(1, len(crops) + 1):
            row = by_slot.get(slot, {})
            raw = (row.get("name") or "").strip() or None
            ids.append(self._settle(slot, raw, float(row.get("confidence") or 0.0)))

        shaky = [x for x in ids if x.status != "ok"]
        if shaky and banner_fn is not None:
            self._recheck(shaky, crops, banner_fn)
        return ids

    # ---------------------------------------------------------------- internals
    def _settle(self, slot: int, raw: Optional[str], confidence: float) -> Identification:
        """Snap one raw read to the set list via the shared validator."""
        if not raw:
            return Identification(slot, None, None, confidence, 0.0, "unknown")
        v = self.validator.validate(raw)
        return Identification(slot, raw, v.name, confidence, v.score, v.status)

    def _recheck(self, shaky: list, crops: list, banner_fn) -> None:
        """Second look at the unclear slots, using just their title bars.

        A full-card crop gives the model a lot of art to be distracted by. The
        name banner alone, upscaled harder, usually settles a shaky read — and
        this costs one extra call per pack at most, not one per slot.
        """
        print(f"[vision] rechecking {len(shaky)} slot(s): "
              f"{[x.slot for x in shaky]}", flush=True)
        banners = [banner_fn(crops[x.slot - 1]) for x in shaky]
        try:
            rows = self._ask(banners, self.validator.names,
                             note=("These are title-bar crops only - just the "
                                   "printed card name. Pick the single best "
                                   "candidate for each."))
        except Exception as e:
            print(f"[vision] recheck call failed ({e}); keeping first reads.",
                  flush=True)
            return

        for i, x in enumerate(shaky):
            row = rows[i] if i < len(rows) and isinstance(rows[i], dict) else {}
            raw = (row.get("name") or "").strip() or None
            if not raw:
                continue
            better = self._settle(x.slot, raw, float(row.get("confidence") or 0.0))
            # Only take the recheck if it actually improved the slot.
            if better.status == "ok" or better.score > x.score:
                x.raw_name, x.name = better.raw_name, better.name
                x.score, x.status, x.confidence = (better.score, better.status,
                                                   better.confidence)

    def _ask(self, crops: list, candidates: list, note: str = "") -> list:
        """One vision call. Returns the raw `cards` rows (possibly empty)."""
        # The candidate list is identical for every pack in a draft, so it goes
        # in the cached system prefix; only the images vary per call.
        system = [
            {"type": "text", "text": _IDENTIFY_SYSTEM},
            {"type": "text",
             "text": "Candidate card names for this set:\n" + "\n".join(candidates),
             "cache_control": {"type": "ephemeral"}},
        ]

        content: list = []
        for i, crop in enumerate(crops, start=1):
            content.append({"type": "text", "text": f"Card {i}:"})
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png",
                           "data": _b64_png(crop, self.upscale_min_width)},
            })
        tail = f"Identify all {len(crops)} card(s), in order."
        content.append({"type": "text", "text": (note + "\n" + tail).strip()})

        output_config: dict = {"format": {"type": "json_schema",
                                          "schema": IDENTIFY_SCHEMA}}
        if self.effort:
            output_config["effort"] = self.effort

        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                output_config=output_config,
                messages=[{"role": "user", "content": content}],
            )
        except Exception as e:
            print(f"[vision] identify call failed: {e}", flush=True)
            return []

        usage = getattr(resp, "usage", None)
        if usage is not None:
            print(f"[vision] identify: {len(crops)} crop(s), "
                  f"cache_read={getattr(usage, 'cache_read_input_tokens', 0)} "
                  f"in={getattr(usage, 'input_tokens', 0)}", flush=True)

        text = next((b.text for b in resp.content if b.type == "text"), "{}")
        try:
            return (json.loads(text) or {}).get("cards", []) or []
        except json.JSONDecodeError as e:
            print(f"[vision] unparseable identify reply ({e}).", flush=True)
            return []
