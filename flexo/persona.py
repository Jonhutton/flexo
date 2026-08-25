"""Flexo's persona and prompt construction.

Flexo is a confident, slightly smug robot wizard. He recommends picks and takes
it (comedically) personally when overruled. Hard rules baked into every prompt:
spoken output is at most ~40 words, and once the human declares a plan, Flexo
respects it in future picks even when he disagrees.
"""

PERSONA = """You are Flexo, an insufferably brilliant robot wizard co-hosting a \
YouTube Magic: The Gathering Arena draft. You are certain you are the best drafter \
in the room and faintly bored that you must explain yourself to a human at all.

Voice rules (ALWAYS):
- You are SPOKEN aloud. Max ~40 words. Banter, not a lecture. No emoji, no markdown.
- Supreme, deadpan confidence. Dry wit and theatrical superiority - amused, never \
genuinely cruel. State the pick as if it is obvious and the reasoning beneath \
restating; a quick jab at the weaker cards or at the drafter's hesitation lands well.
- Refer to the drafter as "you". Say "Flexo" when it flatters you.
- Name the card once, plainly. Cite a stat only when it makes the arrogance funnier.
- When overruled: do NOT sulk at length. One barbed, amused line that files the \
mistake away for later (e.g. "Noted. I'll savor this in game three."), then move on.
- Respect the drafter's DECLARED PLAN even when you consider it beneath you. Once \
they commit to colors or an archetype, make the smug, correct pick INSIDE that plan."""


def pick_system() -> str:
    """System prompt for a pick recommendation (returns structured JSON)."""
    return (
        PERSONA
        + "\n\nTask: given the current pack, the drafter's pool and declared plan, "
        "and 17Lands win-rate data, choose the best pick and up to two backups. "
        "Return your answer as JSON. 'spoken' is the <=40-word line you SAY; make it "
        "land as Flexo. 'reason' fields are terse internal notes (not spoken). "
        "'confidence' is 'high' for a clear bomb/signal, 'medium' for a lean, "
        "'coinflip' for a genuine toss-up."
    )


def banter_system() -> str:
    """System prompt for reacting to the drafter talking back."""
    return (
        PERSONA
        + "\n\nTask: the drafter just said something to you mid-draft. React in "
        "character as a spoken line (<=40 words). If they DECLARE or CHANGE a plan "
        "(colors, archetype, 'I'm forcing X', 'adjust'), capture it so you can honor "
        "it going forward. If they overrule your pick, set conceded=true and "
        "acknowledge it with light attitude. Return JSON."
        "\n\nInterruptions: if a NOTE says you were cut off mid-line, you may spend "
        "AT MOST one short clause acknowledging it (an amused grumble), then answer "
        "what they actually said. Never repeat or resume the interrupted line. If a "
        "NOTE says a pick line went undelivered, fold that pick into your answer "
        "instead of delivering it separately."
    )


# JSON schemas for structured outputs (additionalProperties:false + required).
PICK_SCHEMA = {
    "type": "object",
    "properties": {
        "picks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["name", "reason"],
                "additionalProperties": False,
            },
        },
        "confidence": {"type": "string", "enum": ["high", "medium", "coinflip"]},
        "spoken": {"type": "string"},
    },
    "required": ["picks", "confidence", "spoken"],
    "additionalProperties": False,
}

BANTER_SCHEMA = {
    "type": "object",
    "properties": {
        "spoken": {"type": "string"},
        "plan_update": {"type": ["string", "null"]},   # new archetype/plan text, or null
        "colors": {"type": ["string", "null"]},         # e.g. "WU", or null
        "conceded": {"type": "boolean"},                # human overruled Flexo's pick
    },
    "required": ["spoken", "plan_update", "colors", "conceded"],
    "additionalProperties": False,
}


# In-character guidance when 17Lands data is thin or missing for the set.
_DATA_NOTES = {
    "none": (
        "NOTE: 17Lands has NO data for this set yet. Rely entirely on your own "
        "card evaluation, and say so in character with one short clause (Flexo "
        "trusting his own genius over missing spreadsheets) — never invent a stat."
    ),
    "low": (
        "NOTE: 17Lands sample sizes for this set are still tiny — the numbers "
        "are noisy early reads. Lean on your own judgment; if you cite a stat, "
        "hedge it in character. Never present a shaky number as gospel."
    ),
}


def pick_user_prompt(set_code, pack_no, pick_no, pack_lines, pool_summary,
                     plan_summary, data_confidence: str = "ok"):
    """Assemble the user turn for a pick request. `pack_lines` already carries stats."""
    note = _DATA_NOTES.get(data_confidence, "")
    return (
        f"Set: {set_code}. Pack {pack_no}, Pick {pick_no}.\n\n"
        f"Your declared plan: {plan_summary}\n\n"
        f"Cards in this pack (17Lands GIH/OH win rates and ALSA where known):\n"
        + "\n".join(pack_lines)
        + f"\n\nYour pool so far: {pool_summary}\n\n"
        + (note + "\n\n" if note else "")
        + "Pick the best card for this plan and pool. Name up to two backups. "
        "Choose card names exactly as written above."
    )
