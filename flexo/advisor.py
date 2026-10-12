"""The Claude-backed advisor — extends the original recommend() into Flexo.

Two entry points:
  recommend(pack, state)  -> a Recommendation (top-3 + confidence + spoken line)
  respond(text, state)    -> Flexo's spoken reaction to the drafter, + any plan update

Prompts are kept small (pack + pool + plan + WR) because this runs 40+ times a
draft. Structured outputs guarantee parseable JSON; GIH WR shown to the drafter
is looked up locally from 17Lands, never taken from the model.
"""

import json
import re
from typing import Optional

import anthropic

from . import persona
from .draft_state import DraftState, Recommendation
from .seventeen_lands import SeventeenLands, stat_line


class Advisor:
    def __init__(self, model: str, seventeen: SeventeenLands, max_tokens: int = 200):
        self.model = model
        self.seventeen = seventeen
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    # ------------------------------------------------------------------ picks
    def recommend(self, set_code, pack_no, pick_no, pack_names, state: DraftState,
                  ranking=None, cards=None) -> Recommendation:
        """Ask Flexo for a pick.

        `ranking` (picker.ScoredCard list) and `cards` (name -> Scryfall record)
        are optional. When supplied, the pack is presented already ranked by the
        deterministic picker and the top candidates carry their verbatim oracle
        text — so the model is narrating and sanity-checking a ranking rather
        than inventing one, and it has no reason to recite an ability from
        memory. Without them this falls back to the plain stat lines.
        """
        self.seventeen.ensure_set(set_code)

        wr_by_name = {}
        for name in pack_names:
            stats = self.seventeen.stats_for(name)
            wr_by_name[name] = stats.get("gih_wr") if stats else None

        if ranking:
            pack_lines = self._ranked_lines(ranking, cards or {})
        else:
            # Build "Card — GIH x% | OH y% | ALSA z" lines; the model picks
            # names, we own the numbers.
            pack_lines = [stat_line(n, self.seventeen.stats_for(n))
                          for n in pack_names]

        user = persona.pick_user_prompt(
            set_code, pack_no, pick_no, pack_lines,
            state.pool_summary(), state.plan_summary(),
            data_confidence=self.seventeen.data_confidence(),
            ranked=bool(ranking),
        )
        data = self._call(persona.pick_system(), user, persona.PICK_SCHEMA)

        picks = []
        for p in (data.get("picks") or [])[:3]:
            name = p.get("name", "")
            picks.append({
                "name": name,
                "reason": p.get("reason", ""),
                "gih_wr": wr_by_name.get(name),   # attach our local number
            })
        spoken = data.get("spoken", "").strip()
        top = picks[0] if picks else {}
        spoken = guard_percentages(
            spoken, allowed=percents_in(user),
            top_wr=top.get("gih_wr"),
            other_names=[n for n in pack_names if n != top.get("name")],
        )
        return Recommendation(
            pack_number=pack_no,
            pick_number=pick_no,
            picks=picks,
            confidence=data.get("confidence", "medium"),
            spoken=spoken,
        )

    # Oracle text is only included for the cards Flexo might actually discuss;
    # the tail of a 14-card pack just needs a name and a number.
    ORACLE_DEPTH = 5

    def _ranked_lines(self, ranking, cards: dict) -> list:
        """Present the pack in picker order, with rules text for the top few.

        Never truncate oracle text: a clipped ability is exactly how Flexo
        would end up confidently describing a card wrong, which is the failure
        this whole path exists to prevent.
        """
        lines = []
        for i, s in enumerate(ranking, start=1):
            card = cards.get(s.name, {})
            stats = self.seventeen.stats_for(s.name)
            head = f"{i}. {s.name}"
            if card.get("mana_cost"):
                head += f" {card['mana_cost']}"
            if card.get("type_line"):
                head += f" — {card['type_line']}"
            if card.get("rarity"):
                head += f" [{card['rarity']}]"

            bits = [stat_line(s.name, stats).lstrip("- ").split("—", 1)[-1].strip()]
            bits.append(f"picker {s.score:.3f}: {'; '.join(s.reasons)}")
            line = f"- {head} — " + " | ".join(b for b in bits if b)

            if i <= self.ORACLE_DEPTH and card.get("oracle_text"):
                oracle = card["oracle_text"].replace("\n", " / ")
                line += f"\n    rules text: {oracle}"
            lines.append(line)
        return lines

    # ------------------------------------------------------------------ banter
    def respond(self, text: str, state: DraftState,
                interrupted: bool = False,
                unspoken_reco: Optional[str] = None) -> dict:
        """React to the drafter's speech. Returns {spoken, plan_update, colors, conceded}.

        interrupted:   the drafter barged in while Flexo was mid-line — allow at
                       most one clause of acknowledgment, never resume the line.
        unspoken_reco: a pick line for the CURRENT pack that never aired (barged
                       in / debounced); Flexo folds that pick into his answer.
        """
        notes = []
        if interrupted:
            notes.append(
                "NOTE: the drafter CUT YOU OFF mid-line by keying the mic. Do not "
                "repeat or resume the interrupted line."
            )
        if unspoken_reco:
            notes.append(
                "NOTE: you never delivered your pick line for the current pack "
                f"(it was going to be: \"{unspoken_reco}\"). Fold that pick advice "
                "into your answer."
            )
        user = (
            f"Your declared plan so far: {state.plan_summary()}\n"
            f"Your last recommendation was: {state.last_reco.spoken if state.last_reco else '(none)'}\n"
            + ("".join(f"{n}\n" for n in notes))
            + f"\nThe drafter just said: \"{text}\""
        )
        data = self._call(persona.banter_system(), user, persona.BANTER_SCHEMA)
        return {
            "spoken": data.get("spoken", "").strip(),
            "plan_update": data.get("plan_update"),
            "colors": data.get("colors"),
            "conceded": bool(data.get("conceded")),
        }

    # ------------------------------------------------------------------ internals
    def _call(self, system: str, user: str, schema: dict) -> dict:
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                # Static persona/rules go through prompt caching. (No-op below the
                # model's minimum cacheable size, but free and future-proof: it
                # kicks in the moment the system prompt grows past the threshold.)
                system=[{"type": "text", "text": system,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
                output_config={"format": {"type": "json_schema", "schema": schema}},
            )
            text = next((b.text for b in resp.content if b.type == "text"), "{}")
            return json.loads(text)
        except Exception as e:
            # Never let a model/network hiccup kill the live loop.
            print(f"[advisor] Claude call failed: {e}", flush=True)
            return {}


# ---------------------------------------------------------------------- stat guard
# The prompt says "never invent a stat", and on real packs (2026-10-12 dry run)
# Flexo still said "Sixty percent win rate" for a 56.4% card. A wrong number on
# camera is the one error viewers can check, so the spoken line is policed in
# code: every percentage must be one we actually handed the model.

_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate(
    "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
_NUM_WORD = r"(?:%s)(?:[- ](?:%s))?|(?:%s)" % (
    "|".join(_TENS), "|".join(w for w in _UNITS if _UNITS[w] < 10),
    "|".join(_UNITS))
_PCT = re.compile(
    r"(?P<num>\d+(?:\.\d+)?|%s)(?:\s+point\s+(?P<frac>\w+))?\s*(?:%%|percent\b|per cent\b)"
    % _NUM_WORD, re.IGNORECASE)


def _word_value(tok: str) -> Optional[float]:
    tok = tok.lower()
    try:
        return float(tok)
    except ValueError:
        pass
    parts = re.split(r"[- ]", tok)
    total = 0
    for p in parts:
        if p in _TENS:
            total += _TENS[p]
        elif p in _UNITS:
            total += _UNITS[p]
        else:
            return None
    return float(total)


def percents_in(text: str) -> list:
    """Every 'NN.N%' figure in the prompt we sent: the only numbers Flexo may say."""
    return [float(m) for m in re.findall(r"(\d+(?:\.\d+)?)%", text)]


def guard_percentages(spoken: str, allowed: list, top_wr: Optional[float] = None,
                      other_names: Optional[list] = None, tol: float = 1.0) -> str:
    """Fix or drop any percentage in `spoken` that isn't within `tol` of a real one.

    A bad number in a sentence about the top pick is replaced with the top
    pick's real GIH WR (rounded, spoken as digits). If the sentence names
    another card, or there's no real number to use, the sentence is dropped.
    """
    if not spoken:
        return spoken

    def ok(v: float) -> bool:
        return any(abs(v - a) <= tol for a in allowed)

    others = [n.split(" // ")[0].lower() for n in (other_names or [])]
    sentences = re.split(r"(?<=[.!?])\s+", spoken)
    kept = []
    for sent in sentences:
        bad = [m for m in _PCT.finditer(sent)
               if (v := _word_value(m.group("num"))) is not None and not ok(v)]
        if not bad:
            kept.append(sent)
            continue
        names_other = any(o and o in sent.lower() for o in others)
        if top_wr is None or names_other:
            print(f"[advisor] dropped invented stat: {sent!r}", flush=True)
            continue
        real = f"{round(top_wr * 100)} percent"
        fixed = _PCT.sub(
            lambda m: m.group(0) if ok(_word_value(m.group("num")) or -1.0) else real,
            sent)
        print(f"[advisor] corrected invented stat: {sent!r} -> {fixed!r}", flush=True)
        kept.append(fixed)
    return " ".join(kept).strip()
