"""The Claude-backed advisor — extends the original recommend() into Flexo.

Two entry points:
  recommend(pack, state)  -> a Recommendation (top-3 + confidence + spoken line)
  respond(text, state)    -> Flexo's spoken reaction to the drafter, + any plan update

Prompts are kept small (pack + pool + plan + WR) because this runs 40+ times a
draft. Structured outputs guarantee parseable JSON; GIH WR shown to the drafter
is looked up locally from 17Lands, never taken from the model.
"""

import json
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
    def recommend(self, set_code, pack_no, pick_no, pack_names, state: DraftState) -> Recommendation:
        self.seventeen.ensure_set(set_code)

        # Build "Card — GIH x% | OH y% | ALSA z" lines; the model picks names,
        # we own the numbers.
        wr_by_name = {}
        pack_lines = []
        for name in pack_names:
            stats = self.seventeen.stats_for(name)
            wr_by_name[name] = stats.get("gih_wr") if stats else None
            pack_lines.append(stat_line(name, stats))

        user = persona.pick_user_prompt(
            set_code, pack_no, pick_no, pack_lines,
            state.pool_summary(), state.plan_summary(),
            data_confidence=self.seventeen.data_confidence(),
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
        return Recommendation(
            pack_number=pack_no,
            pick_number=pick_no,
            picks=picks,
            confidence=data.get("confidence", "medium"),
            spoken=data.get("spoken", "").strip(),
        )

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
