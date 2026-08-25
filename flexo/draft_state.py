"""Running draft state that persists across the whole draft.

Tracks: cards picked so far, the declared archetype/plan and color commitment,
the current pack, Flexo's latest recommendation, and the full agreement log.
This is the shared model both Flexo's prompts and the OBS overlay read from.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Recommendation:
    """One turn of Flexo's advice (also what the overlay renders)."""
    pack_number: int
    pick_number: int
    picks: list[dict]              # [{name, reason, gih_wr}]  (up to 3, best first)
    confidence: str                # high | medium | coinflip
    spoken: str                    # the <=40-word line Flexo says

    @property
    def top_name(self) -> Optional[str]:
        return self.picks[0]["name"] if self.picks else None


@dataclass
class DraftState:
    set_code: str = "Unknown"
    pool: list[str] = field(default_factory=list)     # card names picked so far
    plan: str = ""                                    # e.g. "forcing UW fliers"
    colors: str = ""                                  # e.g. "WU"
    conceded: bool = False                            # human won an argument this draft

    current_pack: list[str] = field(default_factory=list)  # card names in the live pack
    last_reco: Optional[Recommendation] = None
    disagreements: list[dict] = field(default_factory=list)

    # ------------------------------------------------------------------ mutations
    def add_picks(self, names: list[str]) -> None:
        self.pool.extend(names)

    def set_plan(self, plan: Optional[str], colors: Optional[str]) -> None:
        if plan:
            self.plan = plan
        if colors:
            self.colors = colors

    def record_turn(self, reco: Recommendation, actual_pick: Optional[str]) -> dict:
        """Log one pick outcome; returns the disagreement row (for CSV + scoreboard)."""
        agreed = bool(actual_pick and reco.top_name and actual_pick == reco.top_name)
        row = {
            "pack": reco.pack_number,
            "pick": reco.pick_number,
            "flexo_top": reco.top_name or "",
            "my_pick": actual_pick or "",
            "agreed": agreed,
            "confidence": reco.confidence,
            "reasoning": reco.spoken,
        }
        self.disagreements.append(row)
        return row

    # ------------------------------------------------------------------ views
    def plan_summary(self) -> str:
        """Compact plan/colors line for prompts and the overlay."""
        bits = []
        if self.colors:
            bits.append(f"colors {self.colors}")
        if self.plan:
            bits.append(self.plan)
        return "; ".join(bits) if bits else "undeclared (open)"

    def pool_summary(self, limit: int = 45) -> str:
        if not self.pool:
            return "(empty)"
        shown = self.pool[-limit:]
        return ", ".join(shown)
