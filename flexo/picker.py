"""Deterministic pack ranking — pure data in, ranked list + reasons out.

No vision, no model call. This is what decides the pick order; Flexo's persona
layer only narrates the result. Keeping the ranking out of the LLM makes it
explainable, reproducible, free, and impossible to hallucinate.

    score = base win rate (17Lands GIH WR, or a rarity prior)
          + color fit vs. the pool, scaled by how committed the pool is
          - penalties for gold cards taken early / a third color taken late

`reasons` is the handoff to persona.py: Flexo explains a pick ("best raw card
but we're already deep in UB") instead of just announcing one.
"""

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Optional

COLORS = ["W", "U", "B", "R", "G"]

# Rarity priors used when 17Lands has no win rate (new set, outage, tiny sample).
RARITY_PRIOR = {"mythic": 0.580, "rare": 0.565, "uncommon": 0.552, "common": 0.538}
UNKNOWN_PRIOR = 0.540

# GIH games below which a card's win rate is blended toward its rarity prior.
MIN_GAMES = 500


@dataclass
class PickerConfig:
    """Ranking tunables. Defaults are deliberate — see the notes on each."""

    # Full-commitment swing between a perfect color fit and a total misfit.
    # Must be able to outweigh a realistic win-rate gap late in the draft: a
    # castable 54% common really should beat an uncastable 60% bomb in pack 3.
    # 0.16 gives +/-0.08 at full commitment, +/-0.04 at half.
    color_weight: float = 0.16

    # Pool size at which color commitment starts to bite, and the size at which
    # it reaches full strength. Pack 1 picks 1-3 stay effectively open.
    commit_start: int = 3
    commit_full: int = 18

    # Flat penalty for a multicolor card taken while still open.
    gold_early_penalty: float = 0.004

    # Extra penalty for a card that drags in a THIRD color once committed.
    # This is the "don't take a splash on pick 30" rule.
    third_color_penalty: float = 0.03

    @classmethod
    def from_cfg(cls, cfg: Optional[dict]) -> "PickerConfig":
        p = ((cfg or {}).get("picker") or {})
        return cls(
            color_weight=float(p.get("color_weight", cls.color_weight)),
            commit_start=int(p.get("commit_start", cls.commit_start)),
            commit_full=int(p.get("commit_full", cls.commit_full)),
            gold_early_penalty=float(p.get("gold_early_penalty", cls.gold_early_penalty)),
            third_color_penalty=float(p.get("third_color_penalty", cls.third_color_penalty)),
        )


@dataclass
class ScoredCard:
    name: str
    score: float
    base_wr: float
    wr_source: str                 # "17lands" | "blend" | "prior"
    color_fit: float               # 0..1
    commitment: float              # 0..1
    colors: list
    rarity: str
    reasons: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- inputs
def rating_for(ratings: dict, name: str) -> Optional[dict]:
    """Look up a card's 17Lands row by Scryfall name.

    Scryfall names a split/DFC card "Front // Back"; 17Lands lists only the
    front face. Without this fallback every transforming card in the set
    silently drops to a rarity prior — which is most of the uncommons in a
    modern set, so the ranking would quietly go blind on them.
    """
    row = ratings.get(name)
    if row is None and " // " in name:
        row = ratings.get(name.split(" // ")[0].strip())
    return row


def base_win_rate(card: dict, rating: Optional[dict]) -> tuple[float, str]:
    """Card + 17Lands row -> (win rate, where it came from).

    A card with a handful of games is worse than no data at all if trusted
    blindly, so low-sample win rates are blended toward the rarity prior.
    """
    prior = RARITY_PRIOR.get(card.get("rarity") or "", UNKNOWN_PRIOR)
    if not rating or rating.get("gih_wr") is None:
        return prior, "prior"
    wr = float(rating["gih_wr"])
    games = int(rating.get("gih_games") or rating.get("gih_count") or 0)
    if games >= MIN_GAMES:
        return wr, "17lands"
    weight = games / MIN_GAMES
    return weight * wr + (1 - weight) * prior, "blend"


def pool_color_weights(pool: list, ratings: dict) -> dict:
    """Weight each color by the QUALITY of what's already been taken in it.

    Two good black cards commit you harder than five bad ones, which is how
    drafters actually read their own pool.
    """
    weights: dict = defaultdict(float)
    for card in pool:
        wr, _ = base_win_rate(card, rating_for(ratings, card.get("name", "")))
        strength = max(0.0, wr - 0.50) + 0.01    # floor so any pick counts a little
        for c in card.get("colors") or []:
            weights[c] += strength
    return dict(weights)


def commitment_level(pool_size: int, cfg: PickerConfig) -> float:
    """How hard the pool should pull picks. ~0 while open, 1.0 once locked in."""
    span = max(1, cfg.commit_full - cfg.commit_start)
    return max(0.0, min(1.0, (pool_size - cfg.commit_start) / span))


def color_fit(card_colors: list, weights: dict) -> float:
    """How well a card's colors match the pool, 0..1.

    Measured against the pool's STRONGEST color, not against the pool total.
    That matters: in an even two-color pool a mono-colored card in either of
    those colors is a perfect fit, but it only holds half the total weight —
    scoring it against the total would call a premium on-color card "neutral"
    and quietly hand every pick back to raw win rate.

    A gold card is judged by its COLDEST color, because it has to be castable
    in all of them.
    """
    if not card_colors:
        return 0.75          # colorless: always playable, never a perfect fit
    if not weights:
        return 0.5           # nothing drafted yet: everything is neutral

    strongest = max(weights.values())
    if strongest <= 0:
        return 0.5

    if len(card_colors) > 1:
        share = min(weights.get(c, 0.0) for c in card_colors)
    else:
        share = weights.get(card_colors[0], 0.0)

    return max(0.0, min(1.0, share / strongest))


def _is_third_color(card_colors: list, weights: dict) -> bool:
    """True if the card needs a color outside the pool's established top two."""
    if not card_colors or not weights:
        return False
    main = {c for c, _ in sorted(weights.items(), key=lambda kv: -kv[1])[:2]}
    return any(c not in main for c in card_colors)


# --------------------------------------------------------------------------- ranking
def rank_pack(pack: list, pool: list, ratings: dict,
              cfg: Optional[PickerConfig] = None) -> list:
    """Rank a pack given the current pool. Best first."""
    cfg = cfg or PickerConfig()
    weights = pool_color_weights(pool, ratings)
    commit = commitment_level(len(pool), cfg)

    out: list = []
    for card in pack:
        name = card.get("name", "")
        wr, source = base_win_rate(card, rating_for(ratings, name))
        colors = card.get("colors") or []
        fit = color_fit(colors, weights)

        score = wr + commit * (fit - 0.5) * cfg.color_weight
        reasons = [f"GIH WR {wr:.1%} ({source})"]
        if commit > 0.2:
            reasons.append(f"color fit {fit:.0%} at {commit:.0%} commitment")

        if len(colors) > 1 and commit < 0.3:
            score -= cfg.gold_early_penalty
            reasons.append("gold card early: small flexibility penalty")

        if commit >= 0.5 and _is_third_color(colors, weights):
            score -= cfg.third_color_penalty * commit
            reasons.append("third color this late: splash penalty")

        if not colors:
            reasons.append("colorless: fits any lane")

        out.append(ScoredCard(
            name=name,
            score=round(score, 4),
            base_wr=round(wr, 4),
            wr_source=source,
            color_fit=round(fit, 3),
            commitment=round(commit, 2),
            colors=colors,
            rarity=card.get("rarity", ""),
            reasons=reasons,
        ))

    out.sort(key=lambda s: -s.score)
    return out
