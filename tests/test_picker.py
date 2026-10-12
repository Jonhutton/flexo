"""Standalone tests for pack ranking and the 17Lands CSV loader.
No network, no API keys:

    python tests/test_picker.py

This is the layer that actually decides the pick, so the cases below are the
ones that were silently wrong before: split-card ratings, what "color fit"
means in a two-color pool, and whether commitment can ever outweigh raw win
rate late in a draft.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flexo.picker import (PickerConfig, base_win_rate, color_fit,
                          commitment_level, pool_color_weights, rank_pack,
                          rating_for)
from flexo.seventeen_lands import find_ratings_csv, parse_ratings_csv


def card(name, colors, rarity="common"):
    return {"name": name, "colors": colors, "rarity": rarity}


def rating(wr, games=5000):
    return {"gih_wr": wr, "gih_count": games}


BOMB_G = card("Green Bomb", ["G"], "mythic")
GOOD_B = card("Black Playable", ["B"])
GOOD_U = card("Blue Playable", ["U"])
GOLD_UB = card("Dimir Gold", ["U", "B"], "uncommon")
GOLD_BG = card("Golgari Gold", ["B", "G"], "uncommon")
ARTIFACT = card("Colorless Thing", [], "uncommon")

RATINGS = {
    "Green Bomb": rating(0.665),
    "Black Playable": rating(0.599),
    "Blue Playable": rating(0.590),
    "Dimir Gold": rating(0.600),
    "Golgari Gold": rating(0.600),
    "Colorless Thing": rating(0.570),
    # 17Lands lists only the front face of a transforming card.
    "Front Face": rating(0.605),
}


def ub_pool(n=20):
    """A hard two-color pool: n/2 black cards, n/2 blue."""
    return ([card(f"B{i}", ["B"]) for i in range(n // 2)]
            + [card(f"U{i}", ["U"]) for i in range(n // 2)])


# --------------------------------------------------------------------- lookups
def test_front_face_rating_fallback():
    """Scryfall says "Front // Back"; 17Lands says "Front". Without the
    fallback every DFC in the set silently drops to a rarity prior."""
    assert rating_for(RATINGS, "Front Face // Back Face") == RATINGS["Front Face"]
    assert rating_for(RATINGS, "Black Playable") == RATINGS["Black Playable"]
    assert rating_for(RATINGS, "Not In Set") is None


def test_base_win_rate_sources():
    assert base_win_rate(GOOD_B, rating(0.60))[1] == "17lands"
    assert base_win_rate(GOOD_B, None)[1] == "prior"
    # A tiny sample is blended toward the prior, not trusted outright.
    wr, src = base_win_rate(GOOD_B, rating(0.80, games=50))
    assert src == "blend" and wr < 0.70, (wr, src)


# ------------------------------------------------------------------ color fit
def test_mono_card_in_even_two_color_pool_fits():
    """The one that was wrong: in an even UB pool a mono-black card is a
    PERFECT fit, even though black holds only half the pool's total weight."""
    w = pool_color_weights(ub_pool(), RATINGS)
    assert color_fit(["B"], w) > 0.9, color_fit(["B"], w)
    assert color_fit(["U"], w) > 0.9, color_fit(["U"], w)
    assert color_fit(["G"], w) < 0.1, color_fit(["G"], w)


def test_gold_card_judged_by_coldest_color():
    w = pool_color_weights(ub_pool(), RATINGS)
    assert color_fit(["U", "B"], w) > 0.9      # both colors are on
    assert color_fit(["B", "G"], w) < 0.1      # green is stone cold


def test_colorless_and_empty_pool():
    assert color_fit([], {}) == 0.75           # artifacts always playable
    assert color_fit(["B"], {}) == 0.5         # nothing drafted: neutral


def test_commitment_ramps():
    cfg = PickerConfig()
    assert commitment_level(0, cfg) == 0.0
    assert commitment_level(3, cfg) == 0.0     # still wide open
    assert commitment_level(18, cfg) == 1.0
    assert 0.0 < commitment_level(10, cfg) < 1.0


# -------------------------------------------------------------------- ranking
def test_open_pool_takes_the_best_card():
    """Pick 1: nothing to be loyal to, so raw win rate wins."""
    ranked = rank_pack([BOMB_G, GOOD_B], [], RATINGS)
    assert ranked[0].name == "Green Bomb"


def test_commitment_beats_raw_win_rate_when_committed():
    """Pack 3, hard UB: a castable 59.9% beats an uncastable 66.5%."""
    ranked = rank_pack([BOMB_G, GOOD_B], ub_pool(20), RATINGS)
    assert ranked[0].name == "Black Playable", [(s.name, s.score) for s in ranked]


def test_third_color_penalty_applies_only_when_committed():
    committed = rank_pack([GOLD_BG], ub_pool(20), RATINGS)[0]
    assert any("splash" in r for r in committed.reasons), committed.reasons
    open_pool = rank_pack([GOLD_BG], [], RATINGS)[0]
    assert not any("splash" in r for r in open_pool.reasons)


def test_on_color_gold_has_no_splash_penalty():
    scored = rank_pack([GOLD_UB], ub_pool(20), RATINGS)[0]
    assert not any("splash" in r for r in scored.reasons), scored.reasons


def test_reasons_are_populated_for_narration():
    """persona.py reads these; an empty list means Flexo has nothing to say."""
    for s in rank_pack([BOMB_G, GOOD_B, ARTIFACT], ub_pool(20), RATINGS):
        assert s.reasons and all(isinstance(r, str) for r in s.reasons)
        assert "GIH WR" in s.reasons[0]


def test_missing_ratings_fall_back_to_priors():
    ranked = rank_pack([BOMB_G, GOOD_B], [], {})
    assert all(s.wr_source == "prior" for s in ranked)
    assert ranked[0].name == "Green Bomb"      # mythic prior > common prior


def test_ranking_is_sorted_and_complete():
    pack = [BOMB_G, GOOD_B, GOOD_U, GOLD_UB, ARTIFACT]
    ranked = rank_pack(pack, ub_pool(12), RATINGS)
    assert len(ranked) == len(pack)
    assert [s.score for s in ranked] == sorted([s.score for s in ranked], reverse=True)


# ------------------------------------------------------------------- CSV load
CSV_TEXT = (
    '"Name","Color","Rarity","# Seen","ALSA","# Picked","ATA","# GP","% GP",'
    '"GP WR","# OH","OH WR","# GD","GD WR","# GIH","GIH WR","# GNS","GNS WR","IIH"\n'
    '"Black Playable","B","C","92,945","7.54","10198","11.08","11567","18.7%",'
    '"53.2%","1971","49.6%","3120","55.2%","5,091","59.9%","6483","53.4%","-0.4pp"\n'
    '"Unplayed Card","G","R","100","9.1","10","9.0","5","1.0%",'
    '"","2","","1","","3","","4","","0.0pp"\n'
)


def test_csv_parsing():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "card-ratings-test.csv"
        p.write_text(CSV_TEXT, encoding="utf-8")
        index = parse_ratings_csv(p)

    row = index["Black Playable"]
    assert abs(row["gih_wr"] - 0.599) < 1e-9, row      # "59.9%" -> 0.599
    assert row["gih_count"] == 5091                    # "5,091" -> int
    assert abs(row["alsa"] - 7.54) < 1e-9
    assert row["rarity"] == "C" and row["color"] == "B"
    # Blank win rates stay None so the picker falls back to a prior.
    assert index["Unplayed Card"]["gih_wr"] is None


def test_csv_discovery_prefers_newest():
    with tempfile.TemporaryDirectory() as td:
        data = Path(td) / "data"
        data.mkdir()
        (data / "card-ratings-old.csv").write_text(CSV_TEXT, encoding="utf-8")
        found = find_ratings_csv(data, "FRA")
        assert found is not None and found.name == "card-ratings-old.csv"
        # An explicit path always wins.
        explicit = data / "card-ratings-old.csv"
        assert find_ratings_csv(data, "FRA", str(explicit)) == explicit
        assert find_ratings_csv(data, "FRA", str(data / "nope.csv")) is None


if __name__ == "__main__":
    test_front_face_rating_fallback()
    test_base_win_rate_sources()
    test_mono_card_in_even_two_color_pool_fits()
    test_gold_card_judged_by_coldest_color()
    test_colorless_and_empty_pool()
    test_commitment_ramps()
    test_open_pool_takes_the_best_card()
    test_commitment_beats_raw_win_rate_when_committed()
    test_third_color_penalty_applies_only_when_committed()
    test_on_color_gold_has_no_splash_penalty()
    test_reasons_are_populated_for_narration()
    test_missing_ratings_fall_back_to_priors()
    test_ranking_is_sorted_and_complete()
    test_csv_parsing()
    test_csv_discovery_prefers_newest()
    print("test_picker: all tests passed.")
