"""Standalone tests for the vision pipeline's validation + event flow.
No screen, no network, no API keys:

    python tests/test_vision_pipeline.py

Exercises VisionPipeline.process_result() — the same path the dry-run and the
live hotkey use after extraction — with a fake set list and recorded callbacks.
Covers: non-pick screens, flagged names, dedupe, and my-pick detection by
diffing the drafted-cards sidebar between captures.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flexo.set_cards import CardValidator
from flexo.vision import VisionResult
from flexo.vision_pipeline import VisionPipeline

FAKE_SET = ["Lightning Bolt", "Giant Growth", "Llanowar Elves",
            "Serra's Angel", "Aetherize"]


def make_pipeline(tmp: Path):
    (tmp / "TST_names.json").write_text(json.dumps(FAKE_SET), encoding="utf-8")
    validator = CardValidator("TST", tmp, threshold=0.82)
    packs, picks = [], []
    p = VisionPipeline(
        set_code="TST",
        capture=None,          # never used via process_result
        extractor=None,
        validator=validator,
        on_pack=packs.append,
        on_my_pick=picks.append,
    )
    return p, packs, picks


def result(**kw) -> VisionResult:
    base = dict(screen_type="draft_pick", pack_number=1, pick_number=1,
                cards_in_pack=[], cards_picked_so_far=[], confidence_notes="")
    base.update(kw)
    return VisionResult(**base)


def test_non_pick_screen_is_silent(tmp: Path):
    p, packs, picks = make_pipeline(tmp)
    out = p.process_result(result(screen_type="deck_build",
                                  cards_in_pack=["Lightning Bolt"]))
    assert out is None and not packs and not picks


def test_pack_validated_and_flagged(tmp: Path):
    p, packs, picks = make_pipeline(tmp)
    out = p.process_result(result(
        cards_in_pack=["Lightnng Bolt", "Giant Growth", "Zzyzx Garbage Read"]))
    assert out is not None
    assert out.card_names == ["Lightning Bolt", "Giant Growth"]   # corrected + kept
    assert out.flagged == ["Zzyzx Garbage Read"]                  # never guessed
    assert packs and packs[0] is out


def test_dedupe_same_decision(tmp: Path):
    p, packs, picks = make_pipeline(tmp)
    r = result(cards_in_pack=["Lightning Bolt", "Giant Growth"])
    assert p.process_result(r) is not None
    assert p.process_result(r) is None       # same pack/pick/cards: one take only
    assert len(packs) == 1


def test_my_pick_detected_from_sidebar_diff(tmp: Path):
    p, packs, picks = make_pipeline(tmp)
    p.process_result(result(pick_number=1,
                            cards_in_pack=["Lightning Bolt", "Giant Growth"],
                            cards_picked_so_far=["Serra's Angel"]))
    assert picks == []                        # first capture is just the baseline
    p.process_result(result(pick_number=2,
                            cards_in_pack=["Aetherize", "Llanowar Elves"],
                            cards_picked_so_far=["Serra's Angel", "Lightning Bolt"]))
    assert picks == ["Lightning Bolt"]        # the newly-appeared sidebar card


def test_multiple_new_picks_not_attributed(tmp: Path):
    p, packs, picks = make_pipeline(tmp)
    p.process_result(result(pick_number=1, cards_in_pack=["Aetherize"],
                            cards_picked_so_far=["Serra's Angel"]))
    p.process_result(result(pick_number=4, cards_in_pack=["Giant Growth"],
                            cards_picked_so_far=["Serra's Angel", "Lightning Bolt",
                                                 "Llanowar Elves"]))
    assert picks == []                        # 2 new cards: can't pin one pick


if __name__ == "__main__":
    tests = [test_non_pick_screen_is_silent, test_pack_validated_and_flagged,
             test_dedupe_same_decision, test_my_pick_detected_from_sidebar_diff,
             test_multiple_new_picks_not_attributed]
    for t in tests:
        with tempfile.TemporaryDirectory() as td:
            t(Path(td))
    print("test_vision_pipeline: all tests passed.")
