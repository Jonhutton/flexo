"""Standalone tests for the card-name validation layer. No network, no API keys:

    python tests/test_validation.py

A fake Scryfall cache is written to a temp data dir, so CardValidator never
touches the network. This layer is the guardrail that keeps a misread card
name from ever reaching Flexo's mouth.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flexo.set_cards import CardValidator, _norm

FAKE_SET = ["Lightning Bolt", "Serra's Angel", "Fear // Loathing",
            "Llanowar Elves", "Aetherize", "Giant Growth"]


def make_validator(tmp: Path, threshold: float = 0.82) -> CardValidator:
    (tmp / "TST_names.json").write_text(json.dumps(FAKE_SET), encoding="utf-8")
    return CardValidator("TST", tmp, threshold=threshold)


def test_exact_match(v: CardValidator):
    r = v.validate("Lightning Bolt")
    assert r.ok and r.name == "Lightning Bolt" and r.score == 1.0 and not r.corrected


def test_case_and_punctuation(v: CardValidator):
    r = v.validate("serras angel")          # missing apostrophe + lowercase
    assert r.ok and r.name == "Serra's Angel", r
    r = v.validate("AETHERIZE")             # normalization handles case
    assert r.ok and r.name == "Aetherize", r


def test_fuzzy_correction(v: CardValidator):
    r = v.validate("Lightnng Bolt")         # vision misread: dropped letter
    assert r.ok and r.name == "Lightning Bolt" and r.corrected, r
    r = v.validate("Lianowar Elves")        # ll -> li misread
    assert r.ok and r.name == "Llanowar Elves", r


def test_below_threshold_is_flagged(v: CardValidator):
    r = v.validate("Zzyzx Prime Directive")
    assert not r.ok and r.name is None, r   # flagged, never guessed


def test_front_face_maps_to_full_name(v: CardValidator):
    r = v.validate("Fear")                  # vision sees the front face only
    assert r.ok and r.name == "Fear // Loathing", r


def test_validate_all_partitions(v: CardValidator):
    results = v.validate_all(["Giant Growth", "Total Nonsense Here", "Aetherize"])
    good = [r.name for r in results if r.ok]
    bad = [r.raw for r in results if not r.ok]
    assert good == ["Giant Growth", "Aetherize"] and bad == ["Total Nonsense Here"]


def test_norm():
    assert _norm("Serra's  Angel!") == "serras angel"
    assert _norm("Æther-ize") != ""         # accents fold, never crash


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        v = make_validator(Path(td))
        test_exact_match(v)
        test_case_and_punctuation(v)
        test_fuzzy_correction(v)
        test_below_threshold_is_flagged(v)
        test_front_face_maps_to_full_name(v)
        test_validate_all_partitions(v)
        test_norm()
    print("test_validation: all tests passed.")
