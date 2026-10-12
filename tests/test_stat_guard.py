"""Flexo may only say win-rate percentages we actually gave him.

    python tests/test_stat_guard.py

Real failure this pins (2026-10-12 dry run): "Konstrari Charm. Sixty percent
win rate..." for a card at 56.4% GIH WR.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flexo.advisor import guard_percentages, percents_in

PROMPT = ("- Konstrari Charm — GIH 56.4% | OH 55.0% | ALSA 4.1\n"
          "- Icy Reception — GIH 58.5% | OH 57.0% | ALSA 3.0")
ALLOWED = percents_in(PROMPT)


def g(text, top_wr=0.564, others=("Icy Reception",)):
    return guard_percentages(text, ALLOWED, top_wr, list(others))


def test_prompt_numbers_extracted():
    assert ALLOWED == [56.4, 55.0, 58.5, 57.0]


def test_real_numbers_pass_untouched():
    for line in ("Take it. 56.4% says so.",
                 "Fifty-six percent, three modes. Next.",
                 "Fifty-eight point five percent beats you, mortal.",
                 "I am one hundred percent sure."):          # idiom, not a stat
        assert g(line) == line, line


def test_invented_number_about_top_pick_is_corrected():
    assert g("Sixty percent win rate, three modes. Next.") == \
        "56 percent win rate, three modes. Next."
    assert g("Konstrari Charm at 61%. Next.") == "Konstrari Charm at 56 percent. Next."


def test_invented_number_naming_another_card_is_dropped():
    assert g("Konstrari Charm. Icy Reception is a mere 40%. Next.") == \
        "Konstrari Charm. Next."


def test_no_real_number_means_drop():
    assert g("Seventy-two percent. Done.", top_wr=None) == "Done."


if __name__ == "__main__":
    for t in (test_prompt_numbers_extracted, test_real_numbers_pass_untouched,
              test_invented_number_about_top_pick_is_corrected,
              test_invented_number_naming_another_card_is_dropped,
              test_no_real_number_means_drop):
        t()
    print("test_stat_guard: all tests passed.")
