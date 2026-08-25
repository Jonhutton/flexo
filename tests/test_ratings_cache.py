"""Standalone tests for the 17Lands ratings cache + data-confidence logic.
No network, no API keys:

    python tests/test_ratings_cache.py

A fake ratings payload is written straight into the on-disk cache; _fetch is
stubbed to fail loudly if the cache ever misses when it shouldn't.
"""

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flexo.seventeen_lands import (MIN_GIH_SAMPLE, SeventeenLands, fmt_alsa,
                                   fmt_wr, stat_line)


def fake_rows(gih_count: int) -> list[dict]:
    return [
        {"name": "Lightning Bolt", "ever_drawn_win_rate": 0.612,
         "opening_hand_win_rate": 0.598, "avg_pick": 1.8, "avg_seen": 1.4,
         "rarity": "common", "color": "R", "ever_drawn_game_count": gih_count},
        {"name": "Giant Growth", "ever_drawn_win_rate": 0.541,
         "opening_hand_win_rate": None, "avg_pick": 6.2, "avg_seen": 5.9,
         "rarity": "common", "color": "G", "ever_drawn_game_count": gih_count},
    ]


def make(tmp: Path, rows, fmt="QuickDraft") -> SeventeenLands:
    s = SeventeenLands(data_dir=tmp, fmt=fmt, cache_hours=12)
    if rows is not None:
        (tmp / f"TST_{fmt}.json").write_text(json.dumps(rows), encoding="utf-8")
    s._fetch = lambda set_code, f: (_ for _ in ()).throw(
        AssertionError(f"unexpected network fetch for {set_code}/{f}"))
    return s


def test_cache_hit_no_fetch(tmp: Path):
    s = make(tmp, fake_rows(5000))
    assert s.ensure_set("TST") is True
    stats = s.stats_for("Lightning Bolt")
    assert stats["gih_wr"] == 0.612
    assert stats["oh_wr"] == 0.598
    assert stats["alsa"] == 1.4
    assert stats["gih_count"] == 5000


def test_data_confidence_levels(tmp: Path):
    s = make(tmp, fake_rows(MIN_GIH_SAMPLE * 2))
    s.ensure_set("TST")
    assert s.data_confidence() == "ok"

    s2 = make(tmp, fake_rows(10))       # tiny early-set samples
    # overwrite the cache written above
    (tmp / "TST_QuickDraft.json").write_text(json.dumps(fake_rows(10)),
                                             encoding="utf-8")
    s2.ensure_set("TST")
    assert s2.data_confidence() == "low"


def test_no_data_is_none_confidence(tmp: Path):
    s = SeventeenLands(data_dir=tmp, fmt="QuickDraft", cache_hours=12)
    s._fetch = lambda set_code, f: None      # 17Lands has nothing for this set
    assert s.ensure_set("NOPE") is False
    assert s.data_confidence() == "none"
    assert s.stats_for("Anything") is None


def test_stale_cache_refetches(tmp: Path):
    p = tmp / "TST_QuickDraft.json"
    p.write_text(json.dumps(fake_rows(100)), encoding="utf-8")
    import os
    old = time.time() - 100 * 3600
    os.utime(p, (old, old))                  # cache is 100h old, TTL is 12h
    s = SeventeenLands(data_dir=tmp, fmt="QuickDraft", cache_hours=12)
    fetched = {"called": False}

    def fetch(set_code, f):
        fetched["called"] = True
        return fake_rows(9000)

    s._fetch = fetch
    s.ensure_set("TST")
    assert fetched["called"], "stale cache should trigger a refetch"
    assert s.stats_for("Lightning Bolt")["gih_count"] == 9000


def test_formatters():
    assert fmt_wr(0.573) == "57.3%"
    assert fmt_wr(None) == "--"
    assert fmt_alsa(3.25) == "3.2" or fmt_alsa(3.25) == "3.3"   # rounding
    assert fmt_alsa(None) == "--"
    line = stat_line("Lightning Bolt",
                     {"gih_wr": 0.612, "oh_wr": 0.598, "alsa": 1.4})
    assert "GIH 61.2%" in line and "OH 59.8%" in line and "ALSA 1.4" in line
    assert stat_line("Mystery Card", None) == "- Mystery Card — no data"


if __name__ == "__main__":
    tests = [test_cache_hit_no_fetch, test_data_confidence_levels,
             test_no_data_is_none_confidence, test_stale_cache_refetches]
    for t in tests:
        with tempfile.TemporaryDirectory() as td:
            t(Path(td))
    test_formatters()
    print("test_ratings_cache: all tests passed.")
