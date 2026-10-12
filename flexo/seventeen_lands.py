"""17Lands card-ratings fetch + cache (NEW — no equivalent in the original script).

Pulls per-card stats for the active set and format from the public
17Lands card_ratings endpoint, indexes them by card name (using the Scryfall
map that the log parser already relies on), and caches them on disk.

Set-agnostic by design: the caller passes whatever set code the Arena log
reported, so Flexo works on "whatever sets come up" in Quick Draft. If a set's
data isn't published yet, lookups return None and the advisor simply runs on
card names — never crashes.
"""

import csv
import json
import time
from pathlib import Path
from typing import Optional

import requests

_ENDPOINT = "https://www.17lands.com/card_ratings/data"
_HEADERS = {"User-Agent": "FlexoDraftAssistant/1.0 (personal draft tool)"}

# Median games-in-hand sample below which a set's data counts as "low" —
# Flexo cites numbers but hedges (in character) instead of trusting them.
MIN_GIH_SAMPLE = 500

# 17Lands CSV column -> our field. Headers drift between exports, so match
# case-insensitively and ignore anything we don't recognize.
_CSV_FIELDS = {
    "gih wr": "gih_wr",
    "# gih": "gih_count",
    "oh wr": "oh_wr",
    "gp wr": "gp_wr",
    "alsa": "alsa",
    "ata": "ata",
    "color": "color",
    "rarity": "rarity",
}

# Fraction of a CSV's card names that must appear in the set before we trust
# it. Guards against auto-discovering last set's export and silently ranking
# the whole draft on the wrong numbers.
_CSV_MATCH_FLOOR = 0.5


def _csv_num(v):
    """'54.0%' -> 0.540, '1,620' -> 1620.0, '-0.4pp' -> -0.004, '' -> None."""
    if v is None:
        return None
    s = str(v).strip().replace(",", "")
    if s in ("", "-", "NA", "N/A"):
        return None
    try:
        if s.endswith("%"):
            return float(s[:-1]) / 100.0
        if s.endswith("pp"):
            return float(s[:-2]) / 100.0
        return float(s)
    except ValueError:
        return s          # non-numeric columns (Color, Rarity) pass through


def parse_ratings_csv(path: Path) -> dict:
    """Parse a 17lands.com Card Ratings export into our name-indexed shape."""
    out: dict = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name = (row.get("Name") or row.get("name") or "").strip()
            if not name:
                continue
            rec: dict = {}
            for key, raw in row.items():
                field = _CSV_FIELDS.get((key or "").strip().lower())
                if field:
                    rec[field] = _csv_num(raw)
            count = rec.get("gih_count")
            rec["gih_count"] = int(count) if isinstance(count, float) else count
            out[name] = rec
    return out


def find_ratings_csv(data_dir: Path, set_code: str,
                     explicit: Optional[str] = None) -> Optional[Path]:
    """Locate a 17Lands CSV export for this set.

    Checked in order: an explicit config path, conventional per-set filenames,
    then any `card-ratings*.csv` lying in the data dir or the repo root (that's
    the name the 17Lands download button produces, set name and all).
    """
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None

    data_dir = Path(data_dir)
    roots = [data_dir, data_dir.parent]
    code = set_code.lower()
    for root in roots:
        for stem in (f"17lands_{code}", code, set_code.upper()):
            p = root / f"{stem}.csv"
            if p.exists():
                return p

    loose: list[Path] = []
    for root in roots:
        if root.is_dir():
            loose.extend(sorted(root.glob("card-ratings*.csv")))
    # Newest first: a fresh download for the current set should win.
    loose.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return loose[0] if loose else None


class SeventeenLands:
    def __init__(self, data_dir: Path, fmt: str = "QuickDraft", cache_hours: float = 12.0,
                 csv_path: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.fmt = fmt
        self.cache_seconds = cache_hours * 3600
        self.csv_path = csv_path
        self._loaded_set: Optional[str] = None
        self._loaded_fmt: Optional[str] = None
        self._by_name: dict[str, dict] = {}

    # ------------------------------------------------------------------ public
    def ensure_set(self, set_code: str,
                   known_names: Optional[list] = None) -> bool:
        """Load stats for `set_code`. Returns True if usable win-rate data was found.

        A downloaded CSV export wins over the API: as of Oct 2026 the public
        card_ratings endpoint returns card lists with null win rates to script
        callers, so the CSV is the only reliable source of real numbers.

        Failing that, tries the configured format, then falls back to
        PremierDraft, whose GIH WR is the community-standard proxy for card
        quality when a set has no Quick Draft data.

        `known_names` (the set's Scryfall card list) is used to sanity-check a
        auto-discovered CSV actually belongs to this set.
        """
        if set_code == self._loaded_set and self._by_name:
            return True
        if not set_code or set_code == "Unknown":
            return False

        index = self._load_csv(set_code, known_names)
        used_fmt = "csv"

        if not _has_wr(index):
            index = self._load_format(set_code, self.fmt)
            used_fmt = self.fmt
            if not _has_wr(index) and self.fmt != "PremierDraft":
                premier = self._load_format(set_code, "PremierDraft")
                if _has_wr(premier):
                    index, used_fmt = premier, "PremierDraft"
                    print(f"[17Lands] {set_code}/{self.fmt} has no win-rate data; "
                          f"using PremierDraft numbers as a proxy.", flush=True)

        self._by_name = index
        self._loaded_set = set_code
        self._loaded_fmt = used_fmt
        return _has_wr(index)

    def _load_csv(self, set_code: str, known_names: Optional[list]) -> dict:
        """Load a CSV export for this set, if one is present and plausible."""
        path = find_ratings_csv(self.data_dir, set_code, self.csv_path)
        if path is None:
            return {}
        try:
            index = parse_ratings_csv(path)
        except Exception as e:
            print(f"[17Lands] could not read {path.name} ({e}); ignoring it.",
                  flush=True)
            return {}
        if not _has_wr(index):
            print(f"[17Lands] {path.name} has no win-rate column; ignoring it.",
                  flush=True)
            return {}

        # An auto-discovered file could easily be the previous set's export.
        if known_names:
            known = set(known_names)
            known |= {n.split(" // ")[0].strip() for n in known}
            hit = sum(1 for n in index if n in known)
            share = hit / max(1, len(index))
            if share < _CSV_MATCH_FLOOR:
                print(f"[17Lands] {path.name} matches only {share:.0%} of "
                      f"{set_code.upper()}'s cards — looks like a different set. "
                      f"Ignoring it; set seventeen_lands.csv_path to override.",
                      flush=True)
                return {}

        rated = sum(1 for v in index.values() if v.get("gih_wr") is not None)
        print(f"[17Lands] using {path.name}: {len(index)} cards, "
              f"{rated} with GIH WR.", flush=True)
        return index

    def data_confidence(self) -> str:
        """How much to trust the loaded set's numbers: 'ok' | 'low' | 'none'.

        Early in a set (or for sets 17Lands hasn't published), sample sizes are
        tiny or absent — Flexo should lean on his own judgment and say so in
        character rather than fabricating stats.
        """
        if not self._by_name or not _has_wr(self._by_name):
            return "none"
        counts = [v.get("gih_count") or 0 for v in self._by_name.values()
                  if v.get("gih_wr") is not None]
        if not counts:
            return "none"
        counts.sort()
        median = counts[len(counts) // 2]
        return "ok" if median >= MIN_GIH_SAMPLE else "low"

    def stats_for(self, card_name: str) -> Optional[dict]:
        """Return {'gih_wr','oh_wr','ata','alsa','rarity','color','gih_count'}
        for a card name, or None."""
        if not self._by_name:
            return None
        row = self._by_name.get(card_name)
        if row is None and " // " in card_name:      # split / DFC front face
            row = self._by_name.get(card_name.split(" // ")[0])
        return row

    # ------------------------------------------------------------------ internals
    def _load_format(self, set_code: str, fmt: str) -> dict:
        """Return the name-indexed stats for one set+format (cache or network)."""
        raw = self._load_cached(set_code, fmt)
        if raw is None:
            raw = self._fetch(set_code, fmt)
            if raw is not None:
                self._save_cache(set_code, fmt, raw)
        return self._index(raw) if raw else {}

    def _cache_path(self, set_code: str, fmt: str) -> Path:
        return self.data_dir / f"{set_code}_{fmt}.json"

    def _load_cached(self, set_code: str, fmt: str):
        p = self._cache_path(set_code, fmt)
        if not p.exists():
            return None
        if (time.time() - p.stat().st_mtime) > self.cache_seconds:
            return None  # stale -> refetch
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _save_cache(self, set_code: str, fmt: str, raw):
        try:
            self._cache_path(set_code, fmt).write_text(json.dumps(raw), encoding="utf-8")
        except Exception:
            pass  # cache is best-effort

    def _fetch(self, set_code: str, fmt: str):
        params = {"expansion": set_code, "format": fmt}
        try:
            r = requests.get(_ENDPOINT, params=params, headers=_HEADERS, timeout=30)
            r.raise_for_status()
            data = r.json()
            return data if isinstance(data, list) and data else None
        except Exception as e:
            print(f"[17Lands] could not fetch {set_code}/{fmt}: {e}", flush=True)
            return None

    @staticmethod
    def _index(raw: list) -> dict:
        out = {}
        for row in raw:
            name = row.get("name")
            if not name:
                continue
            out[name] = {
                # ever_drawn_win_rate is 17Lands' GIH WR (games-in-hand win rate).
                "gih_wr": row.get("ever_drawn_win_rate"),
                "oh_wr": row.get("opening_hand_win_rate"),
                "ata": row.get("avg_pick"),    # average taken at
                "alsa": row.get("avg_seen"),   # average last seen at
                "rarity": row.get("rarity"),
                "color": row.get("color"),
                # sample size behind GIH WR — tiny early in a set
                "gih_count": row.get("ever_drawn_game_count"),
            }
        return out


def _has_wr(index: dict) -> bool:
    """True if any card in the index has a non-null GIH win rate."""
    return any(v.get("gih_wr") is not None for v in index.values())


def fmt_wr(gih_wr: Optional[float]) -> str:
    """Format a 0..1 GIH WR as a percent string, or '--' if missing."""
    if gih_wr is None:
        return "--"
    return f"{gih_wr * 100:.1f}%"


def fmt_alsa(alsa: Optional[float]) -> str:
    """Format ALSA (average last seen at) to one decimal, or '--'."""
    if alsa is None:
        return "--"
    return f"{alsa:.1f}"


def stat_line(name: str, stats: Optional[dict]) -> str:
    """One '- Card — GIH x% | OH y% | ALSA z' line for the pick prompt."""
    if not stats:
        return f"- {name} — no data"
    return (f"- {name} — GIH {fmt_wr(stats.get('gih_wr'))}"
            f" | OH {fmt_wr(stats.get('oh_wr'))}"
            f" | ALSA {fmt_alsa(stats.get('alsa'))}")
