"""Arena card-ID -> name mapping.

Ported verbatim (logic-wise) from the original mtga_draft_assistant.py.
Builds a map from Scryfall's bulk data once, then caches it to disk.
"""

import json
import urllib.request
from pathlib import Path

SCRYFALL_BULK_INDEX = "https://api.scryfall.com/bulk-data"


ARENA_DB_GLOBS = [
    r"C:/Program Files/Wizards of the Coast/MTGA/MTGA_Data/Downloads/Raw/Raw_CardDatabase_*.mtga",
    r"C:/Program Files (x86)/Steam/steamapps/common/MTGA/MTGA_Data/Downloads/Raw/Raw_CardDatabase_*.mtga",
    r"C:/Program Files/Epic Games/MagicTheGathering/MTGA_Data/Downloads/Raw/Raw_CardDatabase_*.mtga",
]


def arena_db_names() -> dict:
    """grpId -> English name from Arena's own local card database (SQLite).

    Scryfall lags days-to-weeks on arena_id for a new set (FRA had none at launch),
    but Arena ships every grpId it can put in a pack. Names come from TitleId, which
    is the front face for DFC/split cards -- the same convention 17Lands uses.
    Returns {} if Arena isn't installed where we look; never raises.
    """
    import glob
    import re
    import sqlite3

    paths = [p for g in ARENA_DB_GLOBS for p in glob.glob(g)]
    if not paths:
        return {}
    db = max(paths, key=lambda p: Path(p).stat().st_mtime)  # newest if several
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        rows = con.execute(
            "SELECT c.GrpId, l.Loc FROM Cards c JOIN Localizations_enUS l "
            "ON l.LocId = c.TitleId ORDER BY l.Formatted = 1"  # Formatted=1 rows last -> win
        ).fetchall()
        con.close()
    except Exception as e:
        print(f"[cards] Arena card DB unreadable ({e}); using Scryfall map only", flush=True)
        return {}
    out = {}
    for gid, name in rows:
        name = re.sub(r"<[^>]*>", "", name or "").strip()  # Arena markup: <nobr>, <sprite ...>
        if name:
            out[str(gid)] = name
    return out


def build_card_map(cache_path: Path) -> dict:
    """Load arena_id -> card name: Scryfall cache, gaps filled from Arena's local DB."""
    card_map = _scryfall_map(cache_path)
    local = arena_db_names()
    # Fill gaps only: Scryfall names are canonical ("Front // Back"), Arena's are front-face.
    added = 0
    for k, name in local.items():
        if k not in card_map:
            card_map[k] = name
            added += 1
    if added:
        print(f"[cards] +{added:,} card ids from Arena's local database", flush=True)
    return card_map


def _scryfall_map(cache_path: Path) -> dict:
    """arena_id -> card name. Use cache if present, else fetch from Scryfall once."""
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass  # corrupt cache -> rebuild

    print("Building card map from Scryfall (one-time, ~80-120MB download)...", flush=True)
    # Scryfall requires a custom User-Agent + Accept header, else it returns HTTP 400.
    headers = {"User-Agent": "FlexoDraftAssistant/1.0", "Accept": "application/json"}

    req = urllib.request.Request(SCRYFALL_BULK_INDEX, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        index = json.load(r)
    default_cards_uri = next(
        d["download_uri"] for d in index["data"] if d["type"] == "default_cards"
    )
    req2 = urllib.request.Request(default_cards_uri, headers=headers)
    with urllib.request.urlopen(req2, timeout=300) as r:
        cards = json.load(r)

    card_map = {}
    for c in cards:
        aid = c.get("arena_id")
        if aid is not None:
            card_map[str(aid)] = c.get("name", f"#{aid}")
    cache_path.write_text(json.dumps(card_map), encoding="utf-8")
    print(f"Card map built: {len(card_map):,} cards cached to {cache_path.name}", flush=True)
    return card_map


def name_of(card_map: dict, cid: str) -> str:
    return card_map.get(str(cid), f"#{cid}")  # unknown id -> show raw, never crash
