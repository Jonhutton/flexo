"""Arena card-ID -> name mapping.

Ported verbatim (logic-wise) from the original mtga_draft_assistant.py.
Builds a map from Scryfall's bulk data once, then caches it to disk.
"""

import json
import urllib.request
from pathlib import Path

SCRYFALL_BULK_INDEX = "https://api.scryfall.com/bulk-data"


def build_card_map(cache_path: Path) -> dict:
    """Load arena_id -> card name. Use cache if present, else fetch from Scryfall once."""
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
