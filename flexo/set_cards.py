"""Per-set card data + fuzzy name validation — the guardrail on vision output.

Pulls the current set from Scryfall (cached locally per set in the data dir)
and validates every card name the vision model extracted:

  exact match          -> pass through canonical name
  front-face match     -> canonicalize to the full "Front // Back" name
  fuzzy near-miss      -> auto-correct (minor misreads, missing accents)
  near-miss but shaky  -> keep the suggestion, mark it RECHECK
  below threshold      -> FLAG it, never guess

This is what fixes the "Flexo confidently wrong about a card" failure mode:
a name that doesn't exist in the set can't reach the persona layer.

Two caches live here, both keyed by set code:
  {SET}_cards.json   full card records — oracle text, colors, rarity, cost
  {SET}_names.json   names only (legacy; still read if the richer file is absent)

Oracle text matters as much as the name: Flexo quotes abilities from THIS
file, never from the model's memory, so he cannot invent a rules text.
"""

import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz, process

_SEARCH_URL = "https://api.scryfall.com/cards/search"
_HEADERS = {"User-Agent": "FlexoDraftAssistant/1.0", "Accept": "application/json"}

# Scryfall asks for ~100ms between requests. https://scryfall.com/docs/api
_PAGE_DELAY_S = 0.12


# --------------------------------------------------------------------------- fetch
def _search(query: str) -> list[dict]:
    """Run a paginated Scryfall search and return every raw card record."""
    out: list[dict] = []
    url = f"{_SEARCH_URL}?{urllib.parse.urlencode({'q': query, 'unique': 'cards'})}"
    while url:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=30) as r:
            page = json.load(r)
        out.extend(page.get("data", []))
        url = page.get("next_page") if page.get("has_more") else None
        if url:
            time.sleep(_PAGE_DELAY_S)
    return out


def _slim(card: dict) -> dict:
    """Keep only what Flexo needs, flattening double-faced and split cards."""
    faces = card.get("card_faces") or []
    front = faces[0] if faces else card
    oracle = card.get("oracle_text")
    if oracle is None and faces:
        oracle = " // ".join(f.get("oracle_text", "") for f in faces)
    return {
        "name": card.get("name", ""),
        "mana_cost": card.get("mana_cost") or front.get("mana_cost", ""),
        "cmc": card.get("cmc", 0),
        "type_line": card.get("type_line", ""),
        "oracle_text": oracle or "",
        "colors": (card.get("colors") if card.get("colors") is not None
                   else front.get("colors", []) or []),
        "color_identity": card.get("color_identity", []),
        "rarity": card.get("rarity", ""),
        "power": front.get("power"),
        "toughness": front.get("toughness"),
        "keywords": card.get("keywords", []),
        "arena_id": card.get("arena_id"),
    }


def fetch_set_names(set_code: str) -> list[str]:
    """All card names in a set from Scryfall. Raises on failure."""
    names = [c["name"] for c in _search(f"e:{set_code.lower()}") if c.get("name")]
    return sorted(set(names))


def fetch_set_cards(set_code: str, booster_only: bool = True) -> list[dict]:
    """Full card records for a set. Booster-only by default: showcase and
    promo printings share a name with the booster card and only bloat the
    candidate list the vision model has to choose from."""
    query = f"e:{set_code.lower()}" + (" is:booster" if booster_only else "")
    try:
        raw = _search(query)
    except urllib.error.HTTPError as e:
        if e.code == 404 and booster_only:
            print(f"[cards] no is:booster cards for {set_code.upper()}; "
                  f"retrying without that filter.", flush=True)
            return fetch_set_cards(set_code, booster_only=False)
        raise
    if not raw and booster_only:
        return fetch_set_cards(set_code, booster_only=False)
    return [_slim(c) for c in raw if c.get("name")]


def load_set_cards(set_code: str, data_dir: Path) -> dict:
    """Cached name -> full card record for a set. Fetched from Scryfall once.

    This is the source of truth for oracle text, colors, and rarity — both the
    picker and Flexo's narration read from here.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache = data_dir / f"{set_code.upper()}_cards.json"
    if cache.exists():
        try:
            cards = json.loads(cache.read_text(encoding="utf-8"))
            if cards:
                return {c["name"]: c for c in cards}
        except Exception:
            pass   # corrupt cache -> refetch
    print(f"[cards] fetching {set_code.upper()} card data from Scryfall...", flush=True)
    cards = fetch_set_cards(set_code)
    try:
        cache.write_text(json.dumps(cards, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"[cards] could not write {cache.name} ({e}); continuing.", flush=True)
    print(f"[cards] {len(cards)} cards cached to {cache.name}", flush=True)
    return {c["name"]: c for c in cards}


def fetch_cards_by_name(names: list[str]) -> list[dict]:
    """Exact-name lookup for specific cards, batched (Scryfall allows 75/call)."""
    out: list[dict] = []
    url = "https://api.scryfall.com/cards/collection"
    for i in range(0, len(names), 75):
        batch = [{"name": n} for n in names[i:i + 75]]
        body = json.dumps({"identifiers": batch}).encode("utf-8")
        req = urllib.request.Request(
            url, data=body,
            headers={**_HEADERS, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                page = json.load(r)
        except Exception as e:
            print(f"[cards] name lookup batch failed ({e}); skipping it.", flush=True)
            continue
        out.extend(page.get("data", []))
        time.sleep(_PAGE_DELAY_S)
    return out


def supplement_from_names(set_code: str, data_dir: Path,
                          names: list[str]) -> dict:
    """Pull in draftable cards that live under a DIFFERENT Scryfall set code.

    Modern sets ship a bonus sheet — Special Guests (`spg`) and friends — whose
    cards appear in this set's draft boosters but are not in `e:<set>`. They
    are real picks, so leaving them out means the vision model has no candidate
    to match them to and the picker has no oracle text or colors for them.

    `names` is the authoritative draftable list, normally the 17Lands ratings
    index, so this self-corrects per set instead of hardcoding bonus-sheet
    codes. Returns the merged name -> card map and updates the set cache.
    """
    cards = load_set_cards(set_code, data_dir)
    known = set(cards)
    # 17Lands lists only the front face of a split/DFC card; those are already
    # covered by the set fetch, so don't go looking for them by name.
    fronts = {n.split(" // ")[0].strip() for n in known}
    missing = sorted({n for n in names if n and n not in known and n not in fronts})
    if not missing:
        return cards

    print(f"[cards] {len(missing)} draftable card(s) not in e:{set_code.upper()} "
          f"(bonus sheet); fetching by name...", flush=True)
    extra = [_slim(c) for c in fetch_cards_by_name(missing) if c.get("name")]
    if not extra:
        return cards

    for c in extra:
        cards.setdefault(c["name"], c)
    cache = Path(data_dir) / f"{set_code.upper()}_cards.json"
    try:
        cache.write_text(json.dumps(list(cards.values()), ensure_ascii=False),
                         encoding="utf-8")
    except Exception as e:
        print(f"[cards] could not update {cache.name} ({e}); continuing.", flush=True)
    found = {c["name"] for c in extra}
    still = [n for n in missing if n not in found]
    print(f"[cards] added {len(extra)} bonus-sheet card(s); "
          f"set pool is now {len(cards)}.", flush=True)
    if still:
        print(f"[cards] still unresolved (ignored): {still[:8]}", flush=True)
    return cards


def load_set_names(set_code: str, data_dir: Path) -> list[str]:
    """Cached set card list; fetched from Scryfall once per set.

    Prefers the richer {SET}_cards.json when it exists so a set is never
    fetched twice, and falls back to the names-only cache.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    rich = data_dir / f"{set_code.upper()}_cards.json"
    if rich.exists():
        try:
            cards = json.loads(rich.read_text(encoding="utf-8"))
            names = sorted({c["name"] for c in cards if c.get("name")})
            if names:
                return names
        except Exception:
            pass

    cache = data_dir / f"{set_code.upper()}_names.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            pass   # corrupt cache -> refetch
    print(f"[cards] fetching {set_code.upper()} card list from Scryfall...", flush=True)
    names = fetch_set_names(set_code)
    cache.write_text(json.dumps(names), encoding="utf-8")
    print(f"[cards] {len(names)} card names cached to {cache.name}", flush=True)
    return names


# --------------------------------------------------------------------------- matching
def _norm(name: str) -> str:
    """Fold case/accents/punctuation so 'Serras Angel' ~ \"Serra's Angel\"."""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9 ]", "", s.lower())
    return re.sub(r"\s+", " ", s).strip()


@dataclass
class Validated:
    """One vision-extracted name after validation."""
    raw: str                    # what the vision model read
    name: Optional[str]         # canonical card name, or None if unmatched
    score: float                # 1.0 exact, else fuzzy ratio
    corrected: bool             # fuzzy/front-face fix was applied
    suggestion: Optional[str] = None   # best guess when below threshold
    status: str = "unknown"            # ok | recheck | unknown

    @property
    def ok(self) -> bool:
        return self.name is not None


class CardValidator:
    """Fuzzy-match vision reads against the set list.

    Three outcomes rather than two. A read that lands just short of the accept
    threshold is almost always a real card the model half-read, so it is kept
    as a `recheck` with its best guess: the crop pipeline re-asks about exactly
    those slots with a tighter name-banner crop instead of throwing the card
    away. Anything below `recheck_threshold` is still flagged and never guessed.
    """

    def __init__(self, set_code: str, data_dir: Path, threshold: float = 0.82,
                 recheck_threshold: float = 0.60):
        self.set_code = set_code.upper()
        self.threshold = threshold
        self.recheck_threshold = recheck_threshold
        self.names = load_set_names(set_code, data_dir)
        self._by_norm = {_norm(n): n for n in self.names}
        # Front faces of split/DFC cards: "Fear // Loathing" -> "fear" maps to full name
        for n in self.names:
            if " // " in n:
                front = _norm(n.split(" // ")[0])
                self._by_norm.setdefault(front, n)
        self._norm_keys = list(self._by_norm.keys())

    def validate(self, raw: str) -> Validated:
        key = _norm(raw)
        if not key:
            return Validated(raw=raw, name=None, score=0.0, corrected=False,
                             status="unknown")

        exact = self._by_norm.get(key)
        if exact:
            return Validated(raw=raw, name=exact, score=1.0,
                             corrected=(exact != raw), suggestion=exact,
                             status="ok")

        # Fuzzy over normalized names. rapidfuzz's `ratio` is the same edit
        # similarity difflib computes, just vectorized — scores are unchanged,
        # which matters because the thresholds here were tuned against it.
        hit = process.extractOne(key, self._norm_keys, scorer=fuzz.ratio)
        if hit is None:
            return Validated(raw=raw, name=None, score=0.0, corrected=False,
                             status="unknown")
        best_key, best_score = hit[0], hit[1] / 100.0
        best_name = self._by_norm[best_key]

        if best_score >= self.threshold:
            return Validated(raw=raw, name=best_name, score=best_score,
                             corrected=True, suggestion=best_name, status="ok")
        if best_score >= self.recheck_threshold:
            # Close but not trusted: hand the guess back for a second look.
            return Validated(raw=raw, name=None, score=best_score,
                             corrected=False, suggestion=best_name,
                             status="recheck")
        return Validated(raw=raw, name=None, score=best_score, corrected=False,
                         status="unknown")

    def validate_all(self, raws: list[str]) -> list[Validated]:
        return [self.validate(r) for r in raws]
