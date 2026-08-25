"""Per-set card list + fuzzy name validation — the guardrail on vision output.

Pulls the current set's card names from Scryfall (cached locally per set,
one small file per set in the data dir), then validates every card name the
vision model extracted:

  exact match          -> pass through canonical name
  front-face match     -> canonicalize to the full "Front // Back" name
  fuzzy near-miss      -> auto-correct (minor misreads, missing accents)
  below threshold      -> FLAG it, never guess

This is what fixes the "Flexo confidently wrong about a card" failure mode:
a name that doesn't exist in the set can't reach the persona layer.
"""

import json
import re
import unicodedata
import urllib.parse
import urllib.request
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

_SEARCH_URL = "https://api.scryfall.com/cards/search"
_HEADERS = {"User-Agent": "FlexoDraftAssistant/1.0", "Accept": "application/json"}


# --------------------------------------------------------------------------- fetch
def fetch_set_names(set_code: str) -> list[str]:
    """All card names in a set from Scryfall (paginated search). Raises on failure."""
    names: list[str] = []
    query = f"e:{set_code.lower()}"
    url = f"{_SEARCH_URL}?{urllib.parse.urlencode({'q': query, 'unique': 'cards'})}"
    while url:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=30) as r:
            page = json.load(r)
        names.extend(c["name"] for c in page.get("data", []) if c.get("name"))
        url = page.get("next_page") if page.get("has_more") else None
    return sorted(set(names))


def load_set_names(set_code: str, data_dir: Path) -> list[str]:
    """Cached set card list; fetched from Scryfall once per set."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
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

    @property
    def ok(self) -> bool:
        return self.name is not None


class CardValidator:
    def __init__(self, set_code: str, data_dir: Path, threshold: float = 0.82):
        self.set_code = set_code.upper()
        self.threshold = threshold
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
        exact = self._by_norm.get(key)
        if exact:
            return Validated(raw=raw, name=exact, score=1.0,
                             corrected=(exact != raw))
        # Fuzzy: best ratio over normalized names (set lists are ~300 cards; fine).
        best_key, best_score = None, 0.0
        for cand in self._norm_keys:
            score = SequenceMatcher(None, key, cand).ratio()
            if score > best_score:
                best_key, best_score = cand, score
        if best_key is not None and best_score >= self.threshold:
            return Validated(raw=raw, name=self._by_norm[best_key],
                             score=best_score, corrected=True)
        return Validated(raw=raw, name=None, score=best_score, corrected=False)

    def validate_all(self, raws: list[str]) -> list[Validated]:
        return [self.validate(r) for r in raws]
