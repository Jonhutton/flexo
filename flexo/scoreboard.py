"""Disagreement log (draft_log.csv) + running Human vs AI tally (json).

- Every pick is appended to draft_log.csv: timestamp, pack, pick, Flexo's top
  card, the drafter's actual pick, whether they agreed, confidence, reasoning.
- human_vs_ai_score.json holds a running tally across drafts, awarded manually
  after each match via score.py.
"""

import csv
import json
from pathlib import Path
from typing import Optional

CSV_FIELDS = [
    "draft_id", "timestamp", "pack", "pick",
    "flexo_top", "my_pick", "agreed", "confidence", "reasoning",
]


class Scoreboard:
    def __init__(self, csv_path: Path, score_path: Path):
        self.csv_path = Path(csv_path)
        self.score_path = Path(score_path)

    # ------------------------------------------------------------------ per-pick CSV
    def log_pick(self, draft_id: str, timestamp: str, row: dict) -> None:
        new_file = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow({
                "draft_id": draft_id,
                "timestamp": timestamp,
                "pack": row.get("pack", ""),
                "pick": row.get("pick", ""),
                "flexo_top": row.get("flexo_top", ""),
                "my_pick": row.get("my_pick", ""),
                "agreed": row.get("agreed", ""),
                "confidence": row.get("confidence", ""),
                "reasoning": row.get("reasoning", ""),
            })

    def read_rows(self, draft_id: Optional[str] = None) -> list[dict]:
        if not self.csv_path.exists():
            return []
        with self.csv_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if draft_id:
            rows = [r for r in rows if r["draft_id"] == draft_id]
        return rows

    def print_disagreements(self, draft_id: Optional[str] = None) -> None:
        rows = self.read_rows(draft_id)
        disagreed = [r for r in rows if str(r.get("agreed")).lower() == "false"]
        print("\n=== Flexo vs You - disagreements ===")
        if not disagreed:
            print("You agreed with Flexo on every pick. Suspicious.")
        else:
            for r in disagreed:
                print(f"P{r['pack']}p{r['pick']}: Flexo said {r['flexo_top']!r}, "
                      f"you took {r['my_pick']!r}  ({r['confidence']}) - {r['reasoning']}")
        print(f"Disagreed on {len(disagreed)} of {len(rows)} picks.\n")

    # ------------------------------------------------------------------ draft summary
    def write_summary(self, draft_id: str) -> Optional[Path]:
        """Emit one draft's picks as JSON next to the CSV, for joining against a
        17Lands draft log later — so the scoreboard can be settled by real data
        on camera. Returns the path, or None if the draft has no rows."""
        rows = self.read_rows(draft_id)
        if not rows:
            return None
        agreed = sum(1 for r in rows if str(r.get("agreed")).lower() == "true")
        summary = {
            "draft_id": draft_id,
            "picks": [
                {
                    "pack": int(r["pack"]) if r.get("pack") else None,
                    "pick": int(r["pick"]) if r.get("pick") else None,
                    "flexo_top": r.get("flexo_top", ""),
                    "my_pick": r.get("my_pick", ""),
                    "agreed": str(r.get("agreed")).lower() == "true",
                    "confidence": r.get("confidence", ""),
                }
                for r in rows
            ],
            "totals": {"picks": len(rows), "agreed": agreed,
                       "overruled": len(rows) - agreed},
        }
        path = self.csv_path.parent / f"draft_summary_{draft_id}.json"
        try:
            path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[scoreboard] could not write summary: {e}", flush=True)
            return None
        return path

    # ------------------------------------------------------------------ running score
    def load_score(self) -> dict:
        if self.score_path.exists():
            try:
                return json.loads(self.score_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"human": 0, "ai": 0, "log": []}

    def save_score(self, score: dict) -> None:
        self.score_path.write_text(json.dumps(score, indent=2), encoding="utf-8")

    def award(self, side: str, note: str, timestamp: str) -> dict:
        if side not in ("human", "ai"):
            raise ValueError("side must be 'human' or 'ai'")
        score = self.load_score()
        score[side] += 1
        score["log"].append({"ts": timestamp, "side": side, "note": note})
        self.save_score(score)
        return score
