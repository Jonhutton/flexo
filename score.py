#!/usr/bin/env python3
"""Award a match to the Human vs AI scoreboard (run after a match, off-camera or on).

    python score.py --award human --pick 3      # you won match/draft 3
    python score.py --award ai --pick 3          # Flexo won match/draft 3
    python score.py --show                       # print the running tally
"""

import argparse
from datetime import datetime
from pathlib import Path

import yaml

from flexo.scoreboard import Scoreboard

ROOT = Path(__file__).resolve().parent


def _scoreboard() -> Scoreboard:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    paths = cfg.get("paths", {})
    return Scoreboard(
        csv_path=ROOT / paths.get("draft_log_csv", "draft_log.csv"),
        score_path=ROOT / paths.get("score_json", "human_vs_ai_score.json"),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Human vs AI scoreboard")
    ap.add_argument("--award", choices=["human", "ai"], help="who won")
    ap.add_argument("--pick", default="", help="match/draft number or note")
    ap.add_argument("--show", action="store_true", help="print current tally")
    args = ap.parse_args()

    sb = _scoreboard()

    if args.award:
        score = sb.award(args.award, note=str(args.pick),
                         timestamp=datetime.now().isoformat(timespec="seconds"))
        print(f"Awarded to {args.award}. HUMAN {score['human']} - {score['ai']} AI")
    elif args.show:
        score = sb.load_score()
        print(f"HUMAN {score['human']} - {score['ai']} AI  "
              f"({len(score['log'])} matches recorded)")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
