"""Stage 2: align draft_log.csv picks to video time.

Usage: python align.py <draft_log.csv> <session_id> <video_start_iso_local> <transcript.json> <out_dir>

A constant wall-clock offset does NOT hold for this data: the CSV rows are
written with a variable delay after Flexo actually speaks (observed 2-3 min of
drift across the session). So each pick is anchored directly by fuzzy-matching
its reasoning line against the transcript over the WHOLE session, then a
longest-increasing-subsequence pass keeps only mutually consistent (monotonic)
matches. Unanchored picks are interpolated between neighboring anchors by
wall-clock proportion. Writes align.json.
"""
import csv
import json
import re
import sys
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

MATCH_THRESHOLD = 0.60
MAX_WINDOW_SEGS = 4


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", text.lower())).strip()


def main() -> None:
    log_path, session_id, start_iso, transcript_path, out_dir = sys.argv[1:6]
    out = Path(out_dir)
    video_start = datetime.fromisoformat(start_iso)

    picks = []
    with open(log_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["draft_id"] != session_id:
                continue
            ts = datetime.fromisoformat(row["timestamp"])
            picks.append(
                {
                    "pack": int(row["pack"]),
                    "pick": int(row["pick"]),
                    "flexo_top": row["flexo_top"],
                    "my_pick": row["my_pick"],
                    "agreed": row["agreed"] == "True",
                    "confidence": row["confidence"],
                    "reasoning": row["reasoning"],
                    "wall_clock": row["timestamp"],
                    "wall_s": (ts - video_start).total_seconds(),
                }
            )
    if not picks:
        sys.exit(f"no rows for session {session_id}")
    picks.sort(key=lambda p: p["wall_s"])

    transcript = json.loads(Path(transcript_path).read_text(encoding="utf-8"))
    segs = transcript["segments"]

    # global best match per pick (sliding windows of 1-4 transcript segments)
    for p in picks:
        target = norm(p["reasoning"])
        best = {"score": 0.0}
        for i in range(len(segs)):
            joined = ""
            for j in range(i, min(i + MAX_WINDOW_SEGS, len(segs))):
                joined = (joined + " " + norm(segs[j]["text"])).strip()
                if len(joined) > 2.5 * len(target):
                    break
                score = SequenceMatcher(None, target, joined).ratio()
                if score > best["score"]:
                    best = {
                        "score": round(score, 3),
                        "match_start": segs[i]["start"],
                        "match_end": segs[j]["end"],
                        "matched_text": joined[:120],
                    }
        p["match"] = best if best["score"] >= MATCH_THRESHOLD else None

    # keep only a monotonic (in time) subset of matches: weighted LIS by score
    idx = [i for i, p in enumerate(picks) if p["match"]]
    lis_best = {}
    order = sorted(idx, key=lambda i: picks[i]["wall_s"])
    dp = {i: (picks[i]["match"]["score"], None) for i in order}
    for a_pos, i in enumerate(order):
        for j in order[:a_pos]:
            if picks[j]["match"]["match_start"] < picks[i]["match"]["match_start"]:
                cand = dp[j][0] + picks[i]["match"]["score"]
                if cand > dp[i][0]:
                    dp[i] = (cand, j)
    kept_chain = []
    if dp:
        tail = max(dp, key=lambda i: dp[i][0])
        while tail is not None:
            kept_chain.append(tail)
            tail = dp[tail][1]
        kept_chain.reverse()
    dropped = [i for i in idx if i not in kept_chain]
    for i in dropped:
        picks[i]["match"] = None  # inconsistent with the monotonic chain

    anchors = [(p["wall_s"], p["match"]["match_start"], p) for p in picks if p["match"]]
    for p in picks:
        if p["match"]:
            p["video_t"] = p["match"]["match_start"]
            p["anchor"] = True
        else:
            before = [a for a in anchors if a[0] <= p["wall_s"]]
            after = [a for a in anchors if a[0] > p["wall_s"]]
            if before and after:
                (w0, v0, _), (w1, v1, _) = before[-1], after[0]
                frac = (p["wall_s"] - w0) / (w1 - w0) if w1 > w0 else 0
                p["video_t"] = round(v0 + frac * (v1 - v0), 2)
            elif before:
                w0, v0, _ = before[-1]
                p["video_t"] = round(v0 + (p["wall_s"] - w0), 2)
            else:
                w1, v1, _ = after[0]
                p["video_t"] = round(max(0, v1 - (w1 - p["wall_s"])), 2)
            p["anchor"] = False

    lags = [v - w for w, v, _ in anchors]
    frac = len(anchors) / len(picks)
    grade = "high" if frac >= 0.6 else "medium" if frac >= 0.35 else "low"

    result = {
        "session_id": session_id,
        "video_start_local": start_iso,
        "method": "per-pick text anchoring + monotonic LIS filter + interpolation",
        "picks_total": len(picks),
        "picks_anchored": len(anchors),
        "match_fraction": round(frac, 3),
        "dropped_inconsistent_matches": len(dropped),
        "lag_range_s": [round(min(lags), 1), round(max(lags), 1)] if lags else None,
        "confidence": grade,
        "picks": picks,
    }
    out.joinpath("align.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"anchored {len(anchors)}/{len(picks)} picks (dropped {len(dropped)} non-monotonic), "
        f"lag range {result['lag_range_s']}, confidence={grade}"
    )


if __name__ == "__main__":
    main()
