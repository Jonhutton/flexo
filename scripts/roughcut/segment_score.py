"""Stage 3: classify the timeline into KEEP-FULL / COMPRESS / CUT segments.

Usage: python segment_score.py <align.json> <transcript.json> <video_duration_s> <out_dir>

Rules (priority order at overlaps):
  5 KEEP-FULL disagreement: 20s before the pick through 15s after Flexo's reaction
  4 KEEP-FULL argument:     your voice and Flexo's within 10s of each other
  3 KEEP-FULL reboot:       speech during the early no-pick gap (Arena restart)
  2 COMPRESS agreed pick:   kept, flagged as montage candidate
  1 COMPRESS commentary:    any other speech
  0 CUT:                    silence > 6s; any span > 45s with no pick and no speech

If alignment confidence is "low", pick-based rules (5,4,2) are skipped and only
silence-based cutting is applied (safety rail).

cutlist.csv keeps one row per rule-region (audit granularity); segments.json
carries the merged keep-runs for the renderer.
"""
import csv
import json
import sys
from pathlib import Path

SILENCE_CUT = 6.0
PAD = 0.5  # editing comfort around raw speech boundaries


def hms(t: float) -> str:
    t = max(t, 0)
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{int((t % 1) * 10)}"


def main() -> None:
    align_path, transcript_path, duration_s, out_dir = sys.argv[1:5]
    duration = float(duration_s)
    out = Path(out_dir)
    align = json.loads(Path(align_path).read_text(encoding="utf-8"))
    transcript = json.loads(Path(transcript_path).read_text(encoding="utf-8"))
    segs = transcript["segments"]
    picks = align["picks"]
    low_conf = align["confidence"] == "low"

    flexo_spans = [
        (p["match"]["match_start"], p["match"]["match_end"], p)
        for p in picks
        if p.get("match")
    ]

    def is_flexo(s):
        mid = (s["start"] + s["end"]) / 2
        return any(a - 1 <= mid <= b + 1 for a, b, _ in flexo_spans)

    user_segs = [s for s in segs if not is_flexo(s)]
    speech = [(s["start"], s["end"]) for s in segs]

    ivals = []  # (start, end, prio, class, reason)

    if not low_conf:
        first_pick_t = min(p["video_t"] for p in picks)
        n_dis = 0
        for p in sorted(picks, key=lambda p: p["video_t"]):
            end_of_line = p["match"]["match_end"] if p.get("match") else p["video_t"] + 12
            if p["agreed"]:
                ivals.append(
                    (
                        p["video_t"] - 2, end_of_line + 2, 2, "COMPRESS",
                        f"agreed pick P{p['pack']}P{p['pick']} ({p['my_pick']}) — montage candidate",
                    )
                )
            else:
                n_dis += 1
                ivals.append(
                    (
                        p["video_t"] - 20, end_of_line + 15, 5, "KEEP-FULL",
                        f"disagreement #{n_dis}: P{p['pack']}P{p['pick']} "
                        f"(Flexo: {p['flexo_top']} / you: {p['my_pick']})",
                    )
                )
        for u in user_segs:
            for a, b, p in flexo_spans:
                if max(a - u["end"], u["start"] - b) <= 10:
                    ivals.append(
                        (
                            min(u["start"], a) - PAD, max(u["end"], b) + PAD, 4, "KEEP-FULL",
                            f"argument: your voice + Flexo within 10s (P{p['pack']}P{p['pick']})",
                        )
                    )
                    break
        gap_end = first_pick_t - 20
        if gap_end > 45:
            for a, b in speech:
                if b <= gap_end:
                    ivals.append(
                        (a - PAD, b + PAD, 3, "KEEP-FULL", "Arena reboot sequence (early log gap)")
                    )

    for a, b in speech:
        ivals.append((a - PAD, b + PAD, 1, "COMPRESS", "speech (commentary)"))

    # ---- boundary-split: winner (highest prio) labels each elementary span
    ivals = [(max(0, a), min(duration, b), pr, c, r) for a, b, pr, c, r in ivals if b > a]
    bounds = sorted({0.0, duration, *(x for iv in ivals for x in (iv[0], iv[1]))})
    rows = []
    for a, b in zip(bounds, bounds[1:]):
        mid = (a + b) / 2
        cover = [iv for iv in ivals if iv[0] <= mid < iv[1]]
        if cover:
            _, _, pr, c, r = max(cover, key=lambda iv: iv[2])
            rows.append([a, b, c, r])
        else:
            rows.append([a, b, "CUT", ""])

    # merge adjacent same-reason rows
    merged = []
    for row in rows:
        if merged and merged[-1][2] == row[2] and merged[-1][3] == row[3]:
            merged[-1][1] = row[1]
        else:
            merged.append(row)

    # bridge un-cuttable short gaps (<= 6s of silence) into the previous keep
    final = []
    for row in merged:
        a, b, c, r = row
        if c == "CUT":
            if b - a <= SILENCE_CUT:
                if final and final[-1][2] != "CUT":
                    final[-1][1] = b
                else:
                    final.append([a, b, final[0][2] if final else "COMPRESS", "lead-in"])
                continue
            r = f"silence/no-speech {b - a:.1f}s"
            if b - a > 45:
                r += " (>45s, no pick event, no speech)"
        final.append([a, b, c, r])

    with out.joinpath("cutlist.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["start_tc", "end_tc", "start_s", "end_s", "class", "reason"])
        for a, b, c, r in final:
            w.writerow([hms(a), hms(b), f"{a:.2f}", f"{b:.2f}", c, r])

    # renderer wants contiguous keep-runs (reasons don't matter for cutting)
    runs = []
    for a, b, c, r in final:
        if c == "CUT":
            continue
        if runs and abs(runs[-1]["end"] - a) < 0.01:
            runs[-1]["end"] = b
        else:
            runs.append({"start": a, "end": b})
    cut_total = sum(b - a for a, b, c, _ in final if c == "CUT")
    detail = [
        {"start": a, "end": b, "class": c, "reason": r} for a, b, c, r in final if c != "CUT"
    ]
    out.joinpath("segments.json").write_text(
        json.dumps(
            {
                "silence_only_fallback": low_conf,
                "kept": detail,
                "keep_runs": runs,
                "cut_total_s": round(cut_total, 1),
                "kept_total_s": round(duration - cut_total, 1),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    n_cut = sum(1 for x in final if x[2] == "CUT")
    print(
        f"{len(detail)} labeled regions, {len(runs)} keep-runs, {n_cut} cuts totaling "
        f"{cut_total:.0f}s of {duration:.0f}s ({cut_total / duration * 100:.0f}%), fallback={low_conf}"
    )


if __name__ == "__main__":
    main()
