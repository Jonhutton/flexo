"""Stage 5: chapters.txt, shorts_candidates.md, editor_report.md.

Usage: python report.py <align.json> <transcript.json> <segments.json> <out_dir>
"""
import json
import sys
from pathlib import Path

SMUG_WORDS = [
    "you're welcome", "adorable", "obviously", "frankly", "bored", "embarrassing",
    "sad", "window dressing", "begging", "dumpster fire", "nonsense", "restraint",
    "lesson", "practically drafts itself", "doesn't overthink", "moving on", "next.",
    "whispers politely", "gasping", "kindling", "actually", "sight unseen",
]


def hms(t: float) -> str:
    t = max(t, 0)
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def main() -> None:
    align_path, transcript_path, segments_path, out_dir = sys.argv[1:5]
    out = Path(out_dir)
    align = json.loads(Path(align_path).read_text(encoding="utf-8"))
    transcript = json.loads(Path(transcript_path).read_text(encoding="utf-8"))
    segments = json.loads(Path(segments_path).read_text(encoding="utf-8"))
    kept = segments["kept"]
    picks = align["picks"]

    # source-time -> rough-cut-time mapping
    def rough_t(src: float) -> float:
        acc = 0.0
        for k in kept:
            if src < k["start"]:
                return acc  # fell in a cut: snap to next kept segment
            if src <= k["end"]:
                return acc + (src - k["start"])
            acc += k["end"] - k["start"]
        return acc

    # ---------- chapters.txt ----------
    lines = []
    reboot_keeps = [k for k in kept if "reboot" in k["reason"]]
    if reboot_keeps:
        lines.append((rough_t(reboot_keeps[0]["start"]), "Arena won't cooperate (reboot)"))
    first = min(picks, key=lambda p: p["video_t"])
    lines.append((rough_t(first["video_t"]), "Draft start"))
    for pack in (2, 3):
        pp = [p for p in picks if p["pack"] == pack]
        if pp:
            lines.append((rough_t(min(p["video_t"] for p in pp)), f"Pack {pack}"))
    n = 0
    for p in sorted(picks, key=lambda p: p["video_t"]):
        if not p["agreed"]:
            n += 1
            lines.append(
                (rough_t(p["video_t"]), f"Flexo overruled #{n} (P{p['pack']}P{p['pick']}: {p['my_pick']} over {p['flexo_top']})")
            )
    last = max(picks, key=lambda p: p["video_t"])
    end = p["match"]["match_end"] if last.get("match") else last["video_t"] + 10
    lines.append((rough_t(end), "Deck complete"))
    lines.sort()
    dedup, seen = [], set()
    for t, label in lines:
        tc = hms(t)
        while tc in seen:  # YouTube needs unique ascending stamps
            t += 1
            tc = hms(t)
        seen.add(tc)
        dedup.append(f"{tc} {label}")
    out.joinpath("chapters.txt").write_text("\n".join(dedup) + "\n", encoding="utf-8")

    # ---------- shorts_candidates.md ----------
    segs = transcript["segments"]
    flexo_spans = [(p["match"]["match_start"], p["match"]["match_end"], p) for p in picks if p.get("match")]

    def in_flexo(s):
        mid = (s["start"] + s["end"]) / 2
        return any(a - 1 <= mid <= b + 1 for a, b, _ in flexo_spans)

    cands = []
    # (a) barge-ins can't be observed directly (mono mix: overlapping voices
    # merge into one ASR segment). Proxy: your voice landing within 1.5s of a
    # Flexo line ending — instant comebacks.
    for s in segs:
        if in_flexo(s):
            continue
        for a, b, p in flexo_spans:
            gap = s["start"] - b
            if 0 <= gap <= 1.5 and len(s["text"].split()) >= 4:
                cands.append(
                    {
                        "prio": 0, "score": -gap,
                        "start": max(0, a - 2), "end": s["end"] + 2,
                        "why": f"instant comeback ({gap:.1f}s) after Flexo's P{p['pack']}P{p['pick']} line — \"{s['text'][:60]}\"",
                    }
                )
                break
    # (b) condescension: score Flexo's matched lines for smugness
    for a, b, p in flexo_spans:
        text = p["reasoning"].lower()
        hits = [w for w in SMUG_WORDS if w in text]
        if hits:
            cands.append(
                {
                    "prio": 1, "score": len(hits) + (0 if p["agreed"] else 1.5),
                    "start": max(0, a - 2), "end": b + 2,
                    "why": f"peak smugness ({', '.join(hits[:3])}) on P{p['pack']}P{p['pick']}: \"{p['reasoning'][:80]}...\"",
                }
            )
    # (c) reboot sequence
    reboots = [k for k in kept if "reboot" in k["reason"]]
    if reboots:
        start = reboots[0]["start"]
        end = min(reboots[-1]["end"], start + 58)
        cands.append(
            {"prio": 2, "score": 1, "start": start, "end": end,
             "why": "the Arena reboot sequence — tech-failure cold open"}
        )

    for c in cands:  # clamp to 60s
        if c["end"] - c["start"] > 60:
            c["end"] = c["start"] + 60
    cands.sort(key=lambda c: (c["prio"], -c["score"]))
    top = []
    # category mix: 2 comebacks, 2 smugness, 1 reboot (backfill if a category
    # comes up short)
    quota = {0: 2, 1: 2, 2: 1}
    if sum(min(quota[p], sum(1 for c in cands if c["prio"] == p)) for p in quota) < 5:
        quota = {0: 5, 1: 5, 2: 1}
    for c in cands:
        if quota[c["prio"]] <= 0:
            continue
        if all(abs(c["start"] - t["start"]) > 20 for t in top):
            top.append(c)
            quota[c["prio"]] -= 1
        if len(top) == 5:
            break

    md = [
        "# Shorts candidates (source timecodes)",
        "",
        "_Note: true barge-ins (you talking over Flexo) can't be detected from this_",
        "_recording — mic and TTS share one audio track, so overlapping voices merge_",
        "_into a single transcript segment. 'Instant comeback' is the closest proxy._",
        "",
    ]
    for i, c in enumerate(top, 1):
        md.append(
            f"{i}. **{hms(c['start'])} – {hms(c['end'])}** ({c['end'] - c['start']:.0f}s) — {c['why']}"
        )
    out.joinpath("shorts_candidates.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---------- editor_report.md ----------
    dur = transcript["duration"]
    cut_s = segments["cut_total_s"]
    kept_s = segments["kept_total_s"]
    unmatched = [p for p in picks if not p.get("match")]
    rep = [
        "# Editor report",
        "",
        f"- Source: 28:06 OBS recording, session `{align['session_id']}`",
        f"- Kept **{hms(kept_s)}** / cut **{hms(cut_s)}** of {hms(dur)} total "
        f"({cut_s / dur * 100:.0f}% removed)",
        f"- Log↔video alignment: a constant wall-clock offset did NOT hold — the CSV rows are "
        f"written {-align['lag_range_s'][1]:.0f}–{-align['lag_range_s'][0]:.0f}s after Flexo "
        f"actually speaks, drifting across the session. Method used instead: "
        f"{align['method']}. {align['picks_anchored']}/{align['picks_total']} picks anchored "
        f"directly by their spoken line → confidence **{align['confidence']}**",
        f"- Silence-only fallback used: **{segments['silence_only_fallback']}**",
        "",
        "## Flagged for your review",
        "- Pack 1 picks 1–3 are missing from draft_log.csv (consistent with the Arena "
        "restart); the reboot window was kept wherever there is speech, silence inside it was cut.",
        "- Speaker attribution is text-based (transcript lines matching the CSV reasoning = Flexo; "
        "everything else = you). Any unscripted Flexo banter would be misattributed to you — "
        "worst case it inflates KEEP, never deletes content.",
    ]
    if unmatched:
        rep.append(
            "- Picks whose Flexo line was NOT found in the transcript (kept using estimated "
            "durations): " + ", ".join(f"P{p['pack']}P{p['pick']}" for p in unmatched)
        )
    rep += [
        "- Cuts are frame-accurate re-encodes (h264_amf CQP 16 ≈ visually lossless, same GPU "
        "encoder family OBS recorded with); pure stream copy was not possible because OBS "
        "keyframes are seconds apart and cut points land between them.",
        "",
        "See cutlist.csv for every segment decision with the rule that fired.",
    ]
    out.joinpath("editor_report.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("chapters.txt, shorts_candidates.md, editor_report.md written")


if __name__ == "__main__":
    main()
