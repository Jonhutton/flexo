"""Stage 1: transcribe the session audio with word-level timestamps.

Usage: python transcribe.py <audio.wav> <out_dir>
Writes transcript.json and transcript.txt.
"""
import json
import sys
import time
from pathlib import Path

from faster_whisper import WhisperModel


def hms(t: float) -> str:
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main() -> None:
    audio, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    model = WhisperModel("small", device="cpu", compute_type="int8")
    t0 = time.time()
    segments, info = model.transcribe(
        audio,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    out_segments = []
    for seg in segments:
        out_segments.append(
            {
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text.strip(),
                "words": [
                    {"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word}
                    for w in (seg.words or [])
                ],
            }
        )
        print(f"[{hms(seg.start)}] {seg.text.strip()[:80]}", flush=True)

    (out_dir / "transcript.json").write_text(
        json.dumps(
            {
                "language": info.language,
                "duration": info.duration,
                "segments": out_segments,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (out_dir / "transcript.txt").open("w", encoding="utf-8") as fh:
        for s in out_segments:
            fh.write(f"[{hms(s['start'])} - {hms(s['end'])}] {s['text']}\n")

    print(f"DONE segments={len(out_segments)} elapsed={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
