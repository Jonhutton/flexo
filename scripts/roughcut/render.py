"""Stage 4: assemble rough_cut.mp4 from the kept segments.

Usage: python render.py <ffmpeg_bin_dir> <source_video> <segments.json> <out_dir>

Hard cuts, no transitions. Cut points rarely land on keyframes (OBS keyframe
interval is seconds apart), so kept segments are re-encoded at near-lossless
settings (libx264 CRF 16) for frame-accurate cuts, then concatenated. The
source file is only ever read.
"""
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ffbin, source, segments_path, out_dir = sys.argv[1:5]
    ffmpeg = str(Path(ffbin) / "ffmpeg.exe")
    out = Path(out_dir).resolve()
    parts_dir = out / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    kept = json.loads(Path(segments_path).read_text(encoding="utf-8"))["keep_runs"]
    part_files = []
    for i, seg in enumerate(kept):
        dst = parts_dir / f"part_{i:03d}.mp4"
        part_files.append(dst)
        if dst.exists():
            continue  # resumable
        cmd = [
            ffmpeg, "-y", "-v", "error",
            "-ss", f"{seg['start']:.3f}", "-to", f"{seg['end']:.3f}",
            "-i", source,
            "-c:v", "h264_amf", "-rc", "cqp", "-qp_i", "16", "-qp_p", "16",
            "-quality", "quality", "-pix_fmt", "yuv420p", "-r", "60",
            "-c:a", "aac", "-b:a", "192k",
            "-video_track_timescale", "60000",
            str(dst),
        ]
        subprocess.run(cmd, check=True)
        print(f"part {i + 1}/{len(kept)}: {seg['start']:.1f}-{seg['end']:.1f}s", flush=True)

    concat_list = out / "parts" / "concat.txt"
    concat_list.write_text(
        "".join(f"file '{p.name}'\n" for p in part_files), encoding="utf-8"
    )
    subprocess.run(
        [
            ffmpeg, "-y", "-v", "error",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            str(out / "rough_cut.mp4"),
        ],
        check=True,
        cwd=parts_dir,
    )
    print("rough_cut.mp4 written")


if __name__ == "__main__":
    main()
