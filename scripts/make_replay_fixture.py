#!/usr/bin/env python3
"""Reconstruct a raw Player.log fixture from a MTGA_Draft_17Lands saved draft log.

That tool's saved DraftLog captures the real Arena payload lines but strips the
`<==` marker lines the parser keys on. This re-inserts them so the fixture drives
flexo/log_watch.py exactly like a genuine Player.log — with real card IDs, so
`--replay` produces real card names.

    python scripts/make_replay_fixture.py <source_draftlog> <out.log>
"""

import sys
from pathlib import Path

PAYLOAD_MARK = '{"CurrentModule":"BotDraft","Payload":"'


def build(src: Path, out: Path) -> None:
    lines_out: list[str] = []
    first_payload = True
    picks = packs = 0

    for raw in src.read_text(encoding="utf-8", errors="replace").splitlines():
        if "==> BotDraftDraftPick" in raw:
            # Pass the pick request through unchanged; the parser reads the inline JSON.
            lines_out.append(raw)
            picks += 1
            continue

        if PAYLOAD_MARK in raw:
            # Emit the marker line the parser expects, then the bare JSON payload.
            marker = "BotDraftDraftStatus" if first_payload else "BotDraftDraftPick"
            first_payload = False
            lines_out.append(f"[UnityCrossThreadLogger]<== {marker}")
            lines_out.append(raw[raw.index("{"):])   # strip "<timestamp>," prefix
            packs += 1

    out.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    print(f"Wrote {out}  ({packs} payloads, {picks} pick requests)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    build(Path(sys.argv[1]), Path(sys.argv[2]))
