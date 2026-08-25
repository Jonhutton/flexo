"""Arena Player.log parsing for Quick Draft (bot draft) events.

Ported from the original mtga_draft_assistant.py. The parsing rules are
UNCHANGED — they were built against the exact log schema observed in a real
SOS Quick Draft:

  "<== BotDraftDraftStatus"  -> next line is JSON carrying the FIRST pack
  "<== BotDraftDraftPick"    -> next line is JSON carrying the NEXT pack
  "==> BotDraftDraftPick"    -> inline JSON, contains the CardIds you just picked

The pack lives in a DOUBLE-ENCODED payload:
  outer  {"CurrentModule":...,"Payload":"{...}"}
  inner  {"DraftStatus":"PickNext","PackNumber":n,"PickNumber":m,"DraftPack":[...]}
DraftStatus "Completed" + empty DraftPack ends the draft.

This module wraps that logic in an incremental parser (LogParser.feed) plus two
drivers — follow_live() for a real draft and follow_replay() for testing against
a saved log — so the rest of Flexo consumes one uniform event stream.
"""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# Default Arena Player.log location on Windows. Override via config or MTGA_LOG.
DEFAULT_LOG = Path(
    os.path.expandvars(r"%APPDATA%\..\LocalLow\Wizards Of The Coast\MTGA\Player.log")
)


# --------------------------------------------------------------------------- events
@dataclass
class PickedEvent:
    """Cards the human just locked in (parsed from a '==>' pick request)."""
    card_ids: list[str]


@dataclass
class PackEvent:
    """A new pack Flexo must advise on."""
    set_code: str
    pack_number: int          # 1-indexed
    pick_number: int          # 1-indexed
    card_ids: list[str]
    num_cards_to_pick: int = 1  # 1 for Quick Draft; would be 2 for Pick-Two draft


@dataclass
class CompletedEvent:
    """The draft finished."""
    pass


# --------------------------------------------------------------------------- parsing
def extract_payload(json_body: str):
    """Parse the outer {CurrentModule, Payload} line, then the inner Payload JSON."""
    try:
        outer = json.loads(json_body)
        return json.loads(outer["Payload"])
    except Exception:
        return None


def picked_ids_from_request(line: str):
    """Pull CardIds out of an inline '==> BotDraftDraftPick {...}' request line."""
    try:
        start = line.index("{")
        outer = json.loads(line[start:])
        req = json.loads(outer["request"])
        return req.get("PickInfo", {}).get("CardIds", [])
    except Exception:
        return []


def _set_code(payload: dict) -> str:
    event = payload.get("EventName", "")
    return event.split("_")[1] if "_" in event else "Unknown"


@dataclass
class LogParser:
    """Incremental line parser. Feed it lines; it yields events.

    Holds only the tiny 'pending' state-machine bit from the original tail loop:
    a '<==' marker line means the NEXT '{' line carries a pack payload.
    `_seen` de-dupes every decision by (pack, pick, exact cards) so an Arena
    status re-send or reconnect can't make Flexo re-advise a pick.
    """
    _pending: bool = field(default=False, init=False)
    _seen: set = field(default_factory=set, init=False)

    def feed(self, raw: str):
        """Process one raw log line; return an event or None."""
        line = raw.rstrip("\r\n")

        if "==> BotDraftDraftPick" in line:
            ids = picked_ids_from_request(line)
            return PickedEvent(ids) if ids else None

        if "<== BotDraftDraftPick" in line or "<== BotDraftDraftStatus" in line:
            self._pending = True
            return None

        if self._pending and line.lstrip().startswith("{"):
            self._pending = False
            payload = extract_payload(line)
            if not payload:
                return None
            if payload.get("DraftStatus") == "Completed":
                self._seen.clear()   # next draft in this session starts fresh
                return CompletedEvent()
            pack_ids = payload.get("DraftPack", [])
            if not pack_ids:
                return None
            p_no = payload.get("PackNumber", 0) + 1
            k_no = payload.get("PickNumber", 0) + 1
            key = (p_no, k_no, tuple(pack_ids))
            if key in self._seen:
                return None  # already advised this exact decision (re-send / reconnect)
            self._seen.add(key)
            return PackEvent(
                set_code=_set_code(payload),
                pack_number=p_no,
                pick_number=k_no,
                card_ids=pack_ids,
                num_cards_to_pick=payload.get("NumCardsToPick", 1),
            )
        return None


# --------------------------------------------------------------------------- drivers
def follow_live(path: Path, poll_seconds: float = 0.4) -> Iterator[object]:
    """Tail a live Player.log. Catches up on existing content, then follows appends.

    Yields PickedEvent / PackEvent / CompletedEvent. Blocks between reads.
    """
    parser = LogParser()
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f.readlines():          # catch up; leaves position at EOF
            ev = parser.feed(line)
            if ev is not None:
                yield ev
        while True:                          # live tail
            line = f.readline()
            if not line:
                time.sleep(poll_seconds)
                continue
            ev = parser.feed(line)
            if ev is not None:
                yield ev


def follow_replay(path: Path, delay: float = 0.0) -> Iterator[object]:
    """Replay a saved Player.log (or reconstructed fixture) end to end.

    `delay` inserts a pause after each emitted pack so you can watch/hear the
    full voice loop off-camera. PickedEvents are not delayed.
    """
    parser = LogParser()
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            ev = parser.feed(line)
            if ev is None:
                continue
            yield ev
            if delay and isinstance(ev, PackEvent):
                time.sleep(delay)
