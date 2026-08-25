"""Vision input pipeline — capture -> extract -> validate -> events.

This replaces the log parser as Flexo's source of truth for draft state.
Two trigger modes:

  hotkey (default)  F9 captures and processes the current screen. One vision
                    call per press; ~45 per draft in normal use.
  auto              lightweight polling: a tiny grayscale thumbnail is diffed
                    each interval, and only a *stable, changed* screen triggers
                    a real capture + vision call. Behind a config flag; costs
                    more API calls than hotkey mode.

Pick detection: the drafted-cards sidebar is diffed between consecutive
captures — a card that newly appears in cards_picked_so_far is the human's
pick. The log parser (when running) remains a cross-check and usually wins
the race; whichever source reports first is recorded, the other is a no-op.

Fail-soft everywhere: a failed capture, vision call, or validation produces a
log line and a skipped beat, never a crash mid-recording.
"""

import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from pynput import keyboard

from .capture import Capture, ScreenCapture, thumb_diff
from .set_cards import CardValidator
from .vision import VisionExtractor, VisionResult


@dataclass
class VisionPack:
    """A validated pick screen, ready for the advisor (vision's PackEvent)."""
    set_code: str
    pack_number: int
    pick_number: int
    card_names: list[str]                       # validated canonical names
    flagged: list[str] = field(default_factory=list)   # unmatched raw reads
    picked_so_far: list[str] = field(default_factory=list)
    notes: str = ""


class VisionPipeline:
    def __init__(
        self,
        set_code: str,
        capture: ScreenCapture,
        extractor: VisionExtractor,
        validator: CardValidator,
        on_pack: Callable[[VisionPack], None],
        on_my_pick: Callable[[str], None],
        hotkey: str = "f9",
        trigger_mode: str = "hotkey",           # hotkey | auto
        auto_poll_seconds: float = 2.0,
        auto_cooldown_seconds: float = 8.0,
        auto_change_threshold: float = 0.04,
        timings: bool = False,
    ):
        self.set_code = set_code
        self.capture = capture
        self.extractor = extractor
        self.validator = validator
        self.on_pack = on_pack
        self.on_my_pick = on_my_pick
        self.hotkey_name = hotkey
        self.trigger_mode = trigger_mode
        self.auto_poll_seconds = auto_poll_seconds
        self.auto_cooldown_seconds = auto_cooldown_seconds
        self.auto_change_threshold = auto_change_threshold
        self.timings = timings

        self._seen: set = set()                 # (pack, pick, cards) dedupe
        self._last_picked: list[str] = []       # validated sidebar from last capture
        self._busy = threading.Lock()           # one vision call at a time
        self._listener: Optional[keyboard.Listener] = None
        self._auto_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self.trigger_mode == "auto":
            self._auto_thread = threading.Thread(target=self._auto_loop, daemon=True)
            self._auto_thread.start()
            print(f"[vision] auto-detect mode: polling every "
                  f"{self.auto_poll_seconds}s (F-key trigger also active).", flush=True)
        self._start_hotkey()
        print(f"[vision] press {self.hotkey_name.upper()} on a pick screen to "
              f"show Flexo the pack.", flush=True)

    def stop(self) -> None:
        self._stop.set()
        if self._listener:
            self._listener.stop()

    # ------------------------------------------------------------------ triggers
    def _start_hotkey(self) -> None:
        key = self._parse_hotkey(self.hotkey_name)

        def on_press(k):
            if k == key:
                threading.Thread(target=self.trigger, args=("hotkey",),
                                 daemon=True).start()

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.daemon = True
        self._listener.start()

    @staticmethod
    def _parse_hotkey(name: str):
        name = (name or "f9").strip().lower()
        special = getattr(keyboard.Key, name, None)
        return special if special is not None else keyboard.KeyCode.from_char(name[0])

    def _auto_loop(self) -> None:
        """Poll a cheap thumbnail; fire a real capture when the screen settles
        on something new. The vision call itself gates on screen_type, so a
        settled non-draft screen costs one call, then goes quiet."""
        last_seen: Optional[bytes] = None
        last_processed: Optional[bytes] = None
        stable = 0
        last_fire = 0.0
        while not self._stop.wait(self.auto_poll_seconds):
            thumb = self.capture.grab_gray_thumb()
            if thumb is None:
                continue
            if thumb_diff(thumb, last_seen) > self.auto_change_threshold:
                stable = 0            # still animating / mid-transition
            else:
                stable += 1
            last_seen = thumb
            if (stable >= 2
                    and thumb_diff(thumb, last_processed) > self.auto_change_threshold
                    and time.time() - last_fire >= self.auto_cooldown_seconds):
                last_fire = time.time()
                last_processed = thumb
                self.trigger("auto")

    # ------------------------------------------------------------------ pipeline
    def trigger(self, source: str = "hotkey") -> Optional[VisionPack]:
        """Capture the screen and run the full pipeline. Thread-safe; a trigger
        while one is in flight is dropped (the newer screen wins next press)."""
        if not self._busy.acquire(blocking=False):
            print("[vision] already processing a capture; ignored.", flush=True)
            return None
        try:
            t0 = time.perf_counter()
            cap = self.capture.capture(tag=source)
            if cap is None:
                return None
            result = self.extractor.extract(cap.jpeg_bytes, cap.media_type)
            if result is None:
                return None
            if self.timings:
                print(f"[timings] vision extract: "
                      f"{(time.perf_counter() - t0) * 1000:.0f}ms", flush=True)
            return self._process(result, cap)
        finally:
            self._busy.release()

    def process_result(self, result: VisionResult) -> Optional[VisionPack]:
        """Validation + event flow for an already-extracted result (dry-run/tests)."""
        return self._process(result, None)

    def _process(self, result: VisionResult, cap: Optional[Capture]) -> Optional[VisionPack]:
        if not result.is_pick_screen:
            print(f"[vision] not a pick screen ({result.screen_type}); no take.",
                  flush=True)
            return None

        pack_cards, flagged = self._validate(result.cards_in_pack)
        picked_cards, _ = self._validate(result.cards_picked_so_far)

        if not pack_cards:
            print("[vision] no card names survived validation; skipping.", flush=True)
            return None
        if flagged:
            print(f"[vision] unreadable/unmatched (ignored): {flagged}", flush=True)
        if result.confidence_notes:
            print(f"[vision] model notes: {result.confidence_notes}", flush=True)

        # My pick = card newly present in the drafted sidebar vs the last capture.
        # Multiset diff so duplicate commons are handled.
        if self._last_picked and picked_cards:
            new = list((Counter(picked_cards) - Counter(self._last_picked)).elements())
            if len(new) == 1:
                self.on_my_pick(new[0])
            elif len(new) > 1:
                print(f"[vision] {len(new)} new picks since last capture; "
                      f"can't attribute to one pick — skipping scoreboard.", flush=True)
        if picked_cards:
            self._last_picked = picked_cards

        key = (result.pack_number, result.pick_number, tuple(sorted(pack_cards)))
        if key in self._seen:
            print("[vision] same pack/pick already processed; no repeat take.",
                  flush=True)
            return None
        self._seen.add(key)

        vp = VisionPack(
            set_code=self.set_code,
            pack_number=result.pack_number,
            pick_number=result.pick_number,
            card_names=pack_cards,
            flagged=flagged,
            picked_so_far=picked_cards,
            notes=result.confidence_notes,
        )
        self.on_pack(vp)
        return vp

    def _validate(self, raws: list[str]) -> tuple[list[str], list[str]]:
        """Split vision reads into (canonical names, flagged raw strings)."""
        good, bad = [], []
        for v in self.validator.validate_all(raws):
            if v.ok:
                if v.corrected and v.score < 1.0:
                    print(f"[vision] corrected {v.raw!r} -> {v.name!r} "
                          f"({v.score:.2f})", flush=True)
                good.append(v.name)
            else:
                bad.append(v.raw)
        return good, bad
