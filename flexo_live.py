#!/usr/bin/env python3
"""Flexo — live voice co-pilot for MTG Arena Quick Draft.

Single entrypoint. Starts the vision input layer (or the legacy log tail), the
OBS overlay web server, the voice output, and (unless disabled) push-to-talk
voice input, then narrates picks in Flexo's voice as packs arrive.

Input layer (v2): VISION is the source of truth — press F9 on a pick screen
and Flexo sees exactly what you see (screenshot -> Claude vision -> validated
card names). The Player.log parser stays as an optional cross-check for pick
detection and draft-end only. Set vision.enabled: false in config.yaml to run
the old log-driven path.

    python flexo_live.py                     # live: vision hotkey (F9) + log cross-check
    python flexo_live.py --dry-run shot.png  # one saved screenshot through the whole
                                             # pipeline (vision -> validate -> ratings
                                             # -> Flexo text), console only
    python flexo_live.py --replay old.log    # replay a saved Player.log end to end
    python flexo_live.py --replay old.log --replay-delay 6
    python flexo_live.py --no-voice          # skip push-to-talk (no mic / testing)
    python flexo_live.py --timings           # log ms per stage (STT/LLM/TTS/playback)

Audio rules (v1.1): all speech goes through one cancellable SpeechQueue
(flexo/audio.py). F8 down = hard barge-in (Flexo dies mid-word, queue flushed);
pack reactions run in their own thread with a generation guard so a newer pack
always invalidates lines about older ones. See FINDINGS.md.

Config lives in config.yaml; API keys come from the environment.
"""

import argparse
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import yaml

from flexo import cards, log_watch
from flexo.advisor import Advisor
from flexo.audio import SpeechQueue
from flexo.draft_state import DraftState
from flexo.log_watch import CompletedEvent, PackEvent, PickedEvent
from flexo.scoreboard import Scoreboard
from flexo.seventeen_lands import SeventeenLands
from flexo.tts import make_tts
from flexo.web import OverlayServer

ROOT = Path(__file__).resolve().parent

# Card names and Scryfall rules text carry characters the Windows console
# codepage can't encode (planeswalker "−", em dashes, "Æ"). A bare print of one
# raises UnicodeEncodeError, and a crash mid-draft costs a whole recording —
# so degrade those characters instead of dying on them.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# --------------------------------------------------------------------------- app
class Flexo:
    def __init__(self, cfg: dict, no_voice: bool = False, timings: bool = False):
        self.cfg = cfg
        self.timings = timings
        paths = cfg["paths"]

        self.card_map = cards.build_card_map(ROOT / paths["card_map"])
        self.seventeen = SeventeenLands(
            data_dir=ROOT / paths["data_dir"],
            fmt=cfg["seventeen_lands"]["format"],
            cache_hours=cfg["seventeen_lands"]["cache_hours"],
        )
        self.advisor = Advisor(
            model=cfg["claude"]["model"],
            seventeen=self.seventeen,
            max_tokens=cfg["claude"]["max_tokens"],
        )
        self.scoreboard = Scoreboard(
            csv_path=ROOT / paths["draft_log_csv"],
            score_path=ROOT / paths["score_json"],
        )
        self.overlay = OverlayServer(cfg["web"]["host"], cfg["web"]["port"])
        self.speech = SpeechQueue(make_tts(cfg), timings=timings)

        self.state = DraftState()
        self.draft_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.no_voice = no_voice
        self._ptt = None

        # Pack generation: every pack (and every completed pick) bumps it.
        # An advisor call finishing for a stale generation is discarded — Flexo
        # never speaks about pack N-1 once pack N exists.
        self._pack_gen = 0
        self._gen_lock = threading.Lock()
        self._last_pack_utt = None      # Utterance of the latest pack line
        self._interrupted = False       # F8 cut Flexo off mid-line

        # One pick decision is "open" per pack. Both the vision diff and the
        # log cross-check can detect the same pick; whichever reports first
        # records it, the other is a no-op.
        self._pick_open = False

        # Vision input layer (v2 source of truth). Built lazily in run_live so
        # --replay and --dry-run never touch the screen or Scryfall.
        self.vision = None

    # ------------------------------------------------------------------ names
    def _names(self, ids) -> list[str]:
        return [cards.name_of(self.card_map, i) for i in ids]

    # ------------------------------------------------------------------ overlay
    def _push_overlay(self) -> None:
        score = self.scoreboard.load_score()
        reco = self.state.last_reco
        self.overlay.update(
            set=self.state.set_code,
            pack=reco.pack_number if reco else 0,
            pick=reco.pick_number if reco else 0,
            confidence=reco.confidence if reco else "",
            spoken=reco.spoken if reco else "Flexo is watching...",
            plan=self.state.plan_summary(),
            picks=[{"name": p["name"], "gih_wr": p["gih_wr"]} for p in (reco.picks if reco else [])],
            human=score.get("human", 0),
            ai=score.get("ai", 0),
        )

    # ------------------------------------------------------------------ voice input
    def _on_ptt_down(self) -> None:
        """F8 went down: hard barge-in. Kill audio NOW; hold the queue until the
        transcript (or abort) resolves so Flexo can't talk over the drafter."""
        self._interrupted = self.speech.interrupt_and_hold()

    def _on_ptt_abort(self) -> None:
        """Hold produced no usable speech — just let the voice flow again."""
        self._interrupted = False
        self.speech.release_hold()

    def _on_transcript(self, text: str) -> None:
        try:
            print(f"\n[you] {text}", flush=True)
            interrupted, self._interrupted = self._interrupted, False

            # If the latest pack line never aired (barged in / arrived during the
            # hold), don't speak it stale afterwards — fold the pick into the reply.
            unspoken = None
            utt, reco = self._last_pack_utt, self.state.last_reco
            if utt is not None and not utt.playing and reco is not None:
                self.speech.cancel_utterance(utt)
                unspoken = reco.spoken

            t0 = time.perf_counter()
            result = self.advisor.respond(text, self.state,
                                          interrupted=interrupted,
                                          unspoken_reco=unspoken)
            if self.timings:
                print(f"[timings] banter llm: {(time.perf_counter()-t0)*1000:.0f}ms",
                      flush=True)
            self.state.set_plan(result.get("plan_update"), result.get("colors"))
            if result.get("conceded"):
                self.state.conceded = True
            spoken = result.get("spoken", "")
            print(f"[Flexo] {spoken}", flush=True)
            self.speech.say(spoken, kind="banter")
            self._push_overlay()
        finally:
            self.speech.release_hold()

    def _start_voice_input(self) -> None:
        if self.no_voice:
            print("[stt] voice input disabled (--no-voice).", flush=True)
            return
        try:
            from flexo.stt import PushToTalk
            vi = self.cfg["voice_input"]
            self._ptt = PushToTalk(
                on_transcript=self._on_transcript,
                on_ptt_down=self._on_ptt_down,
                on_ptt_abort=self._on_ptt_abort,
                hotkey=vi["hotkey"],
                whisper_model=vi["whisper_model"],
                device=vi["device"],
                sample_rate=vi["sample_rate"],
                early_stt=vi.get("early_stt", True),
                silence_ms=vi.get("silence_ms", 600),
                silence_rms=vi.get("silence_rms", 0.008),
                min_segment_seconds=vi.get("min_segment_seconds", 1.2),
                timings=self.timings,
            )
            self._ptt.start()
        except Exception as e:
            print(f"[stt] voice input unavailable ({e}); continuing without it.",
                  flush=True)

    # ------------------------------------------------------------------ event handling
    def _handle_pack(self, set_code: str, pack_no: int, pick_no: int,
                     pack_names: list[str], ranking=None) -> None:
        """A new pack to advise on — shared by the log parser and vision paths.

        `ranking` is the picker's deterministic order (crops path only). When
        present it rides along to the advisor as evidence, so Flexo narrates a
        ranking instead of building one from card names alone.
        """
        self.state.set_code = set_code
        self.state.current_pack = pack_names
        self._pick_open = True

        with self._gen_lock:
            self._pack_gen += 1
            gen = self._pack_gen
        # A newer pack exists: queued lines about older packs are now stale.
        self.speech.cancel_stale_packs(gen)

        print(f"\n--- Pack {pack_no}, Pick {pick_no} "
              f"({len(pack_names)} cards) ---", flush=True)
        # Kick the Claude call off immediately, off the event loop, so a slow
        # call (or in-progress TTS) never delays seeing the next event.
        threading.Thread(target=self._recommend_async,
                         args=(set_code, pack_no, pick_no, pack_names, gen,
                               ranking),
                         daemon=True).start()

    def _recommend_async(self, set_code: str, pack_no: int, pick_no: int,
                         pack_names: list[str], gen: int, ranking=None) -> None:
        t0 = time.perf_counter()
        try:
            reco = self.advisor.recommend(
                set_code, pack_no, pick_no, pack_names, self.state,
                ranking=ranking,
                cards=getattr(self.vision, "cards", None) if self.vision else None,
            )
        except Exception as e:
            print(f"[advisor] recommend failed: {e}", flush=True)
            return
        llm_ms = (time.perf_counter() - t0) * 1000
        with self._gen_lock:
            if gen != self._pack_gen:
                print(f"[Flexo] (stale advice for pack {pack_no} pick "
                      f"{pick_no} discarded)", flush=True)
                return
            self.state.last_reco = reco
        print(f"[Flexo {reco.confidence}] {reco.spoken}  ({llm_ms/1000:.1f}s)",
              flush=True)
        if self.timings:
            print(f"[timings] pack p{pack_no}p{pick_no} llm: "
                  f"{llm_ms:.0f}ms", flush=True)
        self._last_pack_utt = self.speech.say(reco.spoken, kind="pack", seq=gen)
        self._push_overlay()

    def _record_pick(self, names: list[str], source: str = "log") -> None:
        """The human locked in a pick — from the log parser or the vision diff.
        First reporter wins; the duplicate detection is a no-op."""
        if not names or not self._pick_open:
            return
        self._pick_open = False
        actual = names[0]
        reco = self.state.last_reco
        if reco is not None:
            row = self.state.record_turn(reco, actual)
            self.scoreboard.log_pick(
                self.draft_id,
                datetime.now().isoformat(timespec="seconds"),
                row,
            )
            tag = "agreed" if row["agreed"] else "OVERRULED"
            print(f"[pick/{source}] you took {actual!r} ({tag})", flush=True)
        else:
            print(f"[pick/{source}] you took {actual!r} (no take was recorded)",
                  flush=True)
        self.state.add_picks(names)
        self.state.last_reco = None  # consumed; wait for next pack
        with self._gen_lock:
            self._pack_gen += 1      # advice still in flight for this pack is moot
        self._push_overlay()

    def _on_pack(self, ev: PackEvent) -> None:
        self._handle_pack(ev.set_code, ev.pack_number, ev.pick_number,
                          self._names(ev.card_ids))

    def _on_picked(self, ev: PickedEvent) -> None:
        self._record_pick(self._names(ev.card_ids), source="log")

    def _on_completed(self) -> None:
        print("\n=== Draft complete. ===", flush=True)
        self.speech.say("Draft complete. Let's review where you went wrong.")
        self.scoreboard.print_disagreements(self.draft_id)
        path = self.scoreboard.write_summary(self.draft_id)
        if path:
            print(f"[scoreboard] draft summary written to {path}", flush=True)

    # ------------------------------------------------------------------ vision path
    def _on_vision_pack(self, vp) -> None:
        """VisionPack from the pipeline — vision is the source of truth."""
        # The drafted-cards sidebar is the full pool; trust it when it knows
        # more than we do (e.g. Flexo started mid-draft).
        if len(vp.picked_so_far) > len(self.state.pool):
            self.state.pool = list(vp.picked_so_far)
        # Keep the pipeline's pool view in step so the next pack's color
        # weighting reflects what has actually been taken.
        if self.vision is not None:
            self.vision.set_pool(self.state.pool)
        self._handle_pack(vp.set_code, vp.pack_number, vp.pick_number,
                          vp.card_names, ranking=getattr(vp, "ranking", None))

    def _on_vision_pick(self, name: str) -> None:
        self._record_pick([name], source="vision")

    def _build_vision(self):
        """Construct the vision pipeline from config. Raises with a clear
        message if the set list can't be fetched — vision is the input layer,
        so starting blind would be worse than failing loudly."""
        from flexo.capture import ScreenCapture
        from flexo.picker import PickerConfig
        from flexo.set_cards import CardValidator, supplement_from_names
        from flexo.vision import CardIdentifier, VisionExtractor
        from flexo.vision_pipeline import VisionPipeline

        vcfg = self.cfg["vision"]
        set_code = vcfg["set_code"]
        if not set_code:
            raise SystemExit("[vision] set vision.set_code in config.yaml "
                             "(e.g. the current Quick Draft set).")
        data_dir = ROOT / self.cfg["paths"]["data_dir"]
        capture = ScreenCapture(ROOT / vcfg.get("captures_dir", "captures"),
                                max_long_edge=vcfg.get("max_image_long_edge", 2048),
                                jpeg_quality=vcfg.get("jpeg_quality", 85))
        extractor = VisionExtractor(model=vcfg["model"],
                                    max_tokens=vcfg.get("max_tokens", 4000),
                                    effort=vcfg.get("effort"))

        identifier = cards = ratings = picker_cfg = None
        if (vcfg.get("read_mode") or "screen") == "crops":
            # Ratings first: the 17Lands name list is what tells us which cards
            # are actually draftable, including bonus-sheet cards that live
            # under a different Scryfall set code.
            self.seventeen.ensure_set(set_code)
            ratings = self.seventeen._by_name
            cards = supplement_from_names(set_code, data_dir, list(ratings))
            picker_cfg = PickerConfig.from_cfg(self.cfg)

        # Built after the supplement so the candidate list includes bonus-sheet
        # cards — a name the validator has never heard of can never be matched.
        validator = CardValidator(
            set_code, data_dir,
            threshold=vcfg.get("match_threshold", 0.82),
            recheck_threshold=vcfg.get("recheck_threshold", 0.60))

        if cards is not None:
            identifier = CardIdentifier(
                validator,
                model=vcfg["model"],
                max_tokens=vcfg.get("max_tokens", 4000),
                effort=vcfg.get("effort"),
                upscale_min_width=vcfg.get("upscale_min_width", 420))
            print(f"[vision] crops mode: {len(cards)} cards, "
                  f"{len(validator.names)} candidate names, "
                  f"17Lands confidence '{self.seventeen.data_confidence()}'.",
                  flush=True)

        return VisionPipeline(
            set_code=set_code,
            capture=capture,
            extractor=extractor,
            validator=validator,
            on_pack=self._on_vision_pack,
            on_my_pick=self._on_vision_pick,
            hotkey=vcfg.get("hotkey", "f9"),
            trigger_mode=vcfg.get("trigger_mode", "hotkey"),
            auto_poll_seconds=vcfg.get("auto_poll_seconds", 2.0),
            auto_cooldown_seconds=vcfg.get("auto_cooldown_seconds", 8.0),
            timings=self.timings,
            identifier=identifier,
            vision_cfg=vcfg,
            cards=cards,
            ratings=ratings,
            picker_cfg=picker_cfg,
        )

    def _tail_log_crosscheck(self, log_path: Path) -> None:
        """Optional background tail of Player.log in vision mode: pick events
        and draft-end only. Pack contents come from vision."""
        try:
            while not log_path.exists():
                time.sleep(2.0)
            print("[log] cross-check tail running (picks + draft end).", flush=True)
            for ev in log_watch.follow_live(log_path, self.cfg["log"]["poll_seconds"]):
                if isinstance(ev, PickedEvent):
                    self._record_pick(self._names(ev.card_ids), source="log")
                elif isinstance(ev, CompletedEvent):
                    self._on_completed()
        except Exception as e:
            print(f"[log] cross-check tail stopped ({e}); vision continues alone.",
                  flush=True)

    def _dispatch(self, ev) -> None:
        if isinstance(ev, PackEvent):
            self._on_pack(ev)
        elif isinstance(ev, PickedEvent):
            self._on_picked(ev)
        elif isinstance(ev, CompletedEvent):
            self._on_completed()

    # ------------------------------------------------------------------ run
    def run_live(self, log_path: Path) -> None:
        self.overlay.start()
        self._start_voice_input()

        vcfg = self.cfg.get("vision") or {}
        if vcfg.get("enabled", False):
            # v2: vision is the source of truth; log tail is a cross-check.
            self.vision = self._build_vision()
            self.vision.start()
            if vcfg.get("log_cross_check", True):
                threading.Thread(target=self._tail_log_crosscheck,
                                 args=(log_path,), daemon=True).start()
            self.speech.say("Flexo online. Show me the pack.")
            while True:                      # hotkey/auto threads drive everything
                time.sleep(1.0)
            return

        # Legacy v1 path: the log parser drives everything.
        if not log_path.exists():
            print(f"[log] waiting for Arena log at {log_path} ...", flush=True)
            while not log_path.exists():
                time.sleep(1.0)
        print("[log] Arena log is live.", flush=True)
        self.speech.say("Flexo online. Show me the pack.")

        for ev in log_watch.follow_live(log_path, self.cfg["log"]["poll_seconds"]):
            self._dispatch(ev)

    def run_replay(self, log_path: Path, delay: float) -> None:
        self.overlay.start()
        self._start_voice_input()
        print(f"[replay] {log_path}  (delay {delay}s between packs)", flush=True)
        self.speech.say("Flexo online. Replaying an old draft.")
        for ev in log_watch.follow_replay(log_path, delay):
            self._dispatch(ev)
        self._on_completed()
        self.speech.wait_idle(timeout=30.0)   # let the last line finish speaking


# --------------------------------------------------------------------------- dry run
def run_dry_run_crops(cfg: dict, image_path: str) -> None:
    """One saved screenshot through the per-card pipeline, console only.

    detect rectangles -> read each name -> validate -> rank -> Flexo's take.
    This is the path a live pack takes, so a screenshot that works here works
    on camera. Prints per-slot evidence, because when a read goes wrong the
    question is always *which* slot and *what did it see*.
    """
    from PIL import Image

    from flexo.detect import (crop_cards, crop_fraction, draw_boxes,
                              find_cards, name_banner)
    from flexo.picker import PickerConfig, rank_pack
    from flexo.set_cards import CardValidator, supplement_from_names
    from flexo.vision import CardIdentifier

    vcfg = cfg["vision"]
    set_code = vcfg["set_code"]
    if not set_code:
        raise SystemExit("[dry-run] set vision.set_code in config.yaml first.")
    data_dir = ROOT / cfg["paths"]["data_dir"]

    print(f"=== DRY RUN (crops): {image_path} (set {set_code}) ===\n", flush=True)

    # 1. Data backbone
    seventeen = SeventeenLands(data_dir=data_dir,
                               fmt=cfg["seventeen_lands"]["format"],
                               cache_hours=cfg["seventeen_lands"]["cache_hours"],
                               csv_path=cfg["seventeen_lands"].get("csv_path"))
    seventeen.ensure_set(set_code)
    ratings = seventeen._by_name
    card_pool = supplement_from_names(set_code, data_dir, list(ratings))
    print(f"[1. data]    {len(card_pool)} cards, "
          f"17Lands confidence '{seventeen.data_confidence()}'", flush=True)

    # 2. Detection
    img = Image.open(image_path).convert("RGB")
    frame = crop_fraction(img, vcfg.get("pack_region"))
    boxes, mode = find_cards(frame, vcfg)
    print(f"[2. detect]  {len(boxes)} card rect(s) via {mode} "
          f"({frame.width}x{frame.height})", flush=True)
    if not boxes:
        raise SystemExit("[dry-run] no cards detected. Run "
                         "scripts/calibrate.py on this image to see why.")

    # 3. Identification
    validator = CardValidator(set_code, data_dir,
                              threshold=vcfg.get("match_threshold", 0.82),
                              recheck_threshold=vcfg.get("recheck_threshold", 0.60))
    identifier = CardIdentifier(validator, model=vcfg["model"],
                                max_tokens=vcfg.get("max_tokens", 4000),
                                effort=vcfg.get("effort"),
                                upscale_min_width=vcfg.get("upscale_min_width", 420))
    crops = crop_cards(frame, boxes)
    ids = identifier.identify(crops, banner_fn=name_banner)

    print("\n[3. identify]", flush=True)
    names = []
    for x in ids:
        mark = {"ok": "  ", "recheck": "~ ", "unknown": "! "}.get(x.status, "? ")
        print(f"  {mark}slot {x.slot:>2}  read={x.raw_name!r:<34} "
              f"-> {x.name or '(unmatched)':<34} "
              f"match={x.score:.2f} conf={x.confidence:.2f}", flush=True)
        if x.name and x.name in card_pool:
            names.append(x.name)
    unread = len(ids) - len(names)
    print(f"  {len(names)}/{len(ids)} slots identified"
          + (f"  ({unread} unread)" if unread else ""), flush=True)
    if not names:
        raise SystemExit("[dry-run] nothing identified; check debug/crops/.")

    # 4. Ranking
    ranking = rank_pack([card_pool[n] for n in names], [], ratings,
                        PickerConfig.from_cfg(cfg))
    print("\n[4. rank] (empty pool — pick 1 conditions)", flush=True)
    for i, s in enumerate(ranking[:5], 1):
        print(f"  {i}. {s.name:<34} {s.score:.4f}  {'; '.join(s.reasons)}",
              flush=True)

    # 5. Debug image
    debug_dir = ROOT / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    out = debug_dir / "dryrun_pack.png"
    draw_boxes(frame, boxes, [x.name or "?" for x in ids]).save(out)
    print(f"\nwrote {out} (names drawn on the boxes)", flush=True)

    # 6. Flexo's take
    state = DraftState(set_code=set_code)
    state.current_pack = [s.name for s in ranking]
    advisor = Advisor(model=cfg["claude"]["model"], seventeen=seventeen,
                      max_tokens=cfg["claude"]["max_tokens"])
    reco = advisor.recommend(set_code, 1, 1, [s.name for s in ranking], state,
                             ranking=ranking, cards=card_pool)
    print(f"\n[5. Flexo {reco.confidence}] {reco.spoken}", flush=True)
    for p in reco.picks:
        wr = p.get("gih_wr")
        print(f"     - {p['name']} ({'--' if wr is None else f'{wr*100:.1f}%'}): "
              f"{p['reason']}", flush=True)


def run_dry_run(cfg: dict, image_path: str) -> None:
    """One saved screenshot through the whole vision pipeline, console only:
    vision -> JSON -> validation -> ratings -> Flexo's take. No TTS, no overlay,
    no hotkeys. This is the test path for a new set, a new model, or a new
    screenshot fixture before going live on camera."""
    from flexo.capture import load_image_for_vision
    from flexo.set_cards import CardValidator
    from flexo.vision import VisionExtractor

    vcfg = cfg["vision"]
    # vision.read_mode picks the path, so --dry-run always exercises whichever
    # one the live run would use.
    if (vcfg.get("read_mode") or "screen") == "crops":
        return run_dry_run_crops(cfg, image_path)

    set_code = vcfg["set_code"]
    if not set_code:
        raise SystemExit("[dry-run] set vision.set_code in config.yaml first.")

    print(f"=== DRY RUN: {image_path} (set {set_code}) ===\n", flush=True)

    # 1. Vision extraction
    jpeg, media_type = load_image_for_vision(
        Path(image_path),
        max_long_edge=vcfg.get("max_image_long_edge", 2048),
        jpeg_quality=vcfg.get("jpeg_quality", 85),
    )
    extractor = VisionExtractor(model=vcfg["model"],
                                max_tokens=vcfg.get("max_tokens", 4000),
                                effort=vcfg.get("effort"))
    result = extractor.extract(jpeg, media_type)
    if result is None:
        raise SystemExit("[dry-run] vision extraction failed; see log above.")
    print(f"[1. vision]  screen_type={result.screen_type}  "
          f"pack {result.pack_number}, pick {result.pick_number}", flush=True)
    print(f"             cards_in_pack: {result.cards_in_pack}", flush=True)
    print(f"             picked_so_far: {result.cards_picked_so_far}", flush=True)
    if result.confidence_notes:
        print(f"             notes: {result.confidence_notes}", flush=True)
    if not result.is_pick_screen:
        print("\nNot a draft pick screen — Flexo would stay quiet. Done.", flush=True)
        return

    # 2. Validation
    validator = CardValidator(set_code, ROOT / cfg["paths"]["data_dir"],
                              threshold=vcfg.get("match_threshold", 0.82))
    validated = validator.validate_all(result.cards_in_pack)
    names = [v.name for v in validated if v.ok]
    print("\n[2. validate]", flush=True)
    for v in validated:
        if not v.ok:
            print(f"    FLAGGED  {v.raw!r} (best score {v.score:.2f})", flush=True)
        elif v.corrected:
            print(f"    fixed    {v.raw!r} -> {v.name!r} ({v.score:.2f})", flush=True)
        else:
            print(f"    ok       {v.name}", flush=True)
    if not names:
        raise SystemExit("[dry-run] no card names survived validation.")

    # 3. Ratings
    from flexo.seventeen_lands import SeventeenLands, stat_line
    seventeen = SeventeenLands(data_dir=ROOT / cfg["paths"]["data_dir"],
                               fmt=cfg["seventeen_lands"]["format"],
                               cache_hours=cfg["seventeen_lands"]["cache_hours"])
    seventeen.ensure_set(set_code)
    print(f"\n[3. ratings]  17Lands data confidence: "
          f"{seventeen.data_confidence()}", flush=True)
    for n in names:
        print(f"    {stat_line(n, seventeen.stats_for(n))}", flush=True)

    # 4. Flexo's take
    advisor = Advisor(model=cfg["claude"]["model"], seventeen=seventeen,
                      max_tokens=cfg["claude"]["max_tokens"])
    reco = advisor.recommend(set_code, result.pack_number, result.pick_number,
                             names, DraftState())
    print(f"\n[4. Flexo {reco.confidence}]  {reco.spoken}", flush=True)
    for p in reco.picks:
        print(f"    pick: {p['name']}  ({p['reason']})", flush=True)
    print("\n=== dry run complete ===", flush=True)


# --------------------------------------------------------------------------- entry
def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no dependency). Sets any KEY=value not already in the
    environment, so API keys live in the git-ignored .env instead of config/chat."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def _resolve_log_path(cfg: dict) -> Path:
    configured = cfg["log"].get("path")
    if configured:
        return Path(configured)
    env = os.environ.get("MTGA_LOG")
    if env:
        return Path(env)
    return log_watch.DEFAULT_LOG


def main() -> None:
    # Model output (Flexo's lines) can contain em-dashes / smart quotes. Never let
    # printing them crash on a strict Windows console codepage.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    _load_dotenv(ROOT / ".env")   # pick up API keys from the git-ignored .env

    ap = argparse.ArgumentParser(description="Flexo - MTGA Quick Draft voice co-pilot")
    ap.add_argument("--replay", metavar="LOG", help="replay a saved Player.log instead of watching live")
    ap.add_argument("--replay-delay", type=float, default=6.0,
                    help="seconds to pause between packs in replay (default 6)")
    ap.add_argument("--dry-run", metavar="IMAGE",
                    help="run one saved screenshot through the vision pipeline "
                         "(vision -> validate -> ratings -> Flexo text), console only")
    ap.add_argument("--no-voice", action="store_true", help="disable push-to-talk voice input")
    ap.add_argument("--timings", action="store_true",
                    help="log ms per stage (STT, LLM, TTS first audio, playback start)")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

    if args.dry_run:
        run_dry_run(cfg, args.dry_run)
        return

    app = Flexo(cfg, no_voice=args.no_voice, timings=args.timings)

    try:
        if args.replay:
            app.run_replay(Path(args.replay), args.replay_delay)
        else:
            app.run_live(_resolve_log_path(cfg))
    except KeyboardInterrupt:
        print("\n[flexo] shutting down.", flush=True)
        if app._ptt:
            app._ptt.stop()
        if app.vision:
            app.vision.stop()


if __name__ == "__main__":
    main()
