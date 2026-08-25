# Migration: log-parser input → vision input (v2)

Vision is now the **source of truth** for draft state. The Player.log parser
still runs, but only as a cross-check. This file is the map of what moved.

## What vision now owns

| Responsibility | v1 (log parser) | v2 (vision) |
|---|---|---|
| Pack contents | `<== BotDraftDraftPick/Status` payload, Arena card IDs → `card_map.json` | Screenshot → Claude vision → card names → fuzzy-validated against the set's Scryfall list |
| Pack/pick numbers | Log payload | Read off the "Pack X, Pick Y" indicator on screen |
| Set code | Auto-detected from the log `EventName` | `vision.set_code` in config.yaml (**the** one config change per set release) |
| Trigger | New log lines appear | F9 hotkey (default) or auto-detect polling (`vision.trigger_mode: auto`) |
| Pool (cards picked so far) | Accumulated from `PickedEvent`s | Accumulated from picks, plus synced from the drafted-cards sidebar when it knows more |

## What the log parser still does (cross-check, `vision.log_cross_check: true`)

- **Pick detection** — the `==> BotDraftDraftPick` request fires the instant you
  click, so the log usually reports your pick before the next F9 press does.
  Whichever source reports first is recorded; the other is a no-op
  (`_pick_open` flag in `flexo_live.py`).
- **Draft-end detection** — `DraftStatus: Completed` triggers the wrap-up line,
  the disagreement review, and the draft summary JSON.
- Pack contents from the log are **ignored** in vision mode.

Vision's own pick detection (used when the log is unavailable): the
drafted-cards sidebar is diffed between consecutive captures — one newly
appeared card = your pick. Two or more new cards can't be attributed to a
single pick and are skipped with a console note.

## Why

The log-based input layer was the #1 source of Flexo confusion:
card-ID→name mappings went stale every set release, and log event formats
broke whenever WotC updated the client. Vision sees exactly what you see, and
the validation layer (`flexo/set_cards.py`) guarantees a misread name is
corrected or flagged — never confidently wrong on camera.

## Rollback

`vision.enabled: false` in config.yaml restores the v1 log-driven path
unchanged (including `--replay`, which never used vision).

## New modules

- `flexo/capture.py` — Arena-window screen grab (mss + Win32), per-session
  capture folders, downscaled JPEG encoding for the API
- `flexo/vision.py` — Claude vision extraction, strict JSON via structured
  outputs, one retry, fail-soft
- `flexo/set_cards.py` — Scryfall per-set card list cache + fuzzy validator
- `flexo/vision_pipeline.py` — hotkey/auto triggers, dedupe, sidebar pick diff
- `flexo/seventeen_lands.py` — extended with OH WR, ALSA in prompts, sample
  sizes, and a data-confidence signal ("early in the set" → Flexo hedges in
  character instead of inventing stats)
