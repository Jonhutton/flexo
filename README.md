# Flexo — MTG Arena Quick Draft voice co-pilot

Flexo is an insufferably brilliant robot wizard who co-hosts Arena Quick
Drafts on camera: he sees the pick screen, recommends a pick in character,
talks back over push-to-talk, and keeps a disagreement scoreboard against the
human. Claude for the brain, ElevenLabs for the voice, faster-whisper for the
ears, an OBS browser-source overlay for the eyes on stream.

## Setup

```
pip install -r requirements.txt
copy .env.example .env        # then fill in the keys
python flexo_live.py
```

`.env` keys: `ANTHROPIC_API_KEY` (required), `ELEVENLABS_API_KEY` (optional —
falls back to free edge-tts). Everything else lives in `config.yaml`.

OBS: add a Browser Source pointing at `http://127.0.0.1:7337`.

## Hotkeys

| Key | Action |
|---|---|
| **F8** (hold) | Push-to-talk. Going down hard-interrupts Flexo mid-word. |
| **F9** (press) | **Show Flexo the pack** — captures the screen, reads the cards, and he takes his shot. Press once per pick. |

## The vision input layer (v2)

Flexo sees the same pick screen you do: F9 → screenshot of the Arena window →
Claude vision → strict JSON → every card name fuzzy-validated against the
set's Scryfall card list → 17Lands ratings attached → Flexo's take. A name
that can't be matched is flagged and dropped, never guessed — Flexo can no
longer be confidently wrong about a card that isn't there.

- Captures are saved to `captures/<session>/` for debugging and replay.
- If 17Lands has little or no data for the set yet, Flexo is told to lean on
  his own judgment — and to say so in character.
- The old Player.log parser still runs as a cross-check (pick detection +
  draft end). See `MIGRATION.md` for exactly what vision owns vs. the log.
- `vision.trigger_mode: auto` (stretch feature, costs more API calls) polls a
  tiny screen thumbnail and fires automatically when the screen settles on
  something new; the vision call itself ignores non-draft screens.

### Config (`config.yaml`, `vision:` section)

| Key | What it does |
|---|---|
| `enabled` | `false` restores the v1 log-driven input path |
| `hotkey` | Capture key (default `f9`; F8 stays push-to-talk) |
| `trigger_mode` | `hotkey` (default) or `auto` |
| `model` / `effort` / `max_tokens` | The vision model. Default `claude-opus-5` at `effort: low`; for cheaper/faster runs use `claude-haiku-4-5` and set `effort: null` |
| `set_code` | **The one config change per set release** — see below |
| `match_threshold` | Fuzzy-match ratio below which a name is flagged, not corrected |
| `max_image_long_edge` / `jpeg_quality` | Screenshot downscaling before the API |
| `log_cross_check` | Tail Player.log for pick events + draft end |

### New set release checklist

1. Change `vision.set_code` in `config.yaml` (e.g. `"ECL"` → whatever's next).
2. That's it. The Scryfall card list and 17Lands ratings for the new set are
   fetched and cached automatically on first run (`data/<SET>_names.json`,
   `data/<SET>_<format>.json`).

## Dry run (test the pipeline without going live)

Run one saved screenshot end-to-end in the console — vision → validation →
ratings → Flexo's line — with no TTS, overlay, or hotkeys:

```
python flexo_live.py --dry-run tests/fixtures/pick_screen_p1p3.png
```

The bundled fixtures are synthetic pick screens built from real card names
(`python scripts/make_vision_fixture.py` regenerates them). Better: after your
next real draft, grab a PNG from `captures/<session>/` and dry-run that.

## Other modes

```
python flexo_live.py --replay old_player.log   # replay a saved log end to end (v1 path)
python flexo_live.py --no-voice                # no mic / push-to-talk
python flexo_live.py --timings                 # per-stage latency logging
```

## Scoreboard

Every pick is logged to `draft_log.csv` (Flexo's top pick vs. yours). At draft
end the disagreements are read out and a `draft_summary_<id>.json` is written
next to the CSV — shaped for joining against your 17Lands draft log later, so
the scoreboard can be settled by real data on camera. The running Human vs AI
tally lives in `human_vs_ai_score.json`, awarded via `score.py`.

## Tests

Standalone, no API keys or network needed:

```
python tests/test_validation.py
python tests/test_ratings_cache.py
python tests/test_vision_pipeline.py
python tests/test_speech_queue.py
python tests/test_stt_early.py
```
