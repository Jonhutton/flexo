# FINDINGS — Flexo Voice Agent v1.1

Institutional memory for this repo. Architecture, why things are the way they
are, how modules connect, and known issues. Read this before changing anything.

## What Flexo is

A live **voice co-pilot** for recording MTG Arena **Quick Draft** YouTube videos.
When a new pack appears in Arena, Flexo (a smug robot-wizard persona) speaks a
short pick recommendation, shows the top 3 on an OBS overlay, and logs whether
you agreed. You can talk back (push-to-talk) and he reacts in character and
updates his model of your deck.

It was extended from a single-file console assistant
(`mtga_draft_assistant.py`, kept in `~/Downloads`). The log parsing and card
mapping are ported from that script; everything else (17Lands data, voice,
persona, overlay, scoreboard, replay) is new.

## Scope decisions (locked)

- **Quick Draft only.** The parser keys on Arena's bot-draft log events
  (`BotDraftDraftStatus` / `BotDraftDraftPick`). Premier/Traditional (human)
  draft uses different events and is out of scope for v1.
- **Set-agnostic.** We do **not** hardcode a set. The set code is read from the
  log's `EventName` (`QuickDraft_SOS_...` → `SOS`) and 17Lands data is fetched
  for that set at runtime, so "whatever set comes up" works.
- **17Lands did not exist in the original script.** The original mapped names
  from Scryfall only. `seventeen_lands.py` is new and is the source of GIH WR.
- **Pick-Two draft was considered and deferred.** 17Lands *does* expose a
  `PickTwoDraft` format, and the pack payload carries `NumCardsToPick` (1 for
  Quick Draft, would be 2 for Pick-Two). But Pick-Two's Player.log schema was
  never captured, so v1 stays on Quick Draft. See "Extending to Pick-Two" below.

## Module map (how it connects)

```
flexo_live.py (orchestrator)
  │
  ├─ flexo/log_watch.py   parse Player.log → PackEvent / PickedEvent / CompletedEvent
  │                       (follow_live tails a real log; follow_replay drives a saved one)
  ├─ flexo/cards.py       arena_id → name via Scryfall (cached to card_map.json)   [PORTED]
  ├─ flexo/seventeen_lands.py  set+format → {name: GIH WR/ATA/ALSA}, cached in data/  [NEW]
  ├─ flexo/draft_state.py DraftState: pool, plan/colors, current pack, last reco, disagreements
  ├─ flexo/persona.py     Flexo system prompts + JSON schemas (pick + banter)
  ├─ flexo/advisor.py     Claude calls: recommend(pack) and respond(you talking back)
  ├─ flexo/audio.py       SpeechQueue: single cancellable playback worker (v1.1)
  ├─ flexo/tts/           swappable TTS: ElevenLabs streaming PCM (if key) else
  │                       edge-tts (free, non-streaming via base-class fallback)
  ├─ flexo/stt.py         push-to-talk: hold F8 → faster-whisper; F8 DOWN also
  │                       fires the barge-in callback; early STT while held
  ├─ flexo/scoreboard.py  draft_log.csv (per pick) + human_vs_ai_score.json (per match)
  └─ flexo/web/           FastAPI + websocket overlay; serves overlay.html for OBS

score.py                  CLI to award matches to the Human vs AI tally
scripts/make_replay_fixture.py   turn a saved DraftLog into a replayable Player.log
```

### Event flow for one pick

1. Arena writes a pack to `Player.log`. `log_watch` emits a **PackEvent**.
2. Orchestrator maps IDs → names, calls `advisor.recommend(...)`.
3. Advisor asks 17Lands for each card's GIH WR, builds a small prompt, calls
   Claude with a structured-output schema, returns a `Recommendation`
   (top-3 + confidence + a ≤40-word spoken line). **GIH WR shown to you is looked
   up locally, never taken from the model.**
4. Orchestrator speaks the line (TTS worker thread) and pushes the overlay.
5. You pick in Arena → `==> BotDraftDraftPick` → **PickedEvent**. Orchestrator
   compares your pick to Flexo's top, appends a row to `draft_log.csv`, adds the
   card to the pool, and clears the pending reco.

## Key design decisions

- **One event stream, two drivers.** `LogParser.feed(line)` holds the tiny
  pending-marker state machine. `follow_live` (tail) and `follow_replay` (saved
  file) both drive it, so the live path and the test path exercise identical
  parsing.
- **Model = Haiku 4.5 by default.** Live picks run 40+ times/draft and must
  speak within ~3s. Haiku 4.5 is fast and supports structured outputs. Swap to
  `claude-opus-4-8` in `config.yaml` for stronger picks at higher latency.
  (`claude-sonnet-4-6` from the original script is a still-valid alias but was
  not a good default here — it's neither the fastest nor the strongest.)
- **Structured outputs** (`output_config.format`) guarantee parseable JSON, so
  the pick/banter results never need brittle text parsing. A model/network
  failure returns `{}` and is swallowed — the draft loop never dies on it.
- **Voice never overlaps — and is always cancellable (v1.1).** All speech goes
  through one `SpeechQueue` (`flexo/audio.py`): a single worker thread owns the
  output device, playback writes ~50ms PCM blocks and checks a per-utterance
  cancel event between writes. See "Audio architecture (v1.1)" below.
- **Push-to-talk ONLY.** The mic is opened only while F8 is held (`stt.py`), so
  narrating to viewers never reaches Flexo. This was a hard requirement.
- **Overlay uses a version-counter poll over the websocket**, not cross-thread
  event-loop scheduling. The orchestrator (its own thread) bumps a counter;
  connected browsers re-render on change. Simple and thread-safe.
- **Agreement is derived from the log, not from voice.** `agreed` in the CSV is
  `your actual pick == Flexo's top`. Voice "you win this one" only flavors banter
  and updates the plan; match wins are awarded manually via `score.py`.

## Audio architecture (v1.1): queue, cancellation, streaming, barge-in

v1's pipeline was fully sequential (PTT release → STT → Claude → full TTS →
playback) and playback was fire-and-forget, so Flexo was slow to speak and
talked over you. v1.1 restructures the audio path; log parsing and card
mapping are untouched.

### The speech queue (`flexo/audio.py`)

- **One `SpeechQueue` owns all voice output.** A single worker thread plays one
  `Utterance` at a time. Utterances carry a kind (`pack` | `banter` | `system`),
  a pack-generation `seq`, and a cancel event.
- **Cancellable mid-word.** Playback writes ~50ms PCM blocks and checks the
  cancel event between writes; on cancel the stream is `abort()`ed (device
  buffer dropped). Measured kill latency in the simulated test: **~42ms**.
- **Hard barge-in on F8 DOWN** (`interrupt_and_hold`): kills current audio,
  cancels + flushes every queued utterance, and blocks playback until the
  transcript resolves (`release_hold`), so Flexo can't resume talking while
  you're mid-sentence or while his reply is being generated. A stray tap
  releases the hold via the STT abort callback.
- **Stale packs never speak.** Every `PackEvent` (and every completed pick)
  bumps a generation counter. A new pack event drops queued lines about older
  packs; an advisor call that finishes for a stale generation is discarded
  before it's even enqueued. A line already mid-air is left to finish — only
  barge-in cuts audible speech.
- **Debounce while holding F8.** Events landing during a hold still kick off
  Claude calls (pipelining), but their lines queue behind the hold and
  stale-replace each other, so after you finish talking at most the latest
  pack line remains. If the latest pick line never aired, `_on_transcript`
  cancels it and passes it to `advisor.respond(unspoken_reco=...)` so Flexo
  folds the pick into his answer instead of reading a stale line afterwards.
- **Post-barge-in persona.** `interrupt_and_hold()` returns whether Flexo was
  audibly cut off; that flag reaches the banter prompt, which allows at most
  one clause of acknowledgment and forbids resuming/repeating the interrupted
  line.

### Latency work

- **Streaming TTS.** ElevenLabs now uses the `/stream` endpoint with
  `output_format=pcm_22050` — raw PCM chunks play the moment they arrive (MP3
  can't be decoded incrementally with soundfile; PCM sidesteps it). edge-tts
  stays non-streaming through the base-class `stream_pcm` fallback, played
  through the same cancellable pipeline.
- **`tts.model`** selects the ElevenLabs tier; default `eleven_flash_v2_5`
  (fastest, banter-quality). A/B against `eleven_turbo_v2_5` by editing config.
- **Keep-alive session.** One `requests.Session` per provider saves
  ~200–400ms of TLS setup on every utterance after the first.
- **Pack reactions are pipelined.** `_on_pack` spawns the Claude call in its
  own thread immediately; the log-tail loop never waits on Claude or TTS.
- **Early STT.** While F8 is held, speech is cut into segments at silence gaps
  (>=600ms silence after >=1.2s speech) and transcribed in the background as
  you talk; at release only the tail remains. faster-whisper has no true
  streaming API — silence-boundary chunking is the practical equivalent and
  never splits words. The whisper model is also warmed up at startup so the
  first hold doesn't pay ctranslate2's lazy-init cost. Toggle via
  `voice_input.early_stt`.
- **Claude call audit:** the pick prompt already sent only the current pack's
  cards (with their GIH WR) + a 45-card pool tail + one plan line — the full
  17Lands table is never sent. The static persona/system prompt now goes
  through prompt caching (`cache_control: ephemeral`); note this is a no-op
  until the system prompt exceeds the model's minimum cacheable size (2048
  tokens on Haiku), but it's free and future-proof.
- **`--timings` flag** logs ms per stage: STT (after release, with
  segments-pre-transcribed count), LLM, TTS-first-audio, and playback-start.
  The TTS/playback deltas are measured from *enqueue*, so they include any
  wait behind a currently-playing line — a large number there usually means
  queue wait, not synthesis time.

### Measured before/after (real ElevenLabs key, 2026-07-18)

TTS stage, time until playback could start (25-word Flexo-length line):

| Path | First audio |
|---|---|
| v1: turbo, full synth + decode, cold | ~1060–1130ms |
| v1.1: flash, streaming, cold (first line of session) | ~740ms |
| v1.1: turbo, streaming, warm keep-alive | ~270ms |
| v1.1: flash, streaming, warm keep-alive | **~130–200ms** |

End-to-end on the replay fixture (`--timings`, 6-pack truncated run): pack
event → speech starts ≈ **LLM 3.0–3.7s + TTS ~0.2s**. The TTS stage dropped
from ~1s to ~0.2s; the Haiku pick call now dominates the pipeline — further
latency work should target the LLM stage (shorter spoken lines, smaller
model, or speculative TTS on the first sentence).

## Replaying / testing without a live draft

Your live `Player.log` had rotated (no draft data), so there was no real log to
replay. But the MTGA_Draft_17Lands app had saved a real SOS Quick Draft under
`C:\MtgaDraftTool\Logs\`. That saved log keeps the real Arena **payload** lines
but strips the `<==` **marker** lines the parser needs.

`scripts/make_replay_fixture.py` re-inserts those markers, producing
`tests/fixtures/player_sos_quickdraft.log` — a faithful Player.log with **real
card IDs** (so `--replay` speaks real card names). Verified: 44 pack payloads,
43 pick requests, set=SOS, P1p1 = 14 cards, `NumCardsToPick=1`.

```
python flexo_live.py --replay tests/fixtures/player_sos_quickdraft.log --replay-delay 6 --no-voice
```

Note: with a short `--replay-delay` (under ~4s), packs arrive faster than the
~3s Claude call, so most advice is discarded as stale by design — you'll see
`(stale advice for pack X pick Y discarded)` and those picks get no CSV row.
Use a delay of 6+ to hear every line.

The audio layer has key-free, device-free unit tests (plain scripts, no pytest):

```
python tests/test_speech_queue.py   # simulated PTT interrupt: barge-in kill (~42ms),
                                    # queue flush, hold/release, stale-pack replacement
python tests/test_stt_early.py      # early-STT segmentation + abort, fake whisper model
```

To capture a fresh fixture: run a Quick Draft in Arena with **Detailed Logs
(Plugin Support)** on, then point `--replay` at the real `Player.log`, or re-run
the fixture script against a new saved DraftLog.

## Setup

1. `pip install -r requirements.txt` (Python 3.11+; 3.13 works).
2. Set `ANTHROPIC_API_KEY` (required). Optionally `ELEVENLABS_API_KEY`
   (else free edge-tts). Copy `.env.example` if you use a `.env`.
3. In Arena: Options → Account → **Detailed Logs (Plugin Support)**.
4. `python flexo_live.py` — it waits for Arena if it isn't running, then says
   "Flexo online" once the log is live.
5. Add `http://127.0.0.1:7337` as an OBS **Browser Source** (~400px wide).

## Known issues / gotchas

- **First run downloads the Scryfall bulk map** (~80–120 MB) to `card_map.json`.
  One time; subsequent runs are instant.
- **First 17Lands fetch per set** hits the network; cached in `data/` for
  `cache_hours`. If a set's Quick Draft data isn't published yet, lookups return
  `None` and Flexo advises on names only (WR shows `--`). Never crashes.
- **17Lands may return a card list with NULL win rates.** The endpoint returns
  the set's cards even when it has no aggregated play data (all WR fields
  `null`, `seen_count` ~0). Confirmed 2026-07 for SOS/EOE and even BLB across
  QuickDraft *and* PremierDraft with date ranges — the endpoint served lists but
  no stats. `ensure_set` detects "no usable WR" and **falls back to
  PremierDraft** (community-standard proxy); if that's also empty it runs
  names-only. So `--` on the overlay means 17Lands has no numbers for that set
  yet, not a bug. The full brain (Claude picks + persona + structured output)
  works regardless — verified live on a real SOS pack.
- **The `data/` cache holds a null-WR result for `cache_hours`.** If 17Lands
  starts serving stats for a set within that window, delete `data/<SET>_*.json`
  (or lower `cache_hours`) to force a refetch.
- **PCM streaming needs a pcm_* output format.** `tts.streaming` silently
  degrades to the full-MP3 path if `tts.output_format` isn't `pcm_*` (MP3
  can't be chunk-decoded with soundfile). `pcm_44100` needs a paid ElevenLabs
  tier; `pcm_22050` (default) is fine. If a streaming request fails mid-flight,
  the SpeechQueue retries the line once via full synth, then drops it.
- **`--timings` playback numbers are measured from enqueue**, so a line that
  waits behind a currently-playing utterance shows a large playback-start —
  that's queue wait, not synthesis latency. Compare tts-first-audio deltas on
  an idle queue.
- **`voice_input.silence_rms` may need tuning per mic.** Early STT counts a
  block as silence below this RMS (default 0.008). A noisy mic that never dips
  below it simply never cuts early segments — everything still transcribes at
  release, just without the head start.
- **MP3 playback needs a modern libsndfile.** `soundfile` 0.12.x wheels bundle
  libsndfile ≥1.1 (MP3 support). If playback fails, that's the first suspect.
- **edge-tts versions rot — keep it current.** Microsoft rotates the token
  (`Sec-MS-GEC`) scheme their free endpoint expects; older edge-tts releases then
  get `403 Invalid response status` on every synth. Seen 2026-07 with the
  originally-pinned `7.0.0`; `>=7.2.8` fixed it. If Flexo suddenly goes silent
  with `[tts:edge] speak failed: 403`, `pip install -U edge-tts` first. (This is
  why requirements pins `edge-tts>=7.2.8`, not `==`.) ElevenLabs is unaffected.
- **Model output can contain non-ASCII** (em-dashes, smart quotes). `flexo_live.main`
  reconfigures stdout/stderr to `utf-8, errors="replace"` at startup so printing
  Flexo's lines can't `UnicodeEncodeError` on a cp1252 console.
- **faster-whisper is heavy** (pulls ctranslate2). If it or the mic is
  unavailable, the orchestrator logs a warning and runs **without** voice input
  rather than crashing. `--no-voice` skips it entirely. Installed + smoke-tested
  2026-07-11; the `small` model is pre-cached under `~/.cache/huggingface`. The
  full interactive push-to-talk (hold F8 → speak → transcript) still needs an
  interactive run to confirm. **F8 is a *global* hotkey and is not suppressed** —
  make sure F8 isn't bound in Arena/OBS, or change `voice_input.hotkey`.
- **Duplicate packs are de-duped by `(pack, pick, exact cards)`** across the
  whole draft (`LogParser._seen`), not just the previous pack — so an Arena
  status re-send or reconnect can't make Flexo re-advise a pick. Verified: the
  SOS fixture drops from 44 raw pack payloads to 42 clean decisions. The set
  clears on a `Completed` event so back-to-back drafts in one session stay clean.
- **Structured-output first call has a one-time schema-compile latency.** The
  very first pick may be a touch slower than the ~3s target; steady state is fast.
- **The saved-log fixture has no `Completed` event** (the app's DraftLog ends at
  the last pick). Replay calls the completion summary itself at end-of-file, so
  this doesn't matter in practice.
- **Windows console encoding:** keep `print()` strings ASCII. Non-ASCII (em
  dashes, etc.) can raise `UnicodeEncodeError` on a cp1252 console. Docstrings
  and the served HTML (UTF-8) are fine.

## Extending to Pick-Two draft (future)

Do NOT guess the schema. Capture a real Pick-Two `Player.log` first. Then:
1. Determine whether Arena logs it as bot draft (reuses `BotDraft*` markers with
   2 IDs per pick) or human draft (entirely different events).
2. `PickedEvent.card_ids` and `pool.extend` already handle multi-card picks; the
   advisor prompt and `record_turn` would need to reason about a **pair** and
   partial agreement (0/1/2 of two).
3. Set `seventeen_lands.format: PickTwoDraft` and branch on `NumCardsToPick == 2`.
