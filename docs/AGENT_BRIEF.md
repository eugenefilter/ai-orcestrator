# Agent brief: Sound Knowledge Base for AI-driven synth patching in Ableton

You are an AI coding agent. Build the project described below inside this folder
(`/Users/eugene/Documents/Ableton/AI orcestrator`). Work through the phases in order, one phase at a time.
Each phase has acceptance criteria: do not start the next phase until they pass. Report at the end of each phase
what you ran and what it printed. Do not claim something works without having run it.

The user speaks Russian; code, comments, docs and commit messages stay English.

## 0. Goal and idea

An AI agent (Claude) cannot hear audio. It can read numbers and images. The goal is a **knowledge base of sounds**
that turns audio into something the agent can read and reason about, and that stores *how to recreate each sound
in Ableton Live*. With it, an agent can build tracks in Ableton through the existing Ableton MCP by turning synth
knobs on its own, instead of guessing.

The knowledge base holds three kinds of entries:

1. **sound**: an audio file plus its automatic analysis (JSON + spectrogram PNG) plus user tags and notes plus,
   once found, a **patch recipe** (synth, device parameters) that reproduces it in Ableton.
2. **note**: free-form knowledge without audio ("808 in Operator: sine carrier, pitch envelope ...").
3. **device_map**: facts about an Ableton device's parameters (names, index, normalised range, unit conversion),
   read through the Ableton MCP, so later agents do not rediscover them.

Three parts to build:

- **A. Audio analysis library** (audio file -> agent-readable JSON + PNG).
- **B. Storage**: Qdrant in Docker (vectors + payload), plus a local folder for the files themselves.
- **C. Interfaces**: a web UI (upload a file, set tags, write extra knowledge) and an MCP server (so the agent
  can search the base, analyze new audio, compare renders and save recipes).

First sample: `/Users/eugene/Documents/Ableton/Loops/Loops/808s/UNISON_808_Wolf (C).wav`. It must work end to end
on this file (the path contains spaces and parentheses: always pass paths as arguments, never build shell strings).

## 1. Environment facts (verified on this machine, 2026-10-23)

- macOS (Darwin 24.6) on **Intel (x86_64)**, `uv` at `/usr/local/bin/uv`, Docker 29.2.1. System Python is 3.13.5, but
  ONNX Runtime (needed by `fastembed`) has no Intel-mac wheels for 3.13, so the project pins `requires-python >=3.11,<3.13`
  and runs on 3.12 in `.venv`. UI port is 8766 (`.env.example`), not 8765.
- The target folder was empty when this brief was written.
- Ableton Live 11.3 Suite. The existing Ableton MCP lives in `/Users/eugene/Documents/Ableton/MCP`
  (read its `AGENTS.md` before touching Ableton; it records limits and traps). Key facts you must respect:
  - MCP can read/set parameters of **top-level devices only** (not devices inside racks/pads); parameters are
    **normalised 0..1** for many controls, with per-device conversions (frequency `f = 10 * 2200**v` for EQ Eight, etc.).
  - MCP **cannot export or record audio**. The user renders audio manually (Cmd+Shift+R, or Resampling), so the
    compare loop (phase 7) needs a file path that the user supplies after each iteration.
  - Track indices are unstable: look a track up by name via `get_session_snapshot(include_notes=False, include_params=False)`.
  - `get_remote_script_info` must say `up_to_date: true` before any device-parameter call.
  - Add only what the task asks for. No extra devices or settings "for convenience".
- Do not hard-code absolute user paths in code. Take paths from config / env (`.env`, `config.toml`).

## 2. Tech choices (use these unless a phase fails; then explain the change)

| Concern | Choice |
|---|---|
| Package manager | `uv` (`pyproject.toml`, `uv.lock`, Python >= 3.11) |
| Analysis | `librosa`, `scipy`, `numpy`, `soundfile`, `pyloudnorm`, `matplotlib` (Agg backend) |
| Pitch | `librosa.pyin` first; make `torchcrepe` an optional extra |
| Text embeddings | `fastembed` (local ONNX, no API key), e.g. `BAAI/bge-small-en-v1.5` |
| Audio embeddings | CLAP via `transformers` (`laion/clap-htsat-unfused`) as an **optional extra**; the base must work without it |
| Vector DB | Qdrant, Docker, named vectors `text` and (optional) `audio` |
| Backend + UI | FastAPI + one static HTML page (vanilla JS, no build step) |
| MCP server | official `mcp` Python SDK (FastMCP), stdio transport |
| Tests | `pytest`; synthetic signals generated in the tests, no copyrighted samples committed |

Keep dependencies minimal. Heavy ones (CLAP/torch) go in optional extras: `uv sync --extra audio-embed`.

## 3. Project layout

```
AI orcestrator/
  README.md                 # how to run (short)
  docs/AGENT_BRIEF.md       # this file
  docs/PATCH_RECIPE.md      # schema + examples of recipes (written in phase 6)
  pyproject.toml
  .env.example              # QDRANT_URL, DATA_DIR, UI_PORT
  .gitignore                # data/, .env, __pycache__, .venv
  docker-compose.yml        # qdrant
  data/                     # gitignored: audio/, analysis/, qdrant/ (volume)
  src/soundkb/
    config.py
    analysis/
      load.py               # robust audio loading, mono mix, sample rate
      features.py           # all feature extractors (pure functions)
      report.py             # assemble report dict + PNG
      compare.py            # distance between two reports/audios
    store/
      qdrant_store.py       # collection setup, upsert, search, delete
      embed.py              # text embeddings (+ optional audio)
      schema.py             # pydantic models: Entry, Analysis, PatchRecipe
    api/
      app.py                # FastAPI
      static/index.html     # UI
    mcp_server.py           # MCP tools for the agent
    cli.py                  # `soundkb analyze|ingest|search`
  tests/
```

## 4. Phases

### Phase 1: scaffold

1. `uv init` style project, `pyproject.toml` with the dependencies above, a `soundkb` console script.
2. `.gitignore` excludes `data/`, `.env`. `git init` only if the folder is not already a repo; do not commit
   anything unless the user asks.
3. `config.py` reads `.env` (defaults: `QDRANT_URL=http://127.0.0.1:6333`, `DATA_DIR=./data`, `UI_PORT=8765`).

Accept: `uv sync` succeeds; `uv run soundkb --help` prints usage.

### Phase 2: analysis library (part A)

`analysis/features.py` implements pure functions over a mono float32 array `y` and sample rate `sr`
(load with `soundfile`, keep original `sr`, do not resample for loudness/duration; resample a copy only if a
feature needs it). Handle stereo (report mid and side energy ratio), silence, DC offset, very short files.

Features to extract (each returns plain Python numbers/lists, rounded sensibly):

- **meta**: duration (s), sr, channels, peak (dBFS), RMS (dBFS), integrated loudness (LUFS, only if >= 0.4 s),
  stereo width (side/mid energy ratio), DC offset, leading silence (s).
- **pitch**: f0 curve with `librosa.pyin` (fmin 20 Hz, fmax 2 kHz for bass sounds; make range configurable),
  voiced mask, then summaries: `f0_start` (median of first 20 ms voiced), `f0_at` for t in [0.05, 0.1, 0.2, 0.5, 1.0 s],
  `f0_steady` (median of the middle of the sustain), nearest note name + cents offset, and a **pitch-drop descriptor**
  `{start_hz, end_hz, time_to_90pct_ms}` for the first 300 ms. Low notes need a long window: use `frame_length`
  >= 4 periods of fmin.
- **amplitude envelope**: Hilbert-magnitude envelope smoothed over one f0 period (short RMS windows ripple on bass
  notes), 5 ms hops; attack time (10% of peak -> -3 dB; a 90% threshold is unstable on soft flat tops), time of peak, decay times to -6/-20/-40 dB,
  tail length (time until -60 dB), sustain level estimate, a boolean `is_one_shot` guess (monotonic decay).
- **spectrum**: average spectrum summary (energy in sub <60 Hz, bass 60-250, low-mid 250-2k, high-mid 2k-6k, high >6k, in %),
  spectral centroid / rolloff / flatness over time sampled at the same checkpoints as f0, and **harmonic profile**:
  amplitude of harmonics 1..16 of the steady f0, in dB relative to harmonic 1, plus even/odd ratio. This is what
  decides "sine vs saw vs distorted sine".
- **distortion/saturation hints**: THD estimate from the harmonic profile, crest factor, clipping flag.
- **modulation**: vibrato/LFO rate and depth on f0 and on amplitude (only if detected, else null).
- **onset/rhythm** (for loops, optional): onset times, tempo estimate.

`analysis/report.py` builds:
- `report.json` (schema versioned: `"schema": 1`), compact (< 8 KB for a one-shot) with a top-level `summary`
  of 5-10 plain-English lines generated from the numbers (e.g. "sub-heavy: 78% energy below 60 Hz; f0 drops
  from 110 Hz to 55 Hz in 60 ms; amplitude decays to -40 dB in 1.4 s; 3rd harmonic -18 dB re fundamental").
  The summary is for the agent to read first; raw numbers below it.
- `report.png`: one figure, 4 panels: waveform + RMS envelope; log-frequency mel spectrogram (dB, 20 Hz-20 kHz);
  f0 curve (log y); harmonic bar chart. Titles and axis labels in English, readable at 1600 px wide.

CLI: `uv run soundkb analyze "<path>" --out <dir>` writes both files and prints the summary.

Tests (`pytest`): generate with numpy (a) a pure 55 Hz sine with exponential decay, (b) a sine with a pitch drop
110 -> 55 Hz in 60 ms, (c) a saw wave, (d) silence. Assert f0, drop time, attack, and harmonic profile are within
tolerance. Real sample is a manual check, not a unit test.

Accept: tests pass; running on the Wolf 808 produces plausible numbers and a PNG that you open (Read tool) and
describe back to the user in 3-4 lines (pitch, pitch drop yes/no, decay, harmonics). If numbers contradict the
picture, fix the extractor before moving on.

### Phase 3: Qdrant in Docker (part B)

`docker-compose.yml`:
- service `qdrant`, image `qdrant/qdrant` (pin a concrete version tag, not `latest`),
- ports bound to loopback only: `127.0.0.1:6333:6333` and `127.0.0.1:6334:6334`,
- volume `./data/qdrant:/qdrant/storage`,
- `restart: unless-stopped`, a healthcheck on `/readyz`,
- telemetry off: env `QDRANT__TELEMETRY_DISABLED=true`.

`store/qdrant_store.py`:
- Collection `soundkb_entries` with named vectors: `text` (size of the chosen embedding model, cosine) and, if the
  optional extra is installed, `audio` (CLAP size, cosine). Create idempotently on startup; store the embedding model
  name in a collection-level metadata entry so a model change is detected (refuse to mix, offer reindex).
- Point id = UUID5 of the file SHA-256 (for sound) or a random UUID (for note/device_map). Re-ingesting the same file
  updates the point instead of duplicating.
- Payload schema (pydantic, validate on write):
  ```
  kind: "sound" | "note" | "device_map"
  title: str
  tags: [str]                # normalised: lower-case, trimmed, spaces -> "-"
  notes: str                 # user's extra knowledge, free text, any language
  audio_path: str | null     # copied into data/audio/<sha256>.<ext>; original path kept in source_path
  source_path: str | null
  analysis: {...} | null     # the report.json content (summary + features)
  patch_recipes: [PatchRecipe]
  created_at, updated_at
  ```
- Payload indexes on `kind`, `tags` (keyword), `analysis.pitch.f0_steady` and `analysis.meta.duration` (float) for filters.
- Search = text embedding of `title + tags + notes + analysis.summary` against `text`, with optional filters
  (kind, tags all/any, f0 range, duration range). Optional `audio` search by example file.

Accept: `docker compose up -d`; `curl http://127.0.0.1:6333/readyz` is healthy; a script upserts two entries and finds them
by text and tag filter; data survives `docker compose down && up -d`.

### Phase 4: ingest pipeline

`soundkb ingest "<audio path>" --title ... --tag 808 --tag sub --note "..."`:
1. hash file, copy to `data/audio/`, run analysis (phase 2), save `data/analysis/<sha>/report.{json,png}`,
2. build embedding text, embed, upsert to Qdrant,
3. print the point id and the summary.

Supported input: wav, aif/aiff, flac, mp3 (via `soundfile`/`librosa` backends; report a clear error otherwise).
Reject files over a configurable size (default 100 MB) and clips over 60 s unless `--allow-long`.
Never modify the original file.

Accept: ingest of the Wolf 808 followed by `soundkb search "deep 808 sub bass with pitch drop"` returns it first.

### Phase 5: web UI (part C, point 3 of the user's request)

FastAPI app served on `127.0.0.1:UI_PORT` only (never `0.0.0.0`). Endpoints:

- `POST /api/entries` multipart: `file` (optional for note/device_map), `kind`, `title`, `tags` (comma list),
  `notes`, optional `patch_recipe` JSON text. Returns the entry with analysis.
- `GET /api/entries?q=&tag=&kind=` list/search; `GET /api/entries/{id}`; `PATCH` (title, tags, notes, recipes);
  `DELETE` (removes point and files, asks for confirmation in the UI).
- `GET /api/entries/{id}/report.png`, `/audio` (served for the in-browser player).
- `GET /api/tags` with counts (autocomplete in the UI).
- `POST /api/path-ingest` with a **local path** (the user's samples live on disk; avoid forcing an upload of large files).
  Only allow paths under directories listed in `ALLOWED_INGEST_DIRS` (default: `~/Documents/Ableton`, `~/Music`);
  resolve symlinks and `..` before checking.

`static/index.html` (single page, no framework needed):
- Left: drop zone / "choose file" / "or paste local path". Fields: title, kind, tags (chips with autocomplete from
  `/api/tags`, Enter to add), free-text "extra knowledge" (multi-line), optional "patch recipe JSON" (textarea with
  validation against the schema, plus a template button).
- After submit: show the analysis summary, the spectrogram PNG, and an `<audio>` player.
- Right/below: search box + tag filters + results list; click opens the entry for editing (tags, notes, recipe).
- Works in light and dark mode, usable at phone width, English UI labels (the user can type Russian notes: store UTF-8).

Accept: from the browser, upload the Wolf 808 with tags `808, sub, bass`, notes "Dark sub 808, C, long tail",
see the PNG and summary, find it again through search, edit a tag, delete a test entry. Take a screenshot to prove it.

### Phase 6: patch recipes and device maps (the part that makes the base useful)

Document `docs/PATCH_RECIPE.md` and implement `PatchRecipe` in `store/schema.py`:

```
PatchRecipe:
  name: str                      # "Wolf 808 via Operator"
  target: { live_version: "11.3", device: "Operator", browser_uri: "query:Synths#Operator" }
  params: [ { name: "Osc A Level", index: <int|null>, value: <float>, value_display: "0 dB",
              note: "optional" } ]
  chain: [ ... further top-level devices on the same track, same shape ... ]   # e.g. Saturator, EQ Eight
  midi: { note: 36, velocity: 100, length_beats: 4 }
  verified: { distance: <float>, iterations: <int>, date: "...", render_path: "..." } | null
  confidence: "guess" | "close" | "verified"
```

Rules for the schema: parameters are stored **as the MCP uses them** (names + normalised values as returned by
`get_device_parameters`), plus the human-readable display value, so the recipe can be replayed with `set_device_parameter`
without conversion mistakes.

`device_map` entries (kind `device_map`): for each device the agent actually uses (Operator first, then Wavetable,
Analog, Saturator, EQ Eight, Compressor, Auto Filter), record parameter name, MCP index, min/max display, unit,
normalised-to-real conversion formula when known, and gotchas. These entries are **written from real
`get_device_parameters` output**, never from memory: have the MCP server (phase 8) expose `save_device_map` so the
agent can store what it just read. Do not invent parameter values.

Seed notes (kind `note`), written by you only from facts you verified or that appear in `/Users/eugene/Documents/Ableton/MCP/AGENTS.md`;
anything else the user adds through the UI. Mark unverified seed text with tag `unverified`.

Accept: a recipe JSON round-trips through UI -> Qdrant -> MCP search unchanged; schema validation rejects a recipe
with a missing `target.device`.

### Phase 7: compare loop (render -> analyse -> adjust)

`analysis/compare.py`: `compare(ref_report_or_audio, candidate_audio) -> {distance, per_feature, advice[]}`:
- per-feature errors: f0 start/steady (cents), pitch-drop time (ms), attack (ms), decay -20/-40 dB (%), tail (%),
  band energy (pp difference), harmonic profile (mean dB error over 1..8), spectral centroid curve (%),
  log-mel spectrogram L1 distance after loudness-normalising both to the same LUFS and aligning onsets.
- `advice[]` is plain text for the agent, e.g. "candidate f0 drop is 25 ms shorter: increase Pitch Env Decay",
  "candidate 3rd harmonic is 9 dB lower: add drive / use Saturator". Keep advice rule-based and conservative.
- A single scalar `distance` (0 = identical), documented, so the agent can tell whether an iteration improved.

Workflow the docs must describe (for the agent that will use the base):
1. Find similar sounds: `search_knowledge`, read their recipes and the device maps.
2. Build the first patch on a track through the Ableton MCP (`load_instrument_or_effect`, `set_device_parameter`,
   notes in a clip).
3. Ask the user to render one note (Cmd+Shift+R or Resampling) and give the file path (MCP cannot record audio).
4. `compare_audio(reference_id, candidate_path)` -> change the parameters the advice points to -> repeat until
   `distance` stops improving or hits a threshold (cap iterations, default 8).
5. `save_patch_recipe(entry_id, recipe)` with `confidence: "verified"` and the final distance.

Accept: compare a sound with itself -> distance ~0; with a pitch-shifted/time-stretched copy -> distance clearly higher and
advice mentions the right feature. Unit tests for both.

### Phase 8: MCP server for the agent

`mcp_server.py` (stdio). Tools, each with a precise docstring (agents read these):

- `search_knowledge(query, kind=None, tags=None, limit=5)`: returns title, id, tags, analysis summary, notes, recipe names.
- `get_entry(id)`: full entry including analysis JSON and recipes (omit raw f0 arrays; give checkpoints only).
- `analyze_audio(path)`: runs phase 2 on any allowed local file, returns summary + features and saves the PNG
  (returns the PNG path so the agent can Read it). Does not store anything in Qdrant.
- `ingest_audio(path, title, tags, notes)`: store as a sound entry.
- `compare_audio(reference_id | reference_path, candidate_path)`: phase 7.
- `save_patch_recipe(entry_id, recipe)`, `save_note(title, text, tags)`, `save_device_map(device, parameters)`.
- `list_tags()`.

Path allow-list applies to every tool that takes a path. Return errors as readable messages, not stack traces.
Provide a ready-to-paste `claude mcp add` command in the README, e.g.
`claude mcp add soundkb -- uv run --directory "<project dir>" soundkb-mcp`. Do not register it yourself without
the user's go-ahead.

Accept: from a fresh Claude Code session with the server registered, the agent can: search for "808 sub", read the entry,
read the PNG, and call `compare_audio` on two files.

### Phase 9: agent-facing guide

Write `docs/AGENT_USAGE.md` (for the agent that builds tracks, not for you): the workflow from phase 7, what the
knowledge base contains, rules of thumb (read device maps before setting parameters; never set parameters from
memory; verify with a render; add only what the task needs), and the limits listed in section 1. Add one worked
example using the Wolf 808. Update the Ableton MCP project's `AGENTS.md` with a 5-line pointer to it **only if the
user agrees** (that repo is separate and has uncommitted changes).

## 5. Quality bars (apply to every phase)

- Every module has a typed public function surface and docstrings that say units (Hz, ms, dBFS, LUFS).
- Analysis is deterministic: same file -> same report (fixed parameters, no random seeds without fixing them).
- No network calls at runtime except localhost Qdrant; the embedding model downloads once on first use (say so in
  the README and cache under `data/models/`). Telemetry off everywhere.
- Never log or print file contents of `.env`. No secrets are needed for this project.
- Bind every server to `127.0.0.1`. Validate all user paths (resolve, allow-list). Limit upload size.
- Errors are explicit: unreadable audio, unsupported format, Qdrant down (tell the user to run `docker compose up -d`).
- Keep it simple: no auth, no multi-user, no cloud, no ORM, no frontend build tooling.
- Do not add features that were not asked for. If you think something is missing, list it at the end as a proposal.

## 6. Definition of done

1. `docker compose up -d` + `uv run soundkb-ui` gives a working UI at `http://127.0.0.1:8765`.
2. The Wolf 808 is ingested with tags and a note; its PNG and summary are correct.
3. MCP server works from Claude Code; search, analyze and compare return sensible results.
4. Tests pass (`uv run pytest`).
5. README shows install and run in under 10 lines; `docs/AGENT_USAGE.md` and `docs/PATCH_RECIPE.md` exist.
6. A final report to the user (in Russian): what was built, how to run it, what is not covered (audio cannot be
   recorded through the Ableton MCP; recipes need a manual render step), and proposals for next steps.
