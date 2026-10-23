# SoundKB

Local knowledge base of sounds for AI-driven synth patching in Ableton. It analyzes audio into numbers and a
spectrogram an agent can read, stores sounds, notes and patch recipes in Qdrant, and exposes everything to agents
through an MCP server.

Intel macOS uses Python 3.12 (ONNX Runtime has no wheels for 3.13 there).

```sh
uv sync --extra dev
docker compose up -d          # Qdrant on 127.0.0.1:6333, data in ./data/qdrant
uv run soundkb-ui             # web UI on http://127.0.0.1:8766
```

The embedding model (multilingual MiniLM, ~220 MB) downloads once on first use into `data/models/`.

CLI: `soundkb analyze FILE --out DIR`, `soundkb ingest FILE --tag 808 --note "..."`, `soundkb search "deep sub"`,
`soundkb compare REF CAND`.

MCP server for Claude Code (register it yourself when ready):

```sh
claude mcp add soundkb -- uv run --directory "/Users/eugene/Documents/Ableton/AI orcestrator" soundkb-mcp
```

Docs: `docs/AGENT_BRIEF.md` (build brief), `docs/AGENT_USAGE.md` (how an agent uses the base),
`docs/PATCH_RECIPE.md` (recipe schema). Copy `.env.example` to `.env` to change ports or allowed folders.
