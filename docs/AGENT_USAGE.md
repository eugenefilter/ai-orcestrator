# Using the sound knowledge base to build patches in Ableton

For an AI agent that builds tracks through the Ableton MCP. Read `/Users/eugene/Documents/Ableton/MCP/AGENTS.md` first
(limits and traps of the Ableton bridge).

## What the base holds
- `sound`: an audio file, its analysis (summary, pitch, pitch drop, envelope, band energies, harmonics, PNG spectrogram),
  the user's tags and notes, and patch recipes.
- `note`: free-form know-how. `device_map`: parameters of a device as read from the MCP.

## Tools (MCP server `soundkb`)
`search_knowledge`, `get_entry`, `analyze_audio`, `ingest_audio`, `compare_audio`, `save_patch_recipe`, `save_note`,
`save_device_map`, `list_tags`. Paths must be inside `ALLOWED_INGEST_DIRS` (see `.env`).

## Workflow: recreate a sound
1. `search_knowledge("deep 808 pitch drop")`, then `get_entry` for the best hits. Read the notes and any recipes.
   If the user gave a file instead, `analyze_audio(path)` and open the returned `png_path` with your image reader.
2. Look for `device_map` entries of the synth you plan to use. If missing, load the device on a track
   (`load_instrument_or_effect`), read `get_device_parameters`, and `save_device_map` what you read. Never set a
   parameter from memory.
3. Build a first patch from the analysis: pitch -> oscillator tuning; pitch drop -> pitch envelope amount/decay;
   attack/decay -> amp envelope; harmonic profile -> waveform/drive (a pure sine has harmonics 2-5 below -30 dB).
   Write one MIDI note (`midi` in the recipe) and fire it.
4. The MCP cannot record audio. Ask the user to render one note (Cmd+Shift+R, or resample) and give you the file path.
5. `compare_audio(candidate_path, reference_id=...)`. Change only the parameters the `advice` points to, one or two at a
   time. Repeat from step 4 until `distance` stops improving (cap: 8 iterations).
6. `save_patch_recipe(entry_id, recipe)` with the final values, `confidence` and the `verified` block.

`distance` is 0 for identical sounds, about 1 when every feature is just off tolerance, 3 when completely different.

## Rules
- Track indices are not stable: find the track by name in `get_session_snapshot(include_notes=False, include_params=False)`.
- Add only what the task needs: no extra devices or settings.
- Check `get_remote_script_info` (`up_to_date: true`) before device-parameter calls.
- Mark guesses as `confidence: "guess"`; only verified recipes are facts.

## Example: Wolf 808
Reference: `UNISON_808_Wolf (C).wav` (entry "Wolf 808 (C)"). Analysis: 33.25 Hz (C1, +29 cents), pitch falls 40 -> 33.25 Hz
by 377 ms, attack 75 ms, -40 dB at 2.7 s, 87% of energy below 60 Hz, 3rd harmonic -11 dB (a slightly saturated sine).
Plan: Operator, sine on oscillator A, pitch envelope for the fall, long amp decay, light Saturator; tune to C1 (+29 cents).
