# Patch recipe schema

A recipe says how to recreate a sound in Ableton Live through the Ableton MCP. It is stored inside an entry
(`patch_recipes`) and validated by `soundkb.store.schema.PatchRecipe`.

Rules:
- Parameters are stored **as the Ableton MCP uses them**: the `name` and `index` from `get_device_parameters`, and the
  `value` that `set_device_parameter` accepts (usually normalised 0..1). `value_display` is the human reading
  ("-3 dB", "55 Hz") for people; never convert it back.
- Only top-level devices on the track (the MCP cannot reach devices inside racks).
- `confidence`: `guess` (built from analysis only), `close` (rendered and compared, not converged), `verified`
  (compare distance stopped improving; fill `verified`).

```json
{
  "name": "Sub 808 via Operator",
  "target": {"live_version": "11.3", "device": "Operator", "browser_uri": "query:Synths#Operator"},
  "params": [
    {"name": "Osc A Level", "index": 7, "value": 0.85, "value_display": "-3 dB"},
    {"name": "Pe Amount", "index": 23, "value": 0.2, "value_display": "+24 st", "note": "pitch envelope"}
  ],
  "chain": [
    {"device": "Saturator", "browser_uri": "query:AudioFx#Saturator",
     "params": [{"name": "Drive", "index": 2, "value": 0.3, "value_display": "6 dB"}]}
  ],
  "midi": {"note": 36, "velocity": 100, "length_beats": 4},
  "verified": {"distance": 0.42, "iterations": 5, "date": "2026-10-23", "render_path": "/Users/.../render.wav"},
  "confidence": "verified"
}
```

The indexes and names above are illustrative; always read the real ones with `get_device_parameters` first.
`target.device` and `name` are required; a recipe without them is rejected.
