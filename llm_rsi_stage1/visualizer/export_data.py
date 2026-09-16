"""
export_data.py - Bundles verified training traces and telemetry into visualizer/data.js
for offline browser viewing via file:// protocol.
"""

import json
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / "data"
vis_dir = base_dir / "visualizer"
vis_dir.mkdir(parents=True, exist_ok=True)

traces_file = data_dir / "verified_training_traces.jsonl"
summary_file = data_dir / "cycle_summary.json"

traces = []
if traces_file.exists():
    with open(traces_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                traces.append(json.loads(line))

summary = {}
if summary_file.exists():
    with open(summary_file, "r", encoding="utf-8") as f:
        summary = json.load(f)

bundle = {
    "summary": summary,
    "traces": traces
}

js_content = f"// Auto-generated telemetry data for Stage 1 LLM Inspector\nwindow.RSI_DATA = {json.dumps(bundle, indent=2)};\n"
with open(vis_dir / "data.js", "w", encoding="utf-8") as f:
    f.write(js_content)

print(f"✅ Exported {len(traces)} traces and summary to {vis_dir / 'data.js'}")
