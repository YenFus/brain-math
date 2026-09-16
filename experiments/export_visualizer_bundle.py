"""
export_visualizer_bundle.py - Packages experimental telemetry into visualizer/data.js
so the dashboard runs smoothly offline via file:// protocol without CORS blocking.
"""

import json
from pathlib import Path

data_dir = Path(__file__).resolve().parent / "data"
vis_dir = Path(__file__).resolve().parent.parent / "visualizer"
vis_dir.mkdir(parents=True, exist_ok=True)

exp1_file = data_dir / "exp1_trajectory.json"
exp2_file = data_dir / "exp2_results.json"
exp3_file = data_dir / "exp3_connectome.json"
synth_file = data_dir / "synthesis_results.json"

bundle = {}

if exp1_file.exists():
    with open(exp1_file, "r") as f:
        bundle["exp1"] = json.load(f)
if exp2_file.exists():
    with open(exp2_file, "r") as f:
        bundle["exp2"] = json.load(f)
if exp3_file.exists():
    with open(exp3_file, "r") as f:
        bundle["exp3"] = json.load(f)
if synth_file.exists():
    with open(synth_file, "r") as f:
        bundle["synthesis"] = json.load(f)

js_content = f"// Auto-generated telemetry data bundle\nwindow.TELEMETRY_BUNDLE = {json.dumps(bundle)};\n"

with open(vis_dir / "data.js", "w", encoding="utf-8") as f:
    f.write(js_content)

print(f"✅ Telemetry bundle exported to {vis_dir / 'data.js'} ({len(js_content) // 1024} KB)")
