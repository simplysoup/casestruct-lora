from __future__ import annotations
import json
from pathlib import Path

def report():
    folder = Path("reports")
    paths = sorted(folder.glob("*.metrics.json")) if folder.exists() else []
    if not paths:
        print("No evaluation metrics yet; run the pipeline first")
        return
    lines = ["# CaseStruct-LoRA evaluation", "", "Generated from saved prediction metrics; synthetic data only.", "",
             "| Run | Examples | JSON valid | Schema valid | Exact match | Macro F1 | Unsupported field rate |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for path in paths:
        m = json.loads(path.read_text())
        lines.append(f"| {path.name.removesuffix('.metrics.json')} | {m['count']} | {m['json_valid_rate']:.1%} | {m['schema_valid_rate']:.1%} | {m['exact_match_rate']:.1%} | {m['macro_field_f1']:.1%} | {m['unsupported_field_rate']:.1%} |")
    lines += ["", "## Caveats", "", "The dataset is synthetic, formulaic, and not representative of real legal claims. No real-world accuracy or fitness-for-use is implied.", "", "Not an assessment of legal merit or liability."]
    dest = folder / "summary.md"
    dest.write_text("\n".join(lines) + "\n")
    print(dest)
