#!/usr/bin/env python3
"""Export aggregate metrics/confusion counts without copying per-example reports."""
from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path

METRIC_FIELDS = (
    "count",
    "accuracy",
    "macro_f1",
    "strict_json_rate",
    "allowed_label_rate",
    "per_label",
)
TASKS = ("constructcie", "osha")


def _svg(data: dict) -> str:
    run = data["runs"]["v2-constrained"]
    matrices = run["confusion_matrices"]
    width, height = 1280, 820
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        "<title>V2 constrained confusion matrices</title>",
        '<rect width="100%" height="100%" fill="#101827"/>',
        '<text x="36" y="42" fill="#f8fafc" font-family="Arial,sans-serif" font-size="24" font-weight="700">V2 constrained · confusion counts</text>',
        '<text x="36" y="68" fill="#cbd5e1" font-family="Arial,sans-serif" font-size="13">Rows are gold labels; columns are predictions. This inspected test set is exploratory, not a clean holdout.</text>',
    ]
    for task, x, cell, legend_y in (("constructcie", 34, 68, 465), ("osha", 666, 43, 545)):
        matrix = matrices[task]
        labels = matrix["labels"][:-1]  # Omit the zero-count INVALID prediction column from the visual only.
        counts = matrix["counts"]
        max_count = max((counts[g].get(p, 0) for g in labels for p in labels), default=1) or 1
        title = "ConstructCIE" if task == "constructcie" else "OSHA injury nature"
        pieces.append(f'<text x="{x}" y="112" fill="#f8fafc" font-family="Arial,sans-serif" font-size="18" font-weight="700">{title}</text>')
        grid_x, grid_y = x + 115, 165
        pieces.append(f'<text x="{grid_x + cell * len(labels) / 2}" y="132" text-anchor="middle" fill="#cbd5e1" font-family="Arial,sans-serif" font-size="11">Predicted label</text>')
        for j, label in enumerate(labels):
            cx = grid_x + j * cell + cell / 2
            pieces.append(f'<text x="{cx}" y="158" text-anchor="middle" fill="#cbd5e1" font-family="Arial,sans-serif" font-size="10">{j + 1}</text>')
        for i, gold in enumerate(labels):
            cy = grid_y + i * cell
            pieces.append(f'<text x="{grid_x - 12}" y="{cy + cell / 2 + 4}" text-anchor="end" fill="#cbd5e1" font-family="Arial,sans-serif" font-size="10">{i + 1}</text>')
            for j, predicted in enumerate(labels):
                value = counts[gold].get(predicted, 0)
                intensity = value / max_count
                blue = int(34 + intensity * 170)
                green = int(52 + intensity * 135)
                red = int(75 + intensity * 85)
                color = f"#{red:02x}{green:02x}{blue:02x}"
                tx, ty = grid_x + j * cell, cy
                pieces.append(f'<rect x="{tx}" y="{ty}" width="{cell - 2}" height="{cell - 2}" rx="3" fill="{color}"/>')
                ink = "#0b1220" if intensity > 0.5 else "#f8fafc"
                pieces.append(f'<text x="{tx + (cell - 2) / 2}" y="{ty + cell / 2 + 4}" text-anchor="middle" fill="{ink}" font-family="Arial,sans-serif" font-size="12">{value}</text>')
        legend_x, y = x, legend_y
        pieces.append(f'<text x="{legend_x}" y="{y}" fill="#cbd5e1" font-family="Arial,sans-serif" font-size="11">Label key (numbered in the matrix):</text>')
        for i, label in enumerate(labels):
            col = 0 if len(labels) == 4 or i < 4 else 1
            row = i if col == 0 else i - 4
            lx = legend_x + col * 310
            ly = y + 20 + row * 18
            pieces.append(f'<text x="{lx}" y="{ly}" fill="#f8fafc" font-family="Arial,sans-serif" font-size="10">{i + 1}. {escape(label)}</text>')
    pieces.append('<text x="36" y="792" fill="#94a3b8" font-family="Arial,sans-serif" font-size="11">Source: saved v2.1 aggregate reports. Raw narratives and per-example predictions are not included.</text>')
    pieces.append("</svg>")
    return "\n".join(pieces) + "\n"


def export(source: Path, output: Path, plot: Path) -> None:
    runs = {}
    provenance = {"source": "saved v2.1 aggregate reports", "raw_narratives_included": False}
    for metric_path in sorted(source.glob("*-test.metrics.json")):
        metadata = json.loads(metric_path.read_text())
        run_name = metric_path.name.removesuffix("-test.metrics.json")
        confusion_path = source / f"{run_name}.confusion.json"
        if not confusion_path.is_file():
            raise FileNotFoundError(f"Missing aggregate confusion matrix: {confusion_path}")
        confusion = json.loads(confusion_path.read_text())
        selected_metrics = {}
        for task in TASKS:
            task_metrics = metadata["metrics"][task]
            selected_metrics[task] = {key: task_metrics[key] for key in METRIC_FIELDS if key in task_metrics}
            if "per_label" in selected_metrics[task]:
                selected_metrics[task]["per_label"] = {
                    label: {key: values[key] for key in ("support", "precision", "recall", "f1") if key in values}
                    for label, values in selected_metrics[task]["per_label"].items()
                }
            if sum(sum(row.values()) for row in confusion[task]["counts"].values()) != task_metrics["count"]:
                raise ValueError(f"Confusion counts do not sum to N for {run_name}/{task}")
        runs[run_name] = {
            "model": metadata["model"],
            "method": metadata["method"],
            "metrics": selected_metrics,
            "confusion_matrices": {task: confusion[task] for task in TASKS},
        }
        if "test_sha256" in metadata:
            provenance["test_sha256"] = metadata["test_sha256"]
    if len(runs) != 9:
        raise ValueError(f"Expected all 9 v2.1 conditions, found {len(runs)}")
    if "v2-constrained" not in runs:
        raise ValueError("v2-constrained results are required for the plot")
    statistics_path = source / "statistics.json"
    raw_statistics = json.loads(statistics_path.read_text())
    statistics = {}
    for task in TASKS:
        source_task = raw_statistics[task]
        statistics[task] = {
            "n": source_task["n"],
            "headline_model": source_task["headline_model"],
            "strongest_baseline": source_task["strongest_baseline"],
            "systems": {
                name: {"point": values["point"], "ci95": values["ci95"]}
                for name, values in source_task["systems"].items()
            },
            "paired_delta": source_task["paired_delta"],
        }
    result = {"provenance": provenance, "statistics": statistics, "runs": runs}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    plot.parent.mkdir(parents=True, exist_ok=True)
    plot.write_text(_svg(result))
    print(f"Wrote sanitized aggregates: {output}")
    print(f"Wrote confusion plot: {plot}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Directory containing saved v2.1 aggregate reports")
    parser.add_argument("--output", type=Path, default=Path("docs/results-aggregate.json"))
    parser.add_argument("--plot", type=Path, default=Path("docs/confusion-matrices-v2.svg"))
    args = parser.parse_args()
    export(args.source, args.output, args.plot)


if __name__ == "__main__":
    main()
