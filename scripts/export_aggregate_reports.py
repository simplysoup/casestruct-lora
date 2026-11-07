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
TASK_LABELS = {"constructcie": "ConstructCIE accident type", "osha": "OSHA injury nature"}
RUN_LABELS = {
    "base-zero": "base zero-shot",
    "base-fewshot": "base few-shot",
    "base-constrained": "base constrained",
    "v1-zero": "v1 zero-shot",
    "v1-fewshot": "v1 few-shot",
    "v1-constrained": "v1 constrained",
    "v2-zero": "v2 zero-shot",
    "v2-fewshot": "v2 few-shot",
    "v2-constrained": "v2 constrained",
}


def _confusion_svg(data: dict) -> str:
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


def _headline_summary(data: dict, eval_commit: str | None, eval_date: str | None) -> dict:
    tasks = []
    for task in ("osha", "constructcie"):
        stats = data["statistics"][task]
        baseline = stats["strongest_baseline"]
        headline = stats["headline_model"]
        delta = stats["paired_delta"][baseline]["macro_f1"]
        tasks.append({
            "name": TASK_LABELS[task],
            "n": stats["n"],
            "metric": "macro_f1",
            "baseline": {"name": RUN_LABELS[baseline], "value": stats["systems"][baseline]["point"]["macro_f1"]},
            "v2": {"name": RUN_LABELS[headline], "value": stats["systems"][headline]["point"]["macro_f1"]},
            "paired_bootstrap_gain_pp": 100 * delta["delta"],
            "paired_bootstrap_95_ci_pp": [100 * value for value in delta["ci95"]],
        })
    provenance = {
        "status": "fresh v2.1 evaluation report export",
        "test_set_status": "inspected during development; not a pristine holdout",
        "evaluation_commit": eval_commit,
        "evaluation_date": eval_date,
        "aggregate_artifacts": "All nine methods' sanitized metrics, per-class scores, and confusion matrices are in docs/results-aggregate.json; v2-constrained matrices are visualized in docs/confusion-matrices-v2.svg.",
        "not_included": [
            "per-example predictions or raw narratives",
            "public W&B run URL (no verified public run link was supplied)",
        ],
    }
    return {**provenance, "tasks": tasks}


def _summary_svg(summary: dict) -> str:
    tasks = {item["name"]: item for item in summary["tasks"]}
    rows = [
        (tasks[TASK_LABELS["osha"]], 134, 118, 150),
        (tasks[TASK_LABELS["constructcie"]], 220, 204, 236),
    ]
    pieces = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="350" viewBox="0 0 900 350" role="img" aria-labelledby="title desc">',
        '<title id="title">Exploratory macro-F1 regenerated from the evaluation report</title>',
        '<desc id="desc">' + escape("; ".join(f"{r[0]['name']}: baseline {r[0]['baseline']['value']:.1%}, v2 {r[0]['v2']['value']:.1%}, n={r[0]['n']}" for r in rows)) + '</desc>',
        '<rect width="900" height="350" rx="16" fill="#101827"/>',
        '<text x="38" y="42" fill="#f8fafc" font-family="Arial,sans-serif" font-size="23" font-weight="700">Exploratory reported macro-F1</text>',
        '<text x="38" y="68" fill="#cbd5e1" font-family="Arial,sans-serif" font-size="13">Fresh report values · test sets were inspected during development</text>',
    ]
    for value, x in zip((0, 25, 50, 75, 100), (250, 390, 530, 670, 810)):
        pieces.append(f'<text x="{x}" y="100" fill="#94a3b8" font-family="Arial,sans-serif" font-size="12">{value}%</text>')
    for x in (250, 390, 530, 670, 810):
        pieces.append(f'<line x1="{x}" y1="106" x2="{x}" y2="286" stroke="#334155" stroke-width="1"/>')
    for item, title_y, baseline_y, v2_y in rows:
        title = escape(item["name"])
        baseline = item["baseline"]["value"]
        v2 = item["v2"]["value"]
        gain = item["paired_bootstrap_gain_pp"]
        low, high = item["paired_bootstrap_95_ci_pp"]
        x0, scale = 250, 5.6
        pieces.extend([
            f'<text x="38" y="{title_y}" fill="#f8fafc" font-family="Arial,sans-serif" font-size="17" font-weight="700">{title}</text>',
            f'<text x="38" y="{title_y + 20}" fill="#94a3b8" font-family="Arial,sans-serif" font-size="12">n = {item["n"]} · Δ +{gain:.1f} pp [{low:.1f}, {high:.1f}]</text>',
            f'<rect x="{x0}" y="{baseline_y - 16}" width="{baseline * scale:.1f}" height="25" rx="5" fill="#64748b"/>',
            f'<rect x="{x0}" y="{v2_y - 16}" width="{v2 * scale:.1f}" height="25" rx="5" fill="#38bdf8"/>',
            f'<text x="{x0 + baseline * scale + 7:.1f}" y="{baseline_y + 2}" fill="#f8fafc" font-family="Arial,sans-serif" font-size="13" font-weight="700">{baseline * 100:.1f}% baseline</text>',
            f'<text x="{x0 + v2 * scale + 7:.1f}" y="{v2_y + 2}" fill="#f8fafc" font-family="Arial,sans-serif" font-size="13" font-weight="700">{v2 * 100:.1f}% v2</text>',
        ])
    construct = tasks[TASK_LABELS["constructcie"]]
    construct_low = construct["paired_bootstrap_95_ci_pp"][0]
    note = f'Descriptive comparison only; ConstructCIE has n={construct["n"]} and a +{construct_low:.1f} pp lower confidence bound.'
    pieces.append(f'<text x="38" y="318" fill="#cbd5e1" font-family="Arial,sans-serif" font-size="12">{escape(note)}</text>')
    pieces.append('</svg>')
    return "\n".join(pieces) + "\n"


def export(source: Path, output: Path, plot: Path, eval_commit: str | None, eval_date: str | None) -> None:
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
    summary = _headline_summary(result, eval_commit, eval_date)
    summary_json = output.parent / "results-summary.json"
    summary_svg = output.parent / "results-summary.svg"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    plot.parent.mkdir(parents=True, exist_ok=True)
    plot.write_text(_confusion_svg(result))
    summary_svg.write_text(_summary_svg(summary))
    print(f"Wrote sanitized aggregates: {output}")
    print(f"Wrote headline summary: {summary_json}")
    print(f"Wrote confusion plot: {plot}")
    print(f"Wrote headline plot: {summary_svg}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Directory containing saved v2.1 aggregate reports")
    parser.add_argument("--output", type=Path, default=Path("docs/results-aggregate.json"))
    parser.add_argument("--plot", type=Path, default=Path("docs/confusion-matrices-v2.svg"))
    parser.add_argument("--eval-commit", help="code commit recorded before evaluation")
    parser.add_argument("--eval-date", help="date of the evaluation in ISO format")
    args = parser.parse_args()
    export(args.source, args.output, args.plot, args.eval_commit, args.eval_date)


if __name__ == "__main__":
    main()
