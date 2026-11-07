#!/usr/bin/env python3
"""Recompute paired metrics with the audit's train/test near-duplicate omitted."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from casestruct.v2.data import load_jsonl
from casestruct.v21.report import bootstrap

MODELS = ("base", "v1", "v2")
METHODS = ("zero", "fewshot", "constrained")


def run(report_dir: Path, data_dir: Path, reps: int = 10_000, seed: int = 71) -> dict:
    audit = json.loads((report_dir / "audit.json").read_text())
    candidates = [
        item for item in audit["near_duplicate_flags"]
        if {item["a_split"], item["b_split"]} == {"train", "test"}
        and item["a_task"] == item["b_task"]
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one same-task train/test candidate; found {len(candidates)}")
    candidate = candidates[0]
    test_id = str(candidate["a_id"] if candidate["a_split"] == "test" else candidate["b_id"])
    task = candidate["a_task"]

    classes = json.loads((data_dir / "manifest.json").read_text())["class_names"]
    all_rows = {}
    for model in MODELS:
        for method in METHODS:
            name = f"{model}-{method}"
            rows = list(load_jsonl(report_dir / f"{name}-test.predictions.jsonl"))
            matches = [row for row in rows if row["task"] == task and str(row["id"]) == test_id]
            if len(matches) != 1:
                raise ValueError(f"{name}: expected one prediction for the flagged test ID, got {len(matches)}")
            all_rows[name] = rows
    full = bootstrap(all_rows, classes, reps=reps, seed=seed)
    filtered_rows = {
        name: [row for row in rows if not (row["task"] == task and str(row["id"]) == test_id)]
        for name, rows in all_rows.items()
    }
    filtered = bootstrap(filtered_rows, classes, reps=reps, seed=seed)
    base_full = full[task]["strongest_baseline"]
    base_filtered = filtered[task]["strongest_baseline"]
    if base_full != base_filtered:
        raise ValueError("Strongest baseline changed; inspect both results before summarizing sensitivity")

    def method_summary(stats: dict, baseline: str) -> dict:
        delta = stats[task]["paired_delta"][baseline]["macro_f1"]
        return {
            "n": stats[task]["n"],
            "strongest_baseline": baseline,
            "baseline_macro_f1": stats[task]["systems"][baseline]["point"]["macro_f1"],
            "v2_constrained_macro_f1": stats[task]["systems"]["v2-constrained"]["point"]["macro_f1"],
            "gain_macro_f1": delta["delta"],
            "gain_ci95_macro_f1": delta["ci95"],
        }

    before = method_summary(full, base_full)
    after = method_summary(filtered, base_filtered)
    return {
        "task": task,
        "candidate": {
            "splits": ["train", "test"],
            "string_similarity": candidate["string_similarity"],
            "shingle_jaccard": candidate["shingle_jaccard"],
        },
        "bootstrap": {"resamples": reps, "seed": seed},
        "full_test": before,
        "without_candidate": after,
        "effect_percentage_points": {
            "v2_constrained_macro_f1": 100 * (after["v2_constrained_macro_f1"] - before["v2_constrained_macro_f1"]),
            "gain_macro_f1": 100 * (after["gain_macro_f1"] - before["gain_macro_f1"]),
            "gain_ci95_lower": 100 * (after["gain_ci95_macro_f1"][0] - before["gain_ci95_macro_f1"][0]),
            "gain_ci95_upper": 100 * (after["gain_ci95_macro_f1"][1] - before["gain_ci95_macro_f1"][1]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=Path("reports/v2.1"))
    parser.add_argument("--data", type=Path, default=Path("data/v2"))
    parser.add_argument("--output", type=Path, default=Path("docs/near-duplicate-sensitivity.json"))
    parser.add_argument("--reps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=71)
    args = parser.parse_args()
    result = run(args.reports, args.data, args.reps, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Wrote sanitized sensitivity results: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
