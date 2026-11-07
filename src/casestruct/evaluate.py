from __future__ import annotations
import json
import time
from pathlib import Path
from .data import load
from .prompts import template
from .runtime import config, model_and_tokenizer, seed_all
from .scoring import score

def evaluate(config_path: str, data_dir: str, split: str, mode: str, limit: int = 0):
    import torch
    c = config(config_path)
    seed_all(c["seed"])
    adapter = str(Path(c["output_dir"]) / "adapter") if mode == "adapter" else None
    if adapter and not Path(adapter).exists():
        raise FileNotFoundError(f"Adapter not found: {adapter}; run training first")
    rows = load(Path(data_dir), split)
    if limit:
        rows = rows[:limit]
    if not rows:
        raise ValueError("empty evaluation dataset")
    examples = load(Path(data_dir), "train")[:3] if mode == "few-shot" else []
    model, tokenizer = model_and_tokenizer(c["model_id"], adapter)
    results = []
    for i, row in enumerate(rows, 1):
        prompt = template(tokenizer, row["narrative"], examples)
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        start = time.perf_counter()
        with torch.inference_mode():
            ids = model.generate(**encoded, max_new_tokens=c["max_new_tokens"],
                                 do_sample=False, pad_token_id=tokenizer.pad_token_id,
                                 eos_token_id=tokenizer.eos_token_id)
        elapsed = time.perf_counter() - start
        prediction = tokenizer.decode(ids[0][encoded.input_ids.shape[-1]:], skip_special_tokens=True)
        results.append({**row, "prediction": prediction, "latency_s": round(elapsed, 4)})
        if i % 10 == 0 or i == len(rows):
            print(f"{mode} {split}: {i}/{len(rows)}", flush=True)
    metrics = score(results)
    metrics["average_latency_s"] = sum(x["latency_s"] for x in results) / len(results)
    output = Path("reports")
    output.mkdir(exist_ok=True)
    suffix = f"-{limit}" if limit else ""
    name = f"{mode}-{split}{suffix}"
    with (output / f"{name}.jsonl").open("w") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    (output / f"{name}.metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({k:v for k,v in metrics.items() if k != "fields"}, indent=2))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics
