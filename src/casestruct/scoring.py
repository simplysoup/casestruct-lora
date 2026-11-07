from __future__ import annotations
import json
import re
from collections import defaultdict
from .schema import Extraction, FIELDS

def parse_prediction(raw: str):
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE).strip()
    try:
        obj = json.loads(s)
    except (ValueError, TypeError):
        return None, False, False
    if not isinstance(obj, dict):
        return None, True, False
    try:
        return Extraction.model_validate(obj).model_dump(mode="json"), True, True
    except Exception:
        return obj, True, False

def normalize(field, x):
    if isinstance(x, list):
        return sorted(set(str(v).strip().lower() for v in x))
    return x.strip().lower() if isinstance(x, str) and field != "incident_date" else x

def score(rows):
    if not rows:
        raise ValueError("no rows")
    totals = defaultdict(lambda: {"correct": 0, "total": 0, "tp": 0, "fp": 0, "fn": 0})
    valid_json = valid_schema = exact = unsupported = 0
    for row in rows:
        gold = Extraction.model_validate(row["label"]).model_dump(mode="json")
        predicted, j, s = parse_prediction(row["prediction"])
        valid_json += j
        valid_schema += s
        exact += int(s and all(normalize(k, predicted[k]) == normalize(k, gold[k]) for k in FIELDS))
        for k in FIELDS:
            item = totals[k]
            item["total"] += 1
            g = normalize(k, gold[k]); p = normalize(k, predicted.get(k)) if isinstance(predicted, dict) else None
            item["correct"] += int(s and p == g)
            if isinstance(g, list):
                gs = set(g)
                ps = set(p) if isinstance(p, list) else set()
            else:
                gs = {str(g)} if g is not None else set()
                ps = {str(p)} if p is not None else set()
            item["tp"] += len(gs & ps)
            item["fp"] += len(ps - gs)
            item["fn"] += len(gs - ps)
            if ps - gs:
                unsupported += 1
    n = len(rows)
    fields = {}
    for k, t in totals.items():
        precision = t["tp"] / (t["tp"] + t["fp"]) if t["tp"] + t["fp"] else 1.0
        recall = t["tp"] / (t["tp"] + t["fn"]) if t["tp"] + t["fn"] else 1.0
        fields[k] = {"accuracy": t["correct"] / n, "precision": precision, "recall": recall,
                     "f1": 2*precision*recall/(precision+recall) if precision+recall else 0.0}
    return {"count": n, "json_valid_rate": valid_json / n, "schema_valid_rate": valid_schema/n,
            "exact_match_rate": exact/n, "unsupported_field_rate": unsupported/(n*len(FIELDS)),
            "macro_field_f1": sum(x["f1"] for x in fields.values())/len(fields), "fields": fields}
