"""Deterministic synthetic narratives; split by disjoint narrative families, not rows."""
from __future__ import annotations
import json
import random
from pathlib import Path
from .schema import Extraction, MISSING_KEYS

INCIDENTS = {
 "vehicle_collision": {
  "events": ["a delivery van struck the rear of their stopped car", "another driver changed lanes into their car", "a car ran a red light and hit their vehicle"],
  "factors": ["the other driver ran a red light", "the other driver was looking at a phone", "the other vehicle crossed the centre line"],
 },
 "slip_and_fall": {
  "events": ["they slipped on a wet grocery-store floor", "they fell on an icy stairway", "they slipped at an office entrance"],
  "factors": ["there was no warning sign", "the handrail was loose", "staff had not cleared the ice"],
 },
 "workplace_injury": {
  "events": ["a shelf collapsed while they were working", "a pallet jack struck their leg", "they fell from a poorly secured ladder"],
  "factors": ["the safety guard was missing", "the supervisor had reported the defect earlier", "the ladder was not anchored"],
 },
}
INJURIES = ["sprained wrist", "bruised shoulder", "lower back strain", "cut on left hand", "swollen ankle"]
TREATMENTS = ["an urgent care visit", "a physiotherapy appointment", "an emergency room assessment", "an X-ray"]
LOCATIONS = ["King Street, Toronto", "Main Street, Ottawa", "Elm Avenue, Hamilton", "Bay Street, Toronto"]
DISTRACTORS = ["They were wearing a blue jacket.", "They had already eaten lunch.", "Their phone battery was low."]
# Each split draws from separate narrative phrase and ordering families.
TEMPLATES = {
 "train": [
  "On {date_text}, at {loc_text}, {event}. {factor_text} {inj_text} {treatment_text} {distractor}",
  "Intake report: {event}. This occurred {date_text} near {loc_text}. {inj_text} {factor_text} {treatment_text} {distractor}",
  "The client says {event}; place: {loc_text}; date: {date_text}. {treatment_text} {inj_text} {factor_text} {distractor}",
 ],
 "validation": [
  "Regarding an incident dated {date_text} at {loc_text}: {event}. {inj_text} {treatment_text} {factor_text} {distractor}",
  "Claimant statement — {event}. Location was {loc_text}; time was {date_text}. {factor_text} {treatment_text} {inj_text} {distractor}",
 ],
 "test": [
  "Please record the following. {event}. It happened {date_text} around {loc_text}. {factor_text} {inj_text} {treatment_text} {distractor}",
  "After {event}, the person reported: {inj_text} {treatment_text} Site: {loc_text}. When: {date_text}. {factor_text} {distractor}",
 ],
 "challenge": [
  "An intake caller mentions {distractor} The event was that {event}. {treatment_text} {factor_text} Date entry: {date_text}; site entry: {loc_text}. {inj_text}",
  "Raw notes: {distractor} {inj_text} {event}. Location? {loc_text}. Date? {date_text}. {factor_text} {treatment_text}",
 ],
}

def one(rng: random.Random, split: str, i: int) -> dict:
    typ = rng.choice(list(INCIDENTS))
    config = INCIDENTS[typ]
    event = rng.choice(config["events"])
    has = {k: rng.random() > (0.34 if split == "challenge" else 0.23) for k in MISSING_KEYS}
    date = f"2025-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}" if has["incident_date"] else None
    location = rng.choice(LOCATIONS) if has["location"] else None
    injury = [rng.choice(INJURIES)] if has["injuries"] else []
    treatment = [rng.choice(TREATMENTS)] if has["medical_treatment"] else []
    factor = [rng.choice(config["factors"])] if has["liability_factors"] else []
    label = Extraction(incident_type=typ, incident_date=date, location=location,
                       injuries=injury, medical_treatment=treatment, liability_factors=factor,
                       missing_information=[k for k in MISSING_KEYS if not has[k]]).model_dump(mode="json")
    vals = {
        "date_text": date or "date not supplied",
        "loc_text": location or "location not provided",
        "event": event,
        "factor_text": ("Witness said " + factor[0] + ".") if factor else "No fault details were given.",
        "inj_text": ("Reported injury: " + injury[0] + ".") if injury else "No injuries described.",
        "treatment_text": ("Treatment: " + treatment[0] + ".") if treatment else "No treatment mentioned.",
        "distractor": rng.choice(DISTRACTORS) if rng.random() < 0.4 else "",
    }
    idx = rng.randrange(len(TEMPLATES[split]))
    narrative = TEMPLATES[split][idx].format(**vals).replace("  ", " ").strip()
    return {"id": f"{split}-{i:05d}", "family": f"{split}-{idx}", "narrative": narrative, "label": label}

def generate(root: Path, seed: int = 71, counts: dict | None = None) -> dict:
    counts = counts or {"train": 480, "validation": 90, "test": 90, "challenge": 60}
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    manifest = {"seed": seed, "counts": counts, "template_families": {k: len(v) for k,v in TEMPLATES.items()},
                "disclaimer": "Synthetic scenario-template data, not real legal records."}
    seen = set()
    for split, count in counts.items():
        if split not in TEMPLATES:
            raise ValueError(f"unknown split: {split}")
        records = []
        attempts = 0
        while len(records) < count:
            attempts += 1
            if attempts > count * 100:
                raise RuntimeError("unable to generate unique narratives")
            row = one(rng, split, len(records))
            if row["narrative"] in seen:
                continue
            seen.add(row["narrative"])
            records.append(row)
        with (root / f"{split}.jsonl").open("w") as f:
            for item in records:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest

def load(root: Path, split: str) -> list[dict]:
    return [json.loads(s) for s in (root / f"{split}.jsonl").read_text().splitlines() if s.strip()]
