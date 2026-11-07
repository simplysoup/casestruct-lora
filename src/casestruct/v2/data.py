"""Prepare reproducible, grouped and label-stratified external evaluation datasets.

Important: OSHA codes are administrative labels; some cannot be inferred from a narrative.
Keep each task separate, and audit errors accordingly.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

TASKS = {'constructcie': 'accident_type', 'osha': 'injury_nature'}


def load_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    obj = json.loads(line)
                except ValueError as e:
                    raise ValueError(f'{path}:{n}: {e}') from e
                yield obj


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def key_for(text):
    """Normalize exact/near-exact trivial variations to keep repeats together."""
    return hashlib.sha256(re.sub(r'\W+', ' ', text.casefold()).strip().encode()).hexdigest()


def clean_label(value):
    if isinstance(value, list):
        return str(value[0]).strip() if len(value) == 1 else None
    return str(value).strip() if value is not None else None


def parse_constructcie(rows):
    out = []
    for r in rows:
        label = clean_label(r.get('accident_type'))
        narrative = r.get('narrative')
        if not isinstance(narrative, str) or len(narrative.strip()) < 50 or not label:
            continue
        out.append({'task': 'constructcie', 'id': str(r.get('source_id','')), 'narrative': narrative.strip(), 'label': label})
    return out


def parse_osha(rows, max_classes=8, min_count=25):
    candidates = []
    for r in rows:
        narrative = r.get('narrative')
        labels = r.get('coded_labels') or {}
        label = clean_label(labels.get('nature_title'))
        if isinstance(narrative, str) and len(narrative.strip()) >= 80 and label:
            candidates.append({'task': 'osha', 'id': str(r.get('source_id','')), 'narrative': narrative.strip(), 'label': label})
    popular = [k for k,n in Counter(r['label'] for r in candidates).most_common(max_classes) if n >= min_count]
    return [r for r in candidates if r['label'] in set(popular)]


def grouped_split(rows, seed=71):
    """Group duplicate normalized narratives; stratify separately by task and label."""
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row['task'], row['label'], key_for(row['narrative']))].append(row)
    by_label = defaultdict(list)
    for (task,label,_), group in buckets.items():
        by_label[(task,label)].append(group)
    splits = {'train': [], 'validation': [], 'test': []}
    for groups in by_label.values():
        rng.shuffle(groups)
        n = len(groups)
        if n < 5:
            raise ValueError('Too few independent narrative groups in a class; cannot split safely')
        n_test = max(1, round(n * .15))
        n_val = max(1, round(n * .15))
        for split, selected in (('test',groups[:n_test]), ('validation',groups[n_test:n_test+n_val]), ('train',groups[n_test+n_val:])):
            for group in selected:
                splits[split].extend(group)
    for split in splits:
        rng.shuffle(splits[split])
    # No text or exact normalized duplicate may cross splits, even across tasks.
    ownership = {}
    for split in ['test','validation','train']:
        kept = []
        for r in splits[split]:
            k = key_for(r['narrative'])
            if k not in ownership:
                ownership[k] = split
            if ownership[k] == split:
                kept.append(r)
        splits[split] = kept
    return splits


def prepare(root='data/external', output='data/v2', seed=71, osha_classes=8):
    root, output = Path(root), Path(output)
    c, o = root/'constructcie.jsonl', root/'osha_sir.jsonl'
    for p in (c,o):
        if not p.exists():
            raise FileNotFoundError(f'Missing {p}; run bash run.sh fetch first')
    rows = parse_constructcie(load_jsonl(c)) + parse_osha(load_jsonl(o), osha_classes)
    splits = grouped_split(rows, seed)
    classes = {task: sorted({r['label'] for r in rows if r['task'] == task}) for task in TASKS}
    # Verify enough examples of every class remain in each split.
    for task, labels in classes.items():
        for split in splits:
            counts = Counter(r['label'] for r in splits[split] if r['task']==task)
            if any(counts[x] == 0 for x in labels):
                raise ValueError(f'Missing class in {split}/{task}; add more source data')
    output.mkdir(parents=True,exist_ok=True)
    for split, subset in splits.items():
        write_jsonl(output/f'{split}.jsonl',subset)
    sources = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (c,o)}
    manifest = {'seed':seed, 'class_names':classes, 'counts':{s:dict(Counter(r['task'] for r in rs)) for s,rs in splits.items()}, 'source_sha256':sources, 'limitations':'OSHA administrative coding may be impossible to reconstruct solely from narratives; task-specific evaluation only. Split by normalized narrative groups.'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',default='data/external'); p.add_argument('--output',default='data/v2'); p.add_argument('--seed',type=int,default=71); p.add_argument('--osha-classes',type=int,default=8)
    a=p.parse_args(); print(json.dumps(prepare(a.root,a.output,a.seed,a.osha_classes),indent=2))

if __name__=='__main__': main()
