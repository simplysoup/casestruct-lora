#!/usr/bin/env python3
"""Download and normalize independently sourced incident data, without inventing labels."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter


def dump_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')


def texts(x):
    if isinstance(x, str):
        return [x]
    if isinstance(x, list):
        return [v for t in x for v in texts(t)]
    if isinstance(x, dict):
        return texts(x.get('text', []))
    return []


def read_records(path: Path):
    """Accept JSONL/NDJSON, a JSON array, or a JSON object containing records."""
    with path.open(encoding='utf-8') as f:
        first = next((ch for line in f for ch in line if not ch.isspace()), None)
    if first is None:
        raise ValueError(f'Empty dataset: {path}')
    # ConstructCIE is NDJSON despite its .json filename.
    try:
        with path.open(encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        if 'Extra data' not in str(exc):
            raise
        with path.open(encoding='utf-8') as f:
            data = [json.loads(line) for line in f if line.strip()]
    if isinstance(data, dict):
        data = next((data[key] for key in ('data','train','examples','records') if isinstance(data.get(key), list)), data)
    if not isinstance(data, list):
        raise ValueError(f'Expected list of records in {path}, got {type(data).__name__}')
    return data


def constructcie(dest: Path):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id='lab-flair/constructcie', repo_type='dataset', filename='constructcie.json')
    raw = read_records(Path(path))
    out = []
    for rec in raw:
        narrative = ' '.join(texts(rec.get('accident_report', {}))).strip()
        if not narrative:
            continue
        def label(key):
            x = rec.get(key, {})
            return x.get('value', []) if isinstance(x, dict) else x
        out.append({
            'source':'ConstructCIE', 'source_id':str(rec.get('id')),
            'narrative':narrative, 'accident_type':label('accident_type'),
            'original_annotations':rec,  # Preserve the original span + causal tree labels
            'task':'causal_information_extraction',
        })
    if not out:
        raise ValueError('ConstructCIE yielded no narratives: check source fields before continuing')
    dump_jsonl(dest/'constructcie.jsonl', out)
    print(f'ConstructCIE: {len(out)} records (original annotations retained)')
    return len(out)


def osha(dest: Path, limit: int):
    from datasets import load_dataset
    ds = load_dataset('masonmarker/osha-severe-injury-reports-enriched', split='data', streaming=True)
    out=[]
    seen=set()
    for rec in ds:
        narrative = (rec.get('final_narrative') or '').strip()
        if len(narrative) < 80:
            continue
        key=hashlib.sha256(narrative.casefold().encode('utf-8')).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        # Incident date and location are external record metadata, NOT necessarily mentioned in the narrative.
        # Never grade narrative extraction against such metadata without evidence-span verification.
        out.append({
            'source':'OSHA Severe Injury Reports (enriched mirror)',
            'source_id':rec.get('incident_id'), 'narrative':narrative,
            'coded_labels':{
                k:rec.get(k) for k in (
                    'nature_title','part_of_body_title','event_title','source_title',
                    'has_hospitalization','has_amputation','has_eye_loss'
                )},
            'metadata_not_narrative_labels':{
                k:str(rec.get(k)) if rec.get(k) is not None else None
                for k in ('event_date','city','state')
            },
            'task':'occupational_injury_classification',
        })
        if len(out)>=limit:
            break
    if not out:
        raise ValueError('OSHA yielded no narratives: check source columns before continuing')
    dump_jsonl(dest/'osha_sir.jsonl',out)
    print(f'OSHA SIR: {len(out)} records (coded outcomes, no fabricated intake labels)')
    return len(out)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path('data/external'))
    p.add_argument('--osha-limit',type=int,default=3000)
    p.add_argument('--source',choices=('all','constructcie','osha'),default='all')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    counts={}
    if args.source in ('all','constructcie'):
        counts['constructcie']=constructcie(args.output)
    if args.source in ('all','osha'):
        counts['osha_sir']=osha(args.output,args.osha_limit)
    (args.output/'manifest.json').write_text(json.dumps({
        'counts':counts,
        'sources':{
          'constructcie':'https://huggingface.co/datasets/lab-flair/constructcie',
          'osha_sir':'https://huggingface.co/datasets/masonmarker/osha-severe-injury-reports-enriched',
        },
        'warning':'Datasets have different targets from CaseStruct synthetic intake schema. Do not score directly against unavailable gold fields.',
        'license_notes':'ConstructCIE Apache-2.0; OSHA mirror lists MIT; confirm upstream source terms and citation requirements.',
    },indent=2)+'\n')
    print('Saved',args.output/'manifest.json')

if __name__=='__main__':
    main()
