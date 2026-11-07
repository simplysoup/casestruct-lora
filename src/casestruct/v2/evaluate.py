"""Per-task evaluation, preserving every prediction for error analysis."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time
from .data import load_jsonl, write_jsonl
from .model import MODEL_ID, load_model,generate,extract_label


def metrics(rows):
    out={}
    for task in sorted({r['task'] for r in rows}):
        rs=[r for r in rows if r['task']==task]
        n=len(rs); labels=sorted({r['gold'] for r in rs})
        stats={}
        for label in labels:
            tp=sum(r['prediction']==r['gold']==label for r in rs)
            fp=sum(r['prediction']==label and r['gold']!=label for r in rs)
            fn=sum(r['gold']==label and r['prediction']!=label for r in rs)
            stats[label]={'support':sum(r['gold']==label for r in rs),'precision':tp/(tp+fp) if tp+fp else 0,'recall':tp/(tp+fn) if tp+fn else 0,'f1':2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0}
        out[task]={'count':n,'accuracy':sum(r['correct'] for r in rs)/n,'macro_f1':sum(s['f1'] for s in stats.values())/len(stats),'strict_json_rate':sum(r['json_valid'] for r in rs)/n,'allowed_label_rate':sum(r['allowed_label'] for r in rs)/n,'mean_latency_s':sum(r['latency_s'] for r in rs)/n,'per_label':stats}
    return out


def evaluate(data='data/v2',split='test',mode='base',adapter=None,limit=0,model_id=MODEL_ID):
    root=Path(data); manifest=json.loads((root/'manifest.json').read_text())
    rows=list(load_jsonl(root/f'{split}.jsonl'))
    if limit:
        rng=random.Random(3001)
        sample=[]
        for task in manifest['class_names']:
            selected=[r for r in rows if r['task']==task]
            rng.shuffle(selected)
            sample.extend(selected[:limit])
        rows=sample
    if not rows: raise RuntimeError('No evaluation examples')
    if mode!='base' and not adapter: raise ValueError('adapter path required for adapter mode')
    tok,model=load_model(model_id,adapter if mode!='base' else None)
    results=[]
    for i,r in enumerate(rows,1):
        start=time.perf_counter()
        raw=generate(tok,model,r,manifest['class_names'])
        latency=time.perf_counter()-start
        label,valid=extract_label(raw)
        allowed=label in manifest['class_names'][r['task']] if valid else False
        results.append({'task':r['task'],'id':r['id'],'narrative':r['narrative'],'gold':r['label'],'prediction':label,'raw':raw,'json_valid':valid,'allowed_label':allowed,'correct':bool(allowed and label==r['label']),'latency_s':round(latency,4)})
        if i % 25 == 0 or i == len(rows): print(f'{mode} {split}: {i}/{len(rows)}',flush=True)
    suffix=f'{split}' + (f'-sample{limit}' if limit else '')
    stem=f'{mode}-{suffix}'
    out=Path('reports/v2'); out.mkdir(parents=True,exist_ok=True)
    write_jsonl(out/f'{stem}.predictions.jsonl',results)
    stats=metrics(results)
    (out/f'{stem}.metrics.json').write_text(json.dumps({'mode':mode,'split':split,'model_id':model_id,'adapter':str(adapter) if adapter else None,'metrics':stats},indent=2)+'\n')
    print(json.dumps(stats,indent=2))
    return stats


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/v2');p.add_argument('--split',choices=['validation','test'],default='test');p.add_argument('--mode',choices=['base','v1','v2'],default='base');p.add_argument('--adapter');p.add_argument('--limit',type=int,default=0);p.add_argument('--model-id',default=MODEL_ID)
    a=p.parse_args();evaluate(a.data,a.split,a.mode,a.adapter,a.limit,a.model_id)

if __name__=='__main__':main()
