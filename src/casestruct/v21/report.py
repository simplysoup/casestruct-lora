"""Paired, split-locked uncertainty and full error analysis (10,000 resamples)."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from casestruct.v2.data import load_jsonl
from casestruct.v2.evaluate import metrics

SEED=71

def index_predictions(rows):
    out={}
    for r in rows:
        key=(r['task'],str(r['id']))
        if key in out:raise ValueError(f'duplicate test ID {key}')
        out[key]=r
    return out

def score_matrix(gold,pred,classes,indices):
    """Vectorized accuracy and fixed-vocabulary macro F1 for bootstrap samples."""
    g=gold[indices];p=pred[indices]
    acc=(g==p).mean(axis=1)
    f1s=[]
    for c in range(len(classes)):
        tp=((g==c)&(p==c)).sum(axis=1)
        fp=((g!=c)&(p==c)).sum(axis=1)
        fn=((g==c)&(p!=c)).sum(axis=1)
        denom=2*tp+fp+fn
        f1s.append(np.divide(2*tp,denom,out=np.zeros(len(indices)),where=denom!=0))
    return {'accuracy':acc,'macro_f1':np.mean(f1s,axis=0)}

def quantiles(values):
    return [float(v) for v in np.quantile(values,[.025,.975])]

def bootstrap(all_rows,classes,reps=10000,seed=SEED):
    anchor=index_predictions(next(iter(all_rows.values())))
    aligned={}
    for name,rows in all_rows.items():
        idx=index_predictions(rows)
        if idx.keys()!=anchor.keys():
            raise ValueError(f'{name}: baseline covers different test IDs')
        for key in anchor:
            if idx[key]['gold']!=anchor[key]['gold'] or idx[key]['narrative']!=anchor[key]['narrative']:
                raise ValueError(f'{name}: gold/narrative mismatch at {key}')
        aligned[name]=idx
    rng=np.random.default_rng(seed)
    results={}
    for task in sorted(classes):
        keys=sorted(k for k in anchor if k[0]==task)
        if not keys:raise ValueError(f'no test examples for {task}')
        vocab=classes[task];mapping={x:i for i,x in enumerate(vocab)}
        gold=np.array([mapping[anchor[k]['gold']] for k in keys],dtype=np.int32)
        draw=rng.integers(0,len(keys),size=(reps,len(keys)))
        systems={}
        for name,rows in aligned.items():
            pred=np.array([mapping.get(rows[k]['prediction'],-1) for k in keys],dtype=np.int32)
            scores=score_matrix(gold,pred,vocab,draw)
            point=score_matrix(gold,pred,vocab,np.arange(len(keys))[None,:])
            systems[name]={'point':{m:float(v[0]) for m,v in point.items()},
                           'ci95':{m:quantiles(v) for m,v in scores.items()},
                           'samples':scores}
        # Choose strongest baseline by macro F1, excluding v2 and v2's alternative methods.
        candidate_names=[n for n in systems if not n.startswith('v2-')]
        if not candidate_names:raise ValueError(f'{task}: no baseline exists')
        strongest=max(candidate_names,key=lambda n:systems[n]['point']['macro_f1'])
        headline='v2-constrained' if 'v2-constrained' in systems else 'v2-zero'
        comparison={}
        for name in systems:
            if name==headline:continue
            comparison[name]={metric:{'delta':systems[headline]['point'][metric]-systems[name]['point'][metric],
                    'ci95':quantiles(systems[headline]['samples'][metric]-systems[name]['samples'][metric])}
                    for metric in ('accuracy','macro_f1')}
        results[task]={'n':len(keys),'headline_model':headline,'strongest_baseline':strongest,
                       'systems':{n:{k:v for k,v in s.items() if k!='samples'} for n,s in systems.items()},
                       'paired_delta':comparison}
    return results

def confusion(rows, classes):
    result={}
    for task,vocab in classes.items():
        rs=[r for r in rows if r['task']==task]
        columns=vocab+['<INVALID>'];counts=Counter((r['gold'],r.get('prediction') if r.get('prediction') in vocab else '<INVALID>') for r in rs)
        result[task]={'labels':columns,'counts':{g:{p:counts[g,p] for p in columns} for g in vocab}}
    return result

def review_errors(rows,seed=SEED,n=15):
    rng=np.random.default_rng(seed)
    selected=[]
    for task in sorted({r['task'] for r in rows}):
        errors=[r for r in rows if r['gold']!=r['prediction']]
        errors=[r for r in errors if r['task']==task]
        by_class=defaultdict(list)
        for r in errors:by_class[r['gold']].append(r)
        for label in by_class:by_class[label]=list(rng.permutation(by_class[label]))
        while len([r for r in selected if r['task']==task])<min(n,len(errors)):
            updated=False
            for label in sorted(by_class):
                if by_class[label] and len([r for r in selected if r['task']==task])<n:
                    selected.append(by_class[label].pop());updated=True
            if not updated:break
    return selected

def run(root='reports/v2.1',data='data/v2',reps=10000):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    classes=json.loads((Path(data)/'manifest.json').read_text())['class_names']
    paths={f'{m}-{method}':root/f'{m}-{method}-test.predictions.jsonl' for m in ('base','v1','v2') for method in ('zero','fewshot','constrained')}
    missing=[str(p) for p in paths.values() if not p.exists()]
    if missing:raise FileNotFoundError('Missing full test predictions: '+', '.join(missing))
    rows={name:list(load_jsonl(path)) for name,path in paths.items()}
    result=bootstrap(rows,classes,reps)
    for name,rs in rows.items():
        (root/f'{name}.confusion.json').write_text(json.dumps(confusion(rs,classes),indent=2)+'\n')
        # Fixed vocabulary precision/recall/F1 including zero-support classes.
        by_task={}
        for task,vocab in classes.items():
            sub=[r for r in rs if r['task']==task];n=len(sub)
            per={}
            for lab in vocab:
                tp=sum(r['gold']==lab and r['prediction']==lab for r in sub)
                fp=sum(r['gold']!=lab and r['prediction']==lab for r in sub)
                fn=sum(r['gold']==lab and r['prediction']!=lab for r in sub)
                per[lab]={'support':tp+fn,'precision':tp/(tp+fp) if tp+fp else 0,
                          'recall':tp/(tp+fn) if tp+fn else 0,
                          'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0}
            by_task[task]=per
        (root/f'{name}.per_class.json').write_text(json.dumps(by_task,indent=2)+'\n')
    (root/'statistics.json').write_text(json.dumps(result,indent=2)+'\n')
    with (root/'errors_to_review.csv').open('w',newline='',encoding='utf-8') as fh:
        writer=csv.DictWriter(fh,fieldnames=['system','task','id','gold','prediction','narrative','proposed_category','owner_confirmed_category','notes'])
        writer.writeheader()
        for name,rs in rows.items():
            if not name.startswith('v2-'):continue
            for r in review_errors(rs):
                writer.writerow({'system':name,'task':r['task'],'id':r['id'],'gold':r['gold'],
                                 'prediction':r['prediction'],'narrative':r['narrative'],
                                 'proposed_category':'needs human review: genuine error / weak label evidence / ambiguous',
                                 'owner_confirmed_category':'','notes':''})
    lines=['# CaseStruct v2.1 — exploratory external benchmark','',
           'Data: real construction/OSHA incident descriptions; **not legal case records**. Results do not establish deployment, legal, or safety suitability.',
           'The v2 test set was inspected during earlier development and cannot be called pristine or newly preregistered.',
           'All nine methods use the identical saved test examples and label vocabulary; constrained methods do not generate JSON.',
           'Bootstrap: 10,000 paired resamples, seed 71. Each task is reported separately.','','## Results (point estimate [95% paired-bootstrap CI])','',
           '| Task | System | N | Accuracy | Macro F1 | JSON validity |','|---|---|---:|---:|---:|---:|']
    for task,t in result.items():
        for name,s in t['systems'].items():
            def fmt(metric):
                a=s['point'][metric];lo,hi=s['ci95'][metric]
                return f'{a:.1%} [{lo:.1%}, {hi:.1%}]'
            # JSON validity only for actual generation; load from saved flags, not forced choices.
            sub=[r for r in rows[name] if r['task']==task]
            json_rate='N/A (forced choice)' if name.endswith('constrained') else f"{sum(bool(r['json_valid']) for r in sub)/len(sub):.1%}"
            lines.append(f"| {task} | {name} | {t['n']} | {fmt('accuracy')} | {fmt('macro_f1')} | {json_rate} |")
        base=t['strongest_baseline'];head=t['headline_model'];delta=t['paired_delta'][base]
        lines.extend(['',f"### {task} — headline vs strongest baseline",'',
            f"Strongest non-v2 baseline (macro F1 criterion): **{base}**. Headline v2 method (predefined): **{head}**.",
            f"Accuracy delta: **{delta['accuracy']['delta']:+.1%}** [95% CI {delta['accuracy']['ci95'][0]:+.1%}, {delta['accuracy']['ci95'][1]:+.1%}].",
            f"Macro F1 delta: **{delta['macro_f1']['delta']:+.1%}** [95% CI {delta['macro_f1']['ci95'][0]:+.1%}, {delta['macro_f1']['ci95'][1]:+.1%}].",
            'ConstructCIE has only 80 test cases.' if task=='constructcie' else 'OSHA codes can depend on non-narrative administrative information.',''])
    lines.extend(['## Audit and errors','',
       '- See `audit.md` and `audit.json`. Historical checkpoint provenance and actual collated token masking require independent verification.',
       '- See all `*.confusion.json` and `*.per_class.json` files.',
       '- See `errors_to_review.csv` for balanced, *unreviewed* errors; classifications are proposals, not adjudicated conclusions.',
       '- This is an exploratory holdout because v2 test metrics were viewed before v2.1.',
       '- Do not publish source narratives without checking source redistribution rights and privacy considerations.',
       '','## Reproduction','','`bash run_v21.sh` executes the audit, all 9 baselines, and this report without retraining.'])
    (root/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(root/'summary.md')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--reps',type=int,default=10000)
    parser.add_argument('--root',default='reports/v2.1');args=parser.parse_args()
    run(root=args.root,reps=args.reps)
