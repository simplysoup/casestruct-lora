"""All nine model/method conditions evaluated on the frozen v2 test split."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import torch
from casestruct.v2.data import load_jsonl,write_jsonl
from casestruct.v2.evaluate import metrics
from casestruct.v2.model import MODEL_ID,load_model,extract_label
from .baselines import few_shot_examples,prefix,predict_constrained,SEED,SHOTS

METHODS=('zero','fewshot','constrained')
ADAPTERS={'base':None,'v1':'outputs/local/adapter','v2':'outputs/v2/adapter'}

def evaluate(method='fewshot',model_name='base',data='data/v2',output='reports/v2.1',
             shots=SHOTS,limit=0,model_id=MODEL_ID,adapter=None):
    if method not in METHODS or model_name not in ADAPTERS:
        raise ValueError('Unsupported method or model')
    root=Path(data);dest=Path(output);dest.mkdir(parents=True,exist_ok=True)
    classes=json.loads((root/'manifest.json').read_text())['class_names']
    train=list(load_jsonl(root/'train.jsonl'))
    test=list(load_jsonl(root/'test.jsonl'))
    demos=few_shot_examples(train,classes,k=shots,seed=SEED)
    if limit:
        rng=random.Random(3001)
        sampled=[]
        for task in sorted(classes):
            pool=[r for r in test if r['task']==task]
            rng.shuffle(pool);sampled.extend(pool[:limit])
        test=sampled
    adapter_path=adapter if adapter is not None else ADAPTERS[model_name]
    if adapter_path and not Path(adapter_path).exists():
        raise FileNotFoundError(f'Missing {model_name} adapter: {adapter_path}')
    tok,model=load_model(model_id,adapter_path)
    results=[]
    for i,row in enumerate(test,1):
        begin=time.perf_counter()
        selected_demos=demos if method=='fewshot' else None
        if method=='constrained':
            label,mean_label,scores=predict_constrained(model,tok,row,classes)
            raw=None
            valid=None  # NOT a generated JSON validity result
            allowed=label in classes[row['task']]
        else:
            text=prefix(tok,row,classes,selected_demos)
            tokens=tok(text,return_tensors='pt').to(next(model.parameters()).device)
            if tokens['input_ids'].shape[1]>2048:
                raise ValueError('Few-shot context too long: reduce k before test evaluation')
            with torch.inference_mode():
                seq=model.generate(**tokens,do_sample=False,max_new_tokens=72,
                                   pad_token_id=tok.pad_token_id)
            raw=tok.decode(seq[0,tokens['input_ids'].shape[1]:],skip_special_tokens=True).strip()
            label,valid=extract_label(raw)
            mean_label=None;scores=None
            allowed=bool(valid and label in classes[row['task']])
        results.append({'task':row['task'],'id':row['id'],'narrative':row['narrative'],
                        'gold':row['label'],'prediction':label,'raw':raw,
                        'json_valid':valid,'allowed_label':allowed,'correct':bool(allowed and label==row['label']),
                        'latency_s':round(time.perf_counter()-begin,5),'secondary_length_normalized_prediction':mean_label,
                        'scores':scores})
        if i%25==0 or i==len(test):print(f'{model_name}/{method}: {i}/{len(test)}',flush=True)
    suffix=f'test-sample{limit}' if limit else 'test'
    stem=f'{model_name}-{method}-{suffix}'
    write_jsonl(dest/f'{stem}.predictions.jsonl',results)
    stats=metrics([{**r,'json_valid':bool(r['json_valid'])} for r in results])
    if method=='constrained':
        for item in stats.values():item['strict_json_rate']=None
    metadata={'model':model_name,'method':method,'adapter':adapter_path,'seed':SEED,
              'fewshot_k':shots if method=='fewshot' else 0,
              'demonstration_ids':{task:[str(r['id']) for r in rows] for task,rows in demos.items()} if method=='fewshot' else {},
              'test_sha256':hashlib.sha256((root/'test.jsonl').read_bytes()).hexdigest(),
              'scoring_primary':'total conditional log-likelihood of complete JSON candidate' if method=='constrained' else 'greedy generation',
              'scoring_secondary':'mean token conditional log-likelihood' if method=='constrained' else None,
              'metrics':stats}
    (dest/f'{stem}.metrics.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(stats,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--method',choices=METHODS,default='fewshot')
    p.add_argument('--model',choices=sorted(ADAPTERS),default='base')
    p.add_argument('--data',default='data/v2');p.add_argument('--output',default='reports/v2.1')
    p.add_argument('--shots',type=int,default=SHOTS);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--adapter');a=p.parse_args()
    evaluate(a.method,a.model,a.data,a.output,a.shots,a.limit,adapter=a.adapter)
