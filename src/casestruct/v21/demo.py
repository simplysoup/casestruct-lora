"""Inference CLI and reproducible single-request local performance probe."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import time
import torch
from casestruct.v2.model import load_model,extract_label
from .baselines import prefix


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--text',help='Incident narrative (fictional examples recommended)')
    p.add_argument('--task',choices=['constructcie','osha'],default='constructcie')
    p.add_argument('--adapter',default='outputs/v2/adapter')
    p.add_argument('--runs',type=int,default=1,help='50+ for benchmarking')
    p.add_argument('--output',default='reports/v2.1/demo_benchmark.json')
    args=p.parse_args()
    if not args.text: p.error('Pass --text with an incident narrative')
    classes=json.loads(Path('data/v2/manifest.json').read_text())['class_names']
    tok,model=load_model(adapter=args.adapter)
    row={'task':args.task,'narrative':args.text,'id':'demo','label':None}
    x=tok(prefix(tok,row,classes),return_tensors='pt').to(next(model.parameters()).device)
    results=[]
    for i in range(args.runs):
        if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        start=time.perf_counter()
        with torch.inference_mode():
            out=model.generate(**x,do_sample=False,max_new_tokens=72,pad_token_id=tok.pad_token_id)
        if torch.cuda.is_available():torch.cuda.synchronize()
        sec=time.perf_counter()-start
        nt=int(out.shape[1]-x['input_ids'].shape[1])
        text=tok.decode(out[0,x['input_ids'].shape[1]:],skip_special_tokens=True).strip()
        label,valid=extract_label(text)
        results.append({'elapsed_s':sec,'tokens':nt,'tokens_per_s':nt/sec if sec else None,
                        'peak_gpu_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
                        'prediction':label,'json_valid':valid,'raw':text})
    print(json.dumps({'task':args.task,'label':results[-1]['prediction'],'raw':results[-1]['raw']},indent=2))
    if args.runs>1:
        # exclude initial warm-up from latency distribution
        measured=results[1:];times=sorted(r['elapsed_s'] for r in measured)
        def pct(p):return times[min(len(times)-1,int((len(times)-1)*p))]
        report={'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
                'p50_latency_s':pct(.5),'p95_latency_s':pct(.95),
                'mean_tokens_per_s':statistics.mean(r['tokens_per_s'] for r in measured),
                'peak_gpu_bytes':max((r['peak_gpu_bytes'] or 0) for r in measured),
                'runs_total':len(results),'warmup_excluded':1}
        dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))

if __name__=='__main__':main()
