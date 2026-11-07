"""Generate comparison strictly from persisted metrics."""
import json
from pathlib import Path


def main():
    root=Path('reports/v2'); rows=[]
    for path in sorted(root.glob('*.metrics.json')):
        data=json.loads(path.read_text())
        for task,m in data['metrics'].items():
            rows.append([path.stem.replace('.metrics',''),task,str(m['count']),f"{m['accuracy']:.1%}",f"{m['macro_f1']:.1%}",f"{m['strict_json_rate']:.1%}"])
    lines=['# CaseStruct v2: External Data Evaluation','','| Run | Task | N | Accuracy | Macro-F1 | Strict JSON |','|---|---|---:|---:|---:|---:|']
    lines += ['| '+' | '.join(r)+' |' for r in rows]
    lines += ['','## Limitations','','ConstructCIE accident types and OSHA injury-nature codes are **different classification tasks**. Do not average their scores.','OSHA label codes are administrative annotations, sometimes not inferable from the narrative; errors can be label-information mismatches.','No real-world reliability or legal/safety fitness is established. Initial source records are observational and may include bias.','JSON exactness and classification correctness are distinct metrics.','V1 adapter is measured on the new task without retraining; it is a transfer baseline, not a matched fine-tune.','','Detailed per-example predictions: `reports/v2/*.predictions.jsonl`.']
    root.mkdir(parents=True,exist_ok=True)
    (root/'summary.md').write_text('\n'.join(lines)+'\n')
    print(root/'summary.md')

if __name__=='__main__':main()
