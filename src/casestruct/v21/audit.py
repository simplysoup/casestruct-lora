"""Reproducibility, split isolation, and training objective audit."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
from casestruct.v2.data import load_jsonl, key_for


def normalize(text):
    return re.sub(r"\W+", " ", text.casefold()).strip()


def shingles(text, k=4):
    words=normalize(text).split()
    if len(words)<k: return set(words)
    return {tuple(words[i:i+k]) for i in range(len(words)-k+1)}


def candidate_pairs(splits, min_shared=3):
    """Retrieve likely near-duplicates using a 4-token shingle inverted index.

    Also checks exact normalized duplicates independently. Unmatched paraphrases are
    not guaranteed to be detected; this is an audit, not a proof of isolation.
    """
    entries=[(split,row) for split,rs in splits.items() for row in rs]
    index=defaultdict(list)
    found=set()
    for i,(split,row) in enumerate(entries):
        sg=shingles(row['narrative'])
        common=Counter()
        for sh in sg:
            for j in index[sh]:
                if entries[j][0] != split:
                    common[j]+=1
        for j,c in common.items():
            if c>=min_shared:
                found.add((j,i))
        for sh in sg:
            index[sh].append(i)
    return entries, found


def cross_split_audit(splits, threshold=.80):
    entries,pairs=candidate_pairs(splits)
    exact=defaultdict(set)
    for split, rows in splits.items():
        for row in rows:
            exact[key_for(row['narrative'])].add(split)
    exact_collisions=sum(len(v)>1 for v in exact.values())
    flagged=[]
    for i,j in pairs:
        sa,a=entries[i]; sb,b=entries[j]
        aa=normalize(a['narrative']); bb=normalize(b['narrative'])
        # String similarity is intentionally conservative: flag high similarity,
        # not merely common workplace vocabulary.
        aset,bset=shingles(aa),shingles(bb)
        shared=len(aset & bset)
        jac=shared/max(1,len(aset | bset))
        # Fast filters avoid expensive quadratic text similarity across shared
        # boilerplate. A genuinely high-edit-similarity long narrative normally
        # shares many 4-grams, but shorter paraphrases may still be missed.
        ratio=0.0
        if (min(len(aa),len(bb))/max(1,max(len(aa),len(bb))) >= threshold
                and shared >= max(3, int(.35 * min(len(aset),len(bset))))):
            ratio=SequenceMatcher(None,aa,bb,autojunk=False).ratio()
        if ratio>=threshold or jac>=threshold:
            flagged.append({'a_split':sa,'b_split':sb,'a_id':a.get('id'),
                'b_id':b.get('id'),'a_task':a['task'],'b_task':b['task'],
                'string_similarity':round(ratio,4),'shingle_jaccard':round(jac,4),
                'a_excerpt':a['narrative'][:250], 'b_excerpt':b['narrative'][:250]})
    flagged.sort(key=lambda x:max(x['string_similarity'],x['shingle_jaccard']), reverse=True)
    return {'exact_cross_split_duplicate_keys':exact_collisions,
            'near_duplicate_flag_count':len(flagged),'near_duplicate_flags':flagged,
            'threshold':threshold,'candidate_pairs':len(pairs),
            'coverage_note':'4-token-shingle candidate retrieval can miss paraphrases. Review flagged examples manually.'}


def training_audit(repo):
    path=repo/'src/casestruct/v2/train.py'
    source=path.read_text()
    # Static source inspection, not proof that a specific historical run used this code.
    checks={'completion_only_loss_enabled':bool(re.search(r'completion_only_loss\s*=\s*True',source)),
            'explicit_prompt_completion_columns':bool(re.search(r"['\"]prompt['\"]\s*:",source) and re.search(r"['\"]completion['\"]\s*:",source)),
            'training_reads_test_split':bool(re.search(r"convert\(['\"]test['\"]\)",source)),
            'validation_read_for_training_evaluation':bool(re.search(r"convert\(['\"]validation['\"]\)",source))}
    checks['source_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    checks['note']='Static audit of checked-in training code; verify W&B run configuration/commit and collator token labels before claiming run-level masking.'
    return checks


def run(data='data/v2', out='reports/v2.1', repo='.', threshold=.80):
    data=Path(data); out=Path(out); repo=Path(repo)
    manifest=json.loads((data/'manifest.json').read_text())
    splits={s:list(load_jsonl(data/f'{s}.jsonl')) for s in ('train','validation','test')}
    checks=cross_split_audit(splits,threshold)
    checks['counts']={s:dict(Counter(r['task'] for r in rows)) for s,rows in splits.items()}
    checks['classes']={task:sorted({r['label'] for rows in splits.values() for r in rows if r['task']==task}) for task in manifest['class_names']}
    checks['manifest_classes_match']=checks['classes']==manifest['class_names']
    checks['manifest_count_match']=checks['counts']==manifest['counts']
    checks['train_source']=training_audit(repo)
    checks['test_split_sha256']=hashlib.sha256((data/'test.jsonl').read_bytes()).hexdigest()
    checks['manifest_sha256']=hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest()
    out.mkdir(parents=True,exist_ok=True)
    (out/'audit.json').write_text(json.dumps(checks,indent=2)+'\n')
    write_markdown(checks,out/'audit.md')
    print(json.dumps({k:v for k,v in checks.items() if k!='near_duplicate_flags'},indent=2))
    if any(max(x['string_similarity'],x['shingle_jaccard'])>=.95 for x in checks['near_duplicate_flags']):
        raise RuntimeError('Potential cross-split leakage >=0.95; STOP and review audit.md before evaluation')
    if checks['exact_cross_split_duplicate_keys']:
        raise RuntimeError('Exact cross-split duplicates found. STOP: inspect audit.json')
    if not checks['manifest_classes_match'] or not checks['manifest_count_match']:
        raise RuntimeError('Manifest does not match prepared splits. STOP: inspect audit.json')
    if not checks['train_source']['completion_only_loss_enabled'] or checks['train_source']['training_reads_test_split']:
        raise RuntimeError('Training source audit failed. STOP: inspect audit.json')
    return checks


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data/v2');p.add_argument('--out',default='reports/v2.1');p.add_argument('--threshold',type=float,default=.80)
    a=p.parse_args();run(a.data,a.out,threshold=a.threshold)

def write_markdown(audit, output='reports/v2.1/audit.md'):
    """Evidence-first audit; unknown checks are explicitly NOT VERIFIED."""
    p=Path(output);p.parent.mkdir(parents=True,exist_ok=True)
    near=audit.get('near_duplicate_flags',[])
    lines=['# CaseStruct v2.1 — data and training audit','','| Check | Status | Evidence |','|---|---|---|',
      f"| Exact normalized cross-split matches | {'FAIL' if audit['exact_cross_split_duplicate_keys'] else 'PASS'} | {audit['exact_cross_split_duplicate_keys']} groups |",
      f"| Near duplicates at ≥0.80 | {'REVIEW' if near else 'PASS (candidate method)'} | {sum(max(x['string_similarity'],x['shingle_jaccard'])>=.8 for x in near)} candidate pairs |",
      f"| Near duplicates at ≥0.90 | {'REVIEW' if any(max(x['string_similarity'],x['shingle_jaccard'])>=.9 for x in near) else 'PASS (candidate method)'} | {sum(max(x['string_similarity'],x['shingle_jaccard'])>=.9 for x in near)} candidate pairs |",
      f"| Near duplicates at ≥0.95 | {'REVIEW' if any(max(x['string_similarity'],x['shingle_jaccard'])>=.95 for x in near) else 'PASS (candidate method)'} | {sum(max(x['string_similarity'],x['shingle_jaccard'])>=.95 for x in near)} candidate pairs |",
      f"| Class vocabulary and counts match manifest | {'PASS' if audit['manifest_classes_match'] and audit['manifest_count_match'] else 'FAIL'} | Prepared split manifests |",
      f"| Checked-in training source uses completion_only_loss | {'PASS (source)' if audit['train_source']['completion_only_loss_enabled'] else 'FAIL'} | sha256 `{audit['train_source']['source_sha256']}` |",
      '| Historical actual-token -100 masking | NOT VERIFIED | Static configuration alone does not establish runtime labels. Run a TRL collator probe and compare a saved train batch. |',
      '| Historical checkpoint/prompt-selection independence | NOT VERIFIED | Requires original W&B history and an owner-signed test-use chronology. Prior v2 test results were examined. |',
      '| Historical training loss explanation | REVIEW | Short completion classification, schema boilerplate, and high token accuracy are plausible explanations; runtime supervision needs inspection. |',
      '| Same test identities/labels across baseline methods | ENFORCED BY REPORT | Report refuses to compare mismatched IDs, narratives or gold labels. |',
      '| Unified JSON parser | PASS (new runs) | `casestruct.v2.model.extract_label` is used for all new generation methods. Constrained scoring produces no JSON. |',
      '', '## Highest-similarity cross-split candidate pairs', '',
      'Candidate retrieval is a 4-gram filter; it can miss paraphrases. Similarity thresholds are diagnostic, not a guarantee.','']
    for x in near[:25]:
        lines.extend([f"- **{x['a_split']}:{x['a_id']} ↔ {x['b_split']}:{x['b_id']}**: string={x['string_similarity']:.3f}, shingle={x['shingle_jaccard']:.3f}",
                      f"  - A: {x['a_excerpt']}", f"  - B: {x['b_excerpt']}"])
    if not near:lines.append('No cross-split candidates above the configured retrieval threshold.')
    lines.extend(['','## Publication status','',
      '**Not certified** until near-duplicate candidates, historical masking, and W&B checkpoint provenance are reviewed.',
      'Do not infer no leakage from absence of a shingle candidate.'])
    p.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return p


if __name__ == "__main__":
    main()
