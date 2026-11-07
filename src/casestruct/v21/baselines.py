"""Pre-registered train-only demonstrations and conditional label likelihoods."""
from __future__ import annotations
from collections import defaultdict
import json
import random
import torch
from casestruct.v2.prompts import prompt, gold

SEED = 71
SHOTS = 4

def few_shot_examples(training_rows, classes, k=SHOTS, seed=SEED):
    rng = random.Random(seed)
    by_task = defaultdict(list)
    for r in training_rows:
        if r['label'] in classes[r['task']]:
            by_task[r['task']].append(r)
    result = {}
    for task in sorted(classes):
        buckets = defaultdict(list)
        for row in by_task[task]:
            buckets[row['label']].append(row)
        for label in sorted(buckets):
            buckets[label].sort(key=lambda r: (str(r['id']), r['narrative']))
            rng.shuffle(buckets[label])
        selected = []
        labels = sorted(buckets)
        while len(selected) < min(k, len(by_task[task])):
            for label in labels:
                if buckets[label] and len(selected) < k:
                    selected.append(buckets[label].pop())
        result[task] = selected
    return result

def prompt_messages(row, classes, examples=None):
    messages = []
    for ex in (examples or {}).get(row['task'], []):
        messages.extend([{'role':'user','content':prompt(ex,classes)},
                         {'role':'assistant','content':gold(ex)}])
    messages.append({'role':'user','content':prompt(row,classes)})
    return messages

def prefix(tok,row,classes,examples=None):
    return tok.apply_chat_template(prompt_messages(row,classes,examples), tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)

def score_candidates(model,tok,prefix_text,labels,max_input_tokens=2048):
    """Score conditional complete JSON strings; total log-probability is PRIMARY.

    Also persist token-length normalized values as SECONDARY. Every candidate
    shares the identical prompt, and the same unnormalized scoring rule is used
    for each model. Includes the chat end token in the candidate.
    """
    device=next(model.parameters()).device
    prefix_ids=tok(prefix_text,add_special_tokens=False)['input_ids']
    if not prefix_ids or len(prefix_ids)>max_input_tokens:
        raise ValueError(f'Invalid/overlong prompt: {len(prefix_ids)} tokens')
    out={}
    with torch.inference_mode():
        for label in labels:
            completion=json.dumps({'label':label},ensure_ascii=False)+'<|im_end|>'
            completion_ids=tok(completion,add_special_tokens=False)['input_ids']
            ids=torch.tensor([prefix_ids+completion_ids],device=device)
            logits=model(input_ids=ids,attention_mask=torch.ones_like(ids)).logits[0]
            start=len(prefix_ids)-1
            logp=torch.log_softmax(logits[start:start+len(completion_ids)].float(),dim=-1)
            selected=logp.gather(1,torch.tensor(completion_ids,device=device).unsqueeze(1)).squeeze(1)
            out[label]={'total':float(selected.sum().item()), 'mean':float(selected.mean().item()),
                        'tokens':len(completion_ids)}
    return out

def predict_constrained(model,tok,row,classes,examples=None):
    scores=score_candidates(model,tok,prefix(tok,row,classes,examples),classes[row['task']])
    primary=max(scores,key=lambda label:scores[label]['total'])
    secondary=max(scores,key=lambda label:scores[label]['mean'])
    return primary, secondary, scores
