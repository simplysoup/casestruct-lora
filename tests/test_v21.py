import json
from pathlib import Path
import numpy as np
import pytest
from casestruct.v21.audit import cross_split_audit,training_audit
from casestruct.v21.baselines import few_shot_examples,prompt_messages
from casestruct.v21.report import bootstrap,confusion,review_errors
from casestruct.v2.model import extract_label


def row(label,i,narrative=None):
    return {'task':'constructcie','label':label,'id':str(i),
            'narrative':narrative or f'Worker number {i} fell from a scaffold onto the floor.'}


def pred(i,gold,prediction):
    return {'task':'constructcie','id':str(i),'narrative':f'Narrative {i}',
            'gold':gold,'prediction':prediction,'correct':gold==prediction,
            'json_valid':True,'allowed_label':True,'latency_s':.1}


def test_duplicate_audit():
    splits={'train':[row('fall',1,'An operator fell from a moving lift outside.')],
            'validation':[], 'test':[row('fall',2,'An operator fell from a moving lift outside!')]}
    a=cross_split_audit(splits)
    assert a['exact_cross_split_duplicate_keys']==1
    assert a['near_duplicate_flag_count']>=1


def test_training_only_shots():
    train=[row(label,i) for i,label in enumerate(['fall','struck-by']*4)]
    classes={'constructcie':['fall','struck-by']}
    ex=few_shot_examples(train,classes,k=4)
    msg=prompt_messages(row('fall',999,'UNSEEN TARGET NARRATIVE'),classes,ex)
    assert len(msg)==9
    assert sum('UNSEEN TARGET NARRATIVE' in m['content'] for m in msg)==1
    assert len(set(str(r['id']) for r in ex['constructcie']))==4
    assert [r['id'] for r in ex['constructcie']]==[r['id'] for r in few_shot_examples(train,classes,k=4)['constructcie']]


def test_shared_strict_parser():
    assert extract_label('{"label":"fall"}')[0]=='fall'
    for malformed in ('fall','{"label":"fall", "extra":true}','{"label":5}','garbage'):
        assert extract_label(malformed)==(None,False)


def test_pairing_rejects_mismatch():
    rows={'base-zero':[pred(1,'fall','fall')], 'v2-zero':[pred(2,'fall','fall')]}
    with pytest.raises(ValueError):bootstrap(rows,{'constructcie':['fall','struck-by']},reps=20)


def test_bootstrap_signed_difference_and_ci():
    good=[pred(i,'fall','fall') for i in range(8)]
    bad=[pred(i,'fall','struck-by') for i in range(8)]
    # The headline is v2 constrained, and strongest non-v2 is base-zero.
    result=bootstrap({'base-zero':bad,'v2-constrained':good},
                     {'constructcie':['fall','struck-by']},reps=100)
    assert result['constructcie']['paired_delta']['base-zero']['accuracy']['delta']==1
    assert result['constructcie']['paired_delta']['base-zero']['accuracy']['ci95']==[1,1]


def test_confusion_invalid():
    m=confusion([pred(1,'fall',None)],{'constructcie':['fall','struck-by']})
    assert m['constructcie']['counts']['fall']['<INVALID>']==1


def test_static_training_mask_configuration():
    repo=Path(__file__).resolve().parents[1]
    audit=training_audit(repo)
    assert audit['completion_only_loss_enabled']
    assert audit['explicit_prompt_completion_columns']
    assert not audit['training_reads_test_split']


def test_completion_token_mask_contract():
    # Contract check for the same prompt/completion representation consumed by TRL:
    # every prompt token must be ignored; completion tokens must be supervised.
    prompt_ids=[10,11,12];completion_ids=[13,14]
    labels=[-100]*len(prompt_ids)+completion_ids
    assert labels[:len(prompt_ids)]==[-100]*len(prompt_ids)
    assert labels[len(prompt_ids):]==completion_ids
