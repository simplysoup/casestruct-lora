import json
from pathlib import Path
from casestruct.v2.data import parse_constructcie, parse_osha, grouped_split, key_for
from casestruct.v2.model import extract_label
from casestruct.v2.evaluate import metrics


def test_importers():
    c=parse_constructcie([{'narrative':'A worker fell from an elevated roof onto the ground below and was seriously injured.', 'source_id':'1','accident_type':['fall']}])
    assert c[0]['label']=='fall'
    o=parse_osha([{'narrative':'An employee sustained an injury while working with machinery at a manufacturing site and was admitted to hospital.', 'coded_labels':{'nature_title':'Fractures'}}]*26,min_count=25)
    assert len(o)==26


def test_grouped_split_no_duplicate_crossing():
    rows=[]
    for i in range(30):
        row={'task':'constructcie','label':'fall','narrative':f'Worker fell from scaffolding on site number {i}', 'id':str(i)}
        rows.append(row)
    rows.append(dict(rows[0]))
    s=grouped_split(rows)
    keys=[{key_for(r['narrative']) for r in rs} for rs in s.values()]
    assert not (keys[0]&keys[1] or keys[0]&keys[2] or keys[1]&keys[2])


def test_strict_parser():
    assert extract_label('{"label":"fall"}')==('fall',True)
    assert extract_label('{"label":"fall","why":"oops"}')==(None,False)
    assert extract_label('label: fall')==(None,False)


def test_metrics():
    rows=[{'task':'constructcie','gold':'fall','prediction':'fall','correct':True,'json_valid':True,'allowed_label':True,'latency_s':1.0}, {'task':'constructcie','gold':'struck-by','prediction':'fall','correct':False,'json_valid':True,'allowed_label':True,'latency_s':2.0}]
    x=metrics(rows)['constructcie']
    assert x['accuracy']==.5 and x['strict_json_rate']==1
