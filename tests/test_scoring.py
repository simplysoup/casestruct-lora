import json
from casestruct.scoring import score, parse_prediction

def sample():
    return {"incident_type":"vehicle_collision", "incident_date":None, "location":None,
            "injuries": ["sprained wrist"], "medical_treatment": [],
            "liability_factors": [], "missing_information": ["incident_date", "location", "medical_treatment", "liability_factors"]}

def test_perfect():
    a = sample()
    m = score([{"label": a, "prediction": json.dumps(a)}])
    assert m["exact_match_rate"] == 1
    assert m["schema_valid_rate"] == 1
    assert m["macro_field_f1"] == 1

def test_invalid_json():
    assert parse_prediction("not-json") == (None, False, False)
    m = score([{"label": sample(), "prediction": "not-json"}])
    assert m["schema_valid_rate"] == 0
    assert m["exact_match_rate"] == 0

def test_extra_field_not_accepted():
    x = sample(); x["invented"] = 1
    assert parse_prediction(json.dumps(x))[2] is False
