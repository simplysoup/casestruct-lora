from casestruct.data import generate, load
from casestruct.schema import Extraction

def test_splits_are_disjoint_and_deterministic(tmp_path):
    counts = {"train": 25, "validation": 8, "test": 8, "challenge": 8}
    a = tmp_path / "a"
    b = tmp_path / "b"
    generate(a, counts=counts)
    generate(b, counts=counts)
    families = []
    narratives = []
    for split in counts:
        aa, bb = load(a, split), load(b, split)
        assert aa == bb and len(aa) == counts[split]
        families.append(set(r["family"] for r in aa))
        narratives.extend(r["narrative"] for r in aa)
        for r in aa:
            Extraction.model_validate(r["label"])
    assert len(narratives) == len(set(narratives))
    for i in range(len(families)):
        for j in range(i+1, len(families)):
            assert not families[i] & families[j]
