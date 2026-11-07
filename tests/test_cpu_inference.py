import torch

from casestruct.v2 import model as model_module


def test_model_loader_selects_cpu_when_cuda_is_unavailable(monkeypatch):
    calls = {}

    class FakeTokenizer:
        pad_token = None
        eos_token = "<eos>"

    class FakeModel:
        def to(self, device):
            calls["device"] = device
            return self

        def eval(self):
            calls["eval"] = True
            return self

    class FakeTokenizerLoader:
        @staticmethod
        def from_pretrained(model_id):
            calls["tokenizer_model_id"] = model_id
            return FakeTokenizer()

    class FakeModelLoader:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls["model_model_id"] = model_id
            calls["load_kwargs"] = kwargs
            return FakeModel()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(model_module, "MODEL_ID", "test-model")
    monkeypatch.setattr("transformers.AutoTokenizer", FakeTokenizerLoader)
    monkeypatch.setattr("transformers.AutoModelForCausalLM", FakeModelLoader)

    tokenizer, loaded_model = model_module.load_model()

    assert isinstance(tokenizer, FakeTokenizer)
    assert isinstance(loaded_model, FakeModel)
    assert tokenizer.pad_token == tokenizer.eos_token
    assert calls["device"] == "cpu"
    assert calls["load_kwargs"]["dtype"] is torch.float32
    assert calls["eval"] is True
