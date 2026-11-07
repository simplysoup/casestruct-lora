import json
import random
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from trl import SFTConfig, SFTTrainer

from casestruct.v2.data import load_jsonl
from casestruct.v2.train import build_training_dataset


def tiny_tokenizer():
    vocab = {
        "[UNK]": 0,
        "[PAD]": 1,
        "<|im_start|>": 2,
        "<|im_end|>": 3,
        "user": 4,
        "assistant": 5,
    }
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="<|im_end|>",
    )
    fast.chat_template = (
        "{{ '<|im_start|>user\\n' + messages[0]['content'] + "
        "'<|im_end|>\\n<|im_start|>assistant\\n' }}"
    )
    return fast


def test_real_trl_batch_masks_prompt_and_trains_completion(tmp_path):
    tokenizer = tiny_tokenizer()
    row = {
        "task": "constructcie",
        "id": "synthetic-mask-check",
        "narrative": "Synthetic test text only.",
        "label": "fall",
    }
    classes = {"constructcie": ["fall", "other"]}
    dataset = build_training_dataset([row], tokenizer, classes)
    prompt = dataset[0]["prompt"]
    expected_prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=256,
            n_ctx=256,
            n_embd=32,
            n_layer=1,
            n_head=2,
        )
    )
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(tmp_path),
            max_length=256,
            completion_only_loss=True,
            packing=False,
            per_device_train_batch_size=1,
            report_to="none",
            gradient_checkpointing=False,
        ),
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # This is the same collator-produced batch consumed by Trainer during training.
    batch = trainer.data_collator([trainer.train_dataset[0]])
    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]
    assert input_ids[: len(expected_prompt_ids)].tolist() == expected_prompt_ids
    assert all(label == -100 for label in labels[: len(expected_prompt_ids)].tolist())
    attended_length = int(batch["attention_mask"][0].sum())
    completion_labels = labels[len(expected_prompt_ids) : attended_length].tolist()
    assert completion_labels
    assert all(label != -100 for label in completion_labels)


@pytest.mark.skipif(not Path("data/v2").exists(), reason="prepared data not present")
def test_prepared_batch_masks_each_prompt_and_supervises_full_completion(tmp_path):
    data_dir = Path("data/v2")
    manifest = json.loads((data_dir / "manifest.json").read_text())
    rows = list(load_jsonl(data_dir / "train.jsonl"))
    assert len(rows) >= 32
    rows = random.Random(71).sample(rows, 32)
    tokenizer = tiny_tokenizer()
    dataset = build_training_dataset(rows, tokenizer, manifest["class_names"])

    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=2048,
            n_ctx=2048,
            n_embd=32,
            n_layer=1,
            n_head=2,
        )
    )
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(tmp_path),
            max_length=2048,
            completion_only_loss=True,
            packing=False,
            per_device_train_batch_size=len(rows),
            report_to="none",
            gradient_checkpointing=False,
        ),
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Collate all 32 prepared examples together, as in a training batch.
    batch = trainer.data_collator([trainer.train_dataset[i] for i in range(len(rows))])
    for index, example in enumerate(dataset):
        prompt_ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(example["completion"], add_special_tokens=False)["input_ids"]
        input_ids = batch["input_ids"][index]
        labels = batch["labels"][index]
        attended_length = int(batch["attention_mask"][index].sum())

        assert input_ids[: len(prompt_ids)].tolist() == prompt_ids
        assert all(label == -100 for label in labels[: len(prompt_ids)].tolist())
        completion_labels = labels[len(prompt_ids) : attended_length].tolist()
        assert completion_labels
        assert all(label != -100 for label in completion_labels)
        assert len(completion_labels) == len(completion_ids)
