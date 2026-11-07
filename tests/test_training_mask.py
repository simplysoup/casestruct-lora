from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from trl import SFTConfig, SFTTrainer

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
