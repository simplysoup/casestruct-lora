from __future__ import annotations
import json
from pathlib import Path
from datasets import Dataset
from .data import load
from .prompts import template, target
from .runtime import config, ensure_cuda, seed_all

def train(config_path: str, data_dir: str, wandb_mode: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer
    from os import environ
    c = config(config_path)
    ensure_cuda()
    seed_all(c["seed"])
    if wandb_mode:
        environ["WANDB_MODE"] = wandb_mode
    mode = environ.get("WANDB_MODE", "offline")
    if mode == "online" and not environ.get("WANDB_API_KEY"):
        print("W&B online selected: run `wandb login` first if not already authenticated")
    model_id = c["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    def convert(rows):
        return Dataset.from_list([{"prompt": template(tokenizer, r["narrative"]),
                                   "completion": target(r["label"]) + "<|im_end|>"} for r in rows])
    training = convert(load(Path(data_dir), "train"))
    validation = convert(load(Path(data_dir), "validation"))
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype,
                                                 low_cpu_mem_usage=True)
    model.config.use_cache = False
    t = c["train"]
    out = Path(c["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    args = SFTConfig(
        output_dir=str(out), max_length=c["max_length"],
        completion_only_loss=True, packing=False,
        num_train_epochs=t["epochs"], per_device_train_batch_size=t["per_device_batch_size"],
        per_device_eval_batch_size=1, gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"], lr_scheduler_type="cosine", warmup_ratio=0.05,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
        logging_steps=t["logging_steps"], eval_strategy="steps", eval_steps=t["eval_steps"],
        save_strategy="steps", save_steps=t["save_steps"], save_total_limit=2,
        max_steps=t.get("max_steps", -1), optim="adamw_torch",
        report_to=["wandb"] if mode != "disabled" else "none",
        run_name="local-lora-r8", seed=c["seed"],
        dataset_num_proc=1,
    )
    peft = LoraConfig(
        r=t["lora_rank"], lora_alpha=t["lora_alpha"], lora_dropout=t["lora_dropout"],
        bias="none", task_type="CAUSAL_LM", target_modules="all-linear",
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=training, eval_dataset=validation,
                         peft_config=peft, processing_class=tokenizer)
    print("Trainable parameters:", trainer.model.print_trainable_parameters())
    stats = trainer.train()
    trainer.save_model(str(out / "adapter"))
    tokenizer.save_pretrained(str(out / "adapter"))
    (out / "training_metrics.json").write_text(json.dumps(stats.metrics, indent=2) + "\n")
    print("Adapter saved:", out / "adapter")
