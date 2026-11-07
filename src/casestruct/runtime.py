from __future__ import annotations
import os
import random
from pathlib import Path
import numpy as np
import torch
import yaml
from dotenv import load_dotenv

load_dotenv()

def config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: training requires an NVIDIA GPU with CUDA-enabled PyTorch")
    print(f"CUDA: {torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")

def model_and_tokenizer(model_id, adapter=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else
        (torch.float16 if torch.cuda.is_available() else torch.float32),
        low_cpu_mem_usage=True,
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer
