"""Load model; perform identical generation with optional task adapter."""
from __future__ import annotations
import json
import re
import torch
from .prompts import chat_prefix

MODEL_ID = 'Qwen/Qwen3-0.6B'


def load_model(model_id=MODEL_ID, adapter=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for model inference/training; check torch CUDA installation.')
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tok=AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(model_id,dtype=dtype,low_cpu_mem_usage=True).to('cuda')
    if adapter: model=PeftModel.from_pretrained(model,str(adapter))
    model.eval()
    return tok,model


def extract_label(raw):
    # Treat malformed outputs as invalid rather than rescuing arbitrary text.
    raw=re.sub(r'<think>.*?</think>', '', raw, flags=re.S).strip()
    try:
        obj=json.loads(raw)
    except ValueError:
        return None,False
    if not isinstance(obj,dict) or set(obj) != {'label'} or not isinstance(obj['label'],str):
        return None,False
    return obj['label'],True


@torch.inference_mode()
def generate(tok,model,row,classes,max_new_tokens=72):
    p=chat_prefix(tok,row,classes)
    x=tok(p,return_tensors='pt').to('cuda')
    generated=model.generate(**x,max_new_tokens=max_new_tokens,do_sample=False,pad_token_id=tok.pad_token_id)
    return tok.decode(generated[0, x['input_ids'].shape[1]:],skip_special_tokens=True).strip()
