"""Single narrative extraction, with no web server or external API."""
from __future__ import annotations
from pathlib import Path
import torch
from .prompts import template
from .runtime import config, model_and_tokenizer
from .scoring import parse_prediction


def demo(config_path: str, narrative: str, use_base: bool = False):
    c = config(config_path)
    adapter = None if use_base else str(Path(c['output_dir']) / 'adapter')
    if adapter and not Path(adapter).exists():
        raise FileNotFoundError(f'No trained adapter at {adapter}; run scripts/run_all.sh first')
    model, tokenizer = model_and_tokenizer(c['model_id'], adapter)
    prompt = template(tokenizer, narrative)
    inp = tokenizer(prompt, return_tensors='pt').to(model.device)
    with torch.inference_mode():
        out = model.generate(**inp, max_new_tokens=c['max_new_tokens'], do_sample=False,
                             pad_token_id=tokenizer.pad_token_id,
                             eos_token_id=tokenizer.eos_token_id)
    raw = tokenizer.decode(out[0][inp.input_ids.shape[-1]:], skip_special_tokens=True)
    parsed, _, valid = parse_prediction(raw)
    print('Valid schema:', valid)
    print('Output:', parsed if valid else raw)
