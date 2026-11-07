"""Two-task supervised LoRA classification, 8GB GPU starting configuration."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import Dataset

from .data import load_jsonl
from .model import MODEL_ID
from .prompts import chat_prefix, gold


def build_training_dataset(rows, tokenizer, classes):
    """Build prompt/completion examples for TRL completion-only loss."""
    return Dataset.from_list([
        {'prompt': chat_prefix(tokenizer, row, classes),
         'completion': gold(row) + '<|im_end|>'}
        for row in rows
    ])


def train(data='data/v2',output='outputs/v2',epochs=2,lr=0.00015,rank=8,max_length=768,model_id=MODEL_ID):
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    root=Path(data); classes=json.loads((root/'manifest.json').read_text())['class_names']
    tok=AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    def convert(split):
        rows=list(load_jsonl(root/f'{split}.jsonl'))
        if not rows: raise RuntimeError(f'Empty {split} data')
        return build_training_dataset(rows, tok, classes)
    train_ds=convert('train'); eval_ds=convert('validation')
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model=AutoModelForCausalLM.from_pretrained(model_id,dtype=dtype,low_cpu_mem_usage=True)
    model.config.use_cache=False
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    wandb_mode=os.getenv('WANDB_MODE','offline')
    args=SFTConfig(output_dir=str(output/'checkpoints'),max_length=max_length,completion_only_loss=True,packing=False,
       num_train_epochs=epochs,per_device_train_batch_size=1,per_device_eval_batch_size=1,
       gradient_accumulation_steps=8,learning_rate=lr,lr_scheduler_type='cosine',warmup_ratio=.05,
       gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
       bf16=dtype==torch.bfloat16,fp16=dtype==torch.float16,
       logging_steps=10,eval_strategy='epoch',save_strategy='epoch',save_total_limit=2,
       optim='adamw_torch',report_to=['wandb'] if wandb_mode!='disabled' else 'none',
       run_name='casestruct-v2-external-lora',seed=71,dataset_num_proc=1)
    peft=LoraConfig(r=rank,lora_alpha=rank*2,lora_dropout=.05,bias='none',task_type='CAUSAL_LM',target_modules='all-linear')
    trainer=SFTTrainer(model=model,args=args,train_dataset=train_ds,eval_dataset=eval_ds,peft_config=peft,processing_class=tok)
    trainer.model.print_trainable_parameters()
    result=trainer.train()
    trainer.save_model(str(output/'adapter'))
    tok.save_pretrained(str(output/'adapter'))
    (output/'training_metrics.json').write_text(json.dumps(result.metrics,indent=2)+'\n')
    print('Adapter saved:',output/'adapter')


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/v2');p.add_argument('--output',default='outputs/v2');p.add_argument('--epochs',type=float,default=2);p.add_argument('--lr',type=float,default=.00015);p.add_argument('--rank',type=int,default=8);p.add_argument('--max-length',type=int,default=768);p.add_argument('--model-id',default=MODEL_ID)
    a=p.parse_args();train(a.data,a.output,a.epochs,a.lr,a.rank,a.max_length,a.model_id)

if __name__=='__main__':main()
