from __future__ import annotations
import argparse
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description="CaseStruct-LoRA runner")
    s = p.add_subparsers(dest="command", required=True)
    d = s.add_parser("generate")
    d.add_argument("--data", default="data")
    d.add_argument("--seed", type=int, default=71)
    d.add_argument("--small", action="store_true")
    t = s.add_parser("train")
    t.add_argument("--config", default="configs/local.yaml")
    t.add_argument("--data", default="data")
    e = s.add_parser("evaluate")
    e.add_argument("--config", default="configs/local.yaml")
    e.add_argument("--data", default="data")
    e.add_argument("--mode", choices=["zero-shot", "few-shot", "adapter"], required=True)
    e.add_argument("--split", choices=["validation", "test", "challenge"], default="test")
    e.add_argument("--limit", type=int, default=0)
    x = s.add_parser("extract")
    x.add_argument("--config", default="configs/local.yaml")
    x.add_argument("--text", help="Fictional intake narrative; otherwise read stdin")
    x.add_argument("--base", action="store_true", help="Use base model without adapter")
    s.add_parser("report")
    args = p.parse_args()
    if args.command == "generate":
        from .data import generate
        counts = {"train": 80, "validation": 12, "test": 12, "challenge": 12} if args.small else None
        print(generate(Path(args.data), args.seed, counts))
    elif args.command == "train":
        from .train import train
        train(args.config, args.data)
    elif args.command == "evaluate":
        from .evaluate import evaluate
        evaluate(args.config, args.data, args.split, args.mode, args.limit)
    elif args.command == "extract":
        from .demo import demo
        import sys
        narrative = args.text or sys.stdin.read().strip()
        if not narrative:
            p.error("No narrative supplied")
        demo(args.config, narrative, args.base)
    else:
        from .report import report
        report()
if __name__ == "__main__":
    main()
