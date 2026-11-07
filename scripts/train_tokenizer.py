# scripts/train_tokenizer.py
import argparse
import glob

from lean.tokenizer import train_tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--vocab", type=int, default=32000,
                help="желаемый vocab_size; реальный может быть меньше на малом корпусе")
ap.add_argument("--out", default="data/processed/tokenizer.json")
args = ap.parse_args()

tok = train_tokenizer(sorted(glob.glob("data/raw/*.jsonl")), args.vocab, args.out)

print(f"\nжелаемый vocab: {args.vocab}")
print(f"реальный vocab: {tok.get_vocab_size()}")
print(f"ВСТАВЬ В base.yaml: vocab_size: {tok.get_vocab_size()}")
