# scripts/train_tokenizer.py
import argparse
import glob

from lean.tokenizer import train_tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=10795, help="Размер словаря")
    ap.add_argument("--out", default="data/processed/tokenizer.json")
    args = ap.parse_args()

    print("Ищу файлы в data/raw/*.jsonl...")
    paths = sorted(glob.glob("data/raw/*.jsonl"))
    if not paths:
        print("ОШИБКА: не найдено ни одного файла в data/raw/*.jsonl")
        raise SystemExit(1)

    tok = train_tokenizer(paths, args.vocab, args.out)

    print("\nГотово!")
    print(f"Желаемый vocab: {args.vocab}")
    print(f"Реальный vocab: {tok.get_vocab_size()}")
    print(
        "Если числа не совпадают, поменяй vocab_size в configs/base.yaml "
        f"на {tok.get_vocab_size()}"
    )


if __name__ == "__main__":
    main()
