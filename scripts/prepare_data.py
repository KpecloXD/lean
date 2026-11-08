# scripts/prepare_data.py
import argparse
import json
from pathlib import Path

import numpy as np

from lean.config import load_config
from lean.tokenizer import LeanTokenizer


def process_file(jsonl_path, tokenizer, max_seq, min_asst_tokens=8):
    all_ids, all_mask = [], []
    offsets = [0]  # границы диалогов

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            messages = rec.get("messages", [])
            if not messages:
                continue

            ids, mask = tokenizer.dialogue_to_ids(messages)
            if not ids:
                continue
            stride = max_seq  # без перекрытия

            before = len(all_ids)
            start = 0
            while start < len(ids):
                end = min(start + max_seq, len(ids))
                chunk_ids = ids[start:end]
                chunk_mask = mask[start:end]

                if sum(chunk_mask) >= min_asst_tokens:
                    all_ids.extend(chunk_ids)
                    all_mask.extend(chunk_mask)

                if end == len(ids):
                    break
                start += stride

            # Запоминаем границу только если диалог реально что-то дал,
            # иначе в offsets появляются пустые диалоги.
            if len(all_ids) > before:
                offsets.append(len(all_ids))

    return (np.array(all_ids, dtype=np.int32),
            np.array(all_mask, dtype=np.uint8),
            np.array(offsets, dtype=np.int64))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--input", default="data/raw/lean.jsonl")
    ap.add_argument("--tokenizer", default="data/processed/tokenizer.json")
    ap.add_argument("--output", default="data/processed/lean_data.npz")
    args = ap.parse_args()

    cfg = load_config(args.config, tokenizer_path=args.tokenizer)
    tok = LeanTokenizer(args.tokenizer)

    print(f"Обработка {args.input}...")
    ids, mask, offsets = process_file(args.input, tok, cfg.model.max_seq)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, ids=ids, mask=mask, offsets=offsets)

    print(f"Сохранено: {args.output}")
    print(f"Токенов: {len(ids)}, Маска=1: {mask.sum() / max(1, len(mask)):.1%}")

if __name__ == "__main__":
    main()
