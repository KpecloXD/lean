"""Сборка ids + labels + loss_mask для обучения Lean."""
import argparse
import json
from pathlib import Path

import numpy as np

from lean.config import load_config
from lean.tokenizer import LeanTokenizer


def process_file(jsonl_path, tokenizer, max_seq, min_asst_tokens=8):
    """Читает jsonl и собирает плоские ids/mask.

    offsets — границы диалогов: диалог i занимает [offsets[i], offsets[i+1]).
    Окна режутся с stride = max_seq (без перекрытия), чтобы токены не дублировались.
    """
    all_ids, all_mask = [], []
    offsets = [0]
    total_dialogs = kept_dialogs = total_tokens = 0
    skipped_windows = 0

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
            before = len(all_ids)

            for start in range(0, len(ids), max_seq):
                end = min(start + max_seq, len(ids))
                chunk_ids = ids[start:end]
                chunk_mask = mask[start:end]

                # окно должно содержать достаточно токенов ассистента
                if sum(chunk_mask) >= min_asst_tokens:
                    all_ids.extend(chunk_ids)
                    all_mask.extend(chunk_mask)
                else:
                    skipped_windows += 1

            # диалог попал в датасет -> фиксируем его границу
            if len(all_ids) > before:
                offsets.append(len(all_ids))
                kept_dialogs += 1

            total_dialogs += 1
            total_tokens += len(ids)

    ids_arr = np.array(all_ids, dtype=np.int32)
    mask_arr = np.array(all_mask, dtype=np.uint8)
    off_arr = np.array(offsets, dtype=np.int64)
    assert len(ids_arr) == len(mask_arr)
    assert off_arr[-1] == len(ids_arr)
    return (
        ids_arr,
        mask_arr,
        off_arr,
        total_dialogs,
        kept_dialogs,
        total_tokens,
        skipped_windows,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--input", default="data/raw/lean.jsonl")
    ap.add_argument("--tokenizer", default="data/processed/tokenizer.json")
    ap.add_argument("--output", default="data/processed/lean_data.npz")
    ap.add_argument(
        "--max_seq",
        type=int,
        default=None,
        help="переопределить model.max_seq из конфига",
    )
    ap.add_argument("--min_asst_tokens", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(args.config)
    max_seq = args.max_seq if args.max_seq is not None else cfg.model.max_seq

    print(f"загрузка токенизатора из {args.tokenizer}...")
    tok = LeanTokenizer(args.tokenizer)

    print(
        f"обработка {args.input} "
        f"(max_seq={max_seq} из {args.config}, min_asst_tokens={args.min_asst_tokens})..."
    )
    (
        ids,
        mask,
        offsets,
        dialogs,
        kept,
        raw_tokens,
        skipped,
    ) = process_file(args.input, tok, max_seq, args.min_asst_tokens)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, ids=ids, mask=mask, offsets=offsets)

    print("\n--- ГОТОВО ---")
    print(f"Диалогов: {dialogs} (в датасет попало {kept})")
    print(f"Исходных токенов: {raw_tokens}")
    print(f"В датасете: {len(ids)}")
    print(f"Пропущено окон (мало ассистента): {skipped}")
    print(f"Доля маски=1: {mask.sum() / len(mask):.1%}")
    print(f"Сохранено: {args.output}")


if __name__ == "__main__":
    main()
