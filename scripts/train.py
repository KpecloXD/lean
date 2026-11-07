# scripts/train.py
"""Обучение Lean: батчи по окнам, loss только на токенах ассистента."""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lean.config import load_config
from lean.model import Lean


def get_batch(ids, mask, batch_size, max_seq, device, rng):
    """Случайные окна длины max_seq из плоского массива.
    Возвращает x, y, loss_mask. y — сдвинутые на 1 labels.
    """
    n = len(ids)
    if n <= max_seq:
        raise ValueError(f"нужно больше {max_seq} токенов, есть {n}")
    # верхняя граница включается: последний валидный старт = n - max_seq
    starts = rng.integers(0, n - max_seq + 1, size=batch_size)
    x = np.stack([ids[s:s + max_seq] for s in starts])
    y = np.stack([ids[s + 1:s + max_seq + 1] for s in starts])
    m = np.stack([mask[s + 1:s + max_seq + 1] for s in starts])
    return (
        torch.from_numpy(x).long().to(device),
        torch.from_numpy(y).long().to(device),
        torch.from_numpy(m).float().to(device),
    )


def make_labels(y, m):
    """Там, где mask=0, ставим -100 — cross_entropy игнорирует."""
    labels = y.clone()
    labels[m == 0] = -100
    return labels


def slice_dialogs(ids, mask, offsets, idxs):
    """Плоские ids/mask из выбранных диалогов.

    offsets — границы диалогов: диалог i занимает [offsets[i], offsets[i+1]).
    """
    parts_ids = [ids[offsets[i]:offsets[i + 1]] for i in idxs]
    parts_mask = [mask[offsets[i]:offsets[i + 1]] for i in idxs]
    return np.concatenate(parts_ids), np.concatenate(parts_mask)


@torch.no_grad()
def evaluate(model, ids, mask, batch_size, max_seq, device, rng, iters=20):
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(iters):
        x, y, m = get_batch(ids, mask, batch_size, max_seq, device, rng)
        logits, _ = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            make_labels(y, m).reshape(-1),
            ignore_index=-100,
        )
        losses.append(loss.item())
    model.train(was_training)
    return sum(losses) / len(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--data", default="data/processed/lean_data.npz")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"device: {device}")

    cfg = load_config(args.config)
    m_cfg, t_cfg = cfg.model, cfg.training
    print(f"vocab={m_cfg.vocab_size}, d_model={m_cfg.d_model}, layers={m_cfg.n_layers}")

    # данные
    data = np.load(args.data)
    ids, mask, offsets = data["ids"], data["mask"], data["offsets"]
    n_dialogs = len(offsets) - 1
    print(
        f"данные: {len(ids)} токенов, маска=1: {mask.mean():.1%}, "
        f"диалогов: {n_dialogs}"
    )

    # --- train/val split по диалогам (без утечки) ---
    split_rng = np.random.default_rng(args.seed)
    perm = split_rng.permutation(n_dialogs)
    n_val = round(n_dialogs * t_cfg.val_fraction)
    if n_dialogs > 1:
        n_val = min(max(n_val, 1), n_dialogs - 1)
    else:
        n_val = 0

    if n_val > 0:
        val_idx = np.sort(perm[:n_val])
        train_idx = np.sort(perm[n_val:])
        train_ids, train_mask = slice_dialogs(ids, mask, offsets, train_idx)
        val_ids, val_mask = slice_dialogs(ids, mask, offsets, val_idx)
    else:
        print("!! диалогов меньше 2 — валидация будет на train (не информативно)")
        train_ids, train_mask = ids, mask
        val_ids, val_mask = ids, mask

    if len(train_ids) <= m_cfg.max_seq or len(val_ids) <= m_cfg.max_seq:
        raise SystemExit(
            f"train={len(train_ids)}, val={len(val_ids)}, max_seq={m_cfg.max_seq}. "
            f"Нужно больше данных или меньше max_seq/val_fraction."
        )
    print(f"train: {len(train_ids)} токенов, val: {len(val_ids)} токенов")

    model = Lean(m_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"параметров: {n_params:,} (~{n_params/1e6:.1f}M)")

    # оптимизатор + warmup
    opt = torch.optim.AdamW(
        list(model.parameters()), lr=t_cfg.lr, weight_decay=t_cfg.weight_decay
    )

    def lr_lambda(step):
        if step < t_cfg.warmup_steps:
            # +1: первый opt.step() не должен идти с lr=0
            return (step + 1) / max(1, t_cfg.warmup_steps)
        # cosine до 0 к max_steps
        progress = (step - t_cfg.warmup_steps) / max(
            1, t_cfg.max_steps - t_cfg.warmup_steps
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    t0 = time.time()
    running = None
    train_rng = np.random.default_rng(args.seed + 1)
    best_val_loss = float("inf")  # для сохранения лучшей модели
    best_step = 0

    for step in range(1, t_cfg.max_steps + 1):
        x, y, m = get_batch(
            train_ids, train_mask, t_cfg.batch_size, m_cfg.max_seq, device, train_rng
        )
        logits, _ = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            make_labels(y, m).reshape(-1),
            ignore_index=-100,
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg.grad_clip)
        opt.step()
        sched.step()

        running = loss.item() if running is None else 0.95 * running + 0.05 * loss.item()

        if step % 10 == 0:
            dt = time.time() - t0
            lr = sched.get_last_lr()[0]
            print(
                f"step {step:5d}/{t_cfg.max_steps} | loss {running:.4f} | "
                f"lr {lr:.2e} | {dt:.1f}s"
            )

        if step % t_cfg.eval_every == 0 or step == t_cfg.max_steps:
            val = evaluate(
                model,
                val_ids,
                val_mask,
                t_cfg.batch_size,
                m_cfg.max_seq,
                device,
                np.random.default_rng(args.seed + 2),
            )
            print(f"  eval @ {step}: val_loss {val:.4f}")

            # Сохраняем, ТОЛЬКО если улучшились
            if val < best_val_loss:
                best_val_loss = val
                best_step = step
                ckpt = out_dir / "lean_best.pt"
                torch.save(
                    {
                        "step": step,
                        "best_val_loss": best_val_loss,
                        "model": model.state_dict(),
                        "opt": opt.state_dict(),
                        "sched": sched.state_dict(),
                        "config": {
                            "model": m_cfg.__dict__,
                            "training": t_cfg.__dict__,
                        },
                    },
                    ckpt,
                )
                print(f"  новый лучший val_loss: {val:.4f} (шаг {step}) -> сохранено: {ckpt}")

    print(f"\nобучение завершено. лучший val_loss: {best_val_loss:.4f} на шаге {best_step}")
    print(f"веса: {out_dir / 'lean_best.pt'}")

if __name__ == "__main__":
    main()
