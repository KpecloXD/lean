# scripts/train.py
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
    n = len(ids)
    if n <= max_seq:
        raise ValueError(f"нужно больше {max_seq} токенов, а есть {n}")
    starts = rng.integers(0, n - max_seq + 1, size=batch_size)
    x = np.stack([ids[s:s + max_seq] for s in starts])
    y = np.stack([ids[s + 1:s + max_seq + 1] for s in starts])
    m = np.stack([mask[s + 1:s + max_seq + 1] for s in starts])
    return (torch.from_numpy(x).long().to(device),
            torch.from_numpy(y).long().to(device),
            torch.from_numpy(m).float().to(device))

def make_labels(y, m):
    labels = y.clone()
    labels[m == 0] = -100 # -100 игнорируется при подсчете ошибки
    return labels

@torch.no_grad()
def evaluate(model, ids, mask, batch_size, max_seq, device, rng, iters=20):
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(iters):
        x, y, m = get_batch(ids, mask, batch_size, max_seq, device, rng)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), make_labels(y, m).reshape(-1), ignore_index=-100)
        losses.append(loss.item())
    if was_training:
        model.train()
    return sum(losses) / len(losses)

def split_dialogs(ids, mask, offsets, val_fraction, seed):
    rng = np.random.default_rng(seed)
    num_dialogs = len(offsets) - 1
    indices = np.arange(num_dialogs)
    rng.shuffle(indices)

    val_size = int(num_dialogs * val_fraction)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    def gather(idxs):
        out_ids, out_mask = [], []
        for i in idxs:
            start, end = offsets[i], offsets[i+1]
            out_ids.extend(ids[start:end])
            out_mask.extend(mask[start:end])
        return np.array(out_ids), np.array(out_mask)

    train_ids, train_mask = gather(train_idx)
    val_ids, val_mask = gather(val_idx)
    return train_ids, train_mask, val_ids, val_mask

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
    cfg = load_config(args.config)
    m_cfg, t_cfg = cfg.model, cfg.training

    print(f"Device: {device}")
    data = np.load(args.data)
    ids, mask, offsets = data["ids"], data["mask"], data["offsets"]

    train_ids, train_mask, val_ids, val_mask = split_dialogs(ids, mask, offsets, t_cfg.val_fraction, args.seed)
    print(f"Train tokens: {len(train_ids)}, Val tokens: {len(val_ids)}")

    model = Lean(m_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t_cfg.lr, weight_decay=t_cfg.weight_decay)

    def lr_lambda(step):
        if step < t_cfg.warmup_steps:
            return (step + 1) / max(1, t_cfg.warmup_steps)
        progress = (step - t_cfg.warmup_steps) / max(1, t_cfg.max_steps - t_cfg.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    train_rng = np.random.default_rng(args.seed + 1)
    best_val_loss = float("inf")
    t0 = time.time()
    running = None

    for step in range(1, t_cfg.max_steps + 1):
        x, y, m = get_batch(train_ids, train_mask, t_cfg.batch_size, m_cfg.max_seq, device, train_rng)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), make_labels(y, m).reshape(-1), ignore_index=-100)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg.grad_clip)
        opt.step()
        sched.step()

        running = loss.item() if running is None else 0.95 * running + 0.05 * loss.item()

        if step % 10 == 0:
            print(f"step {step:4d}/{t_cfg.max_steps} | loss {running:.4f} | lr {sched.get_last_lr()[0]:.2e} | {time.time()-t0:.1f}s")

        if step % t_cfg.eval_every == 0 or step == t_cfg.max_steps:
            # Валидация на честных данных
            val_rng = np.random.default_rng(args.seed + 2)
            val = evaluate(model, val_ids, val_mask, t_cfg.batch_size, m_cfg.max_seq, device, val_rng)
            print(f"  eval @ {step}: val_loss {val:.4f}")

            if val < best_val_loss:
                best_val_loss = val
                torch.save({
                    "step": step, "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict()
                }, out_dir / "lean_best.pt")
                print("  -> Новый лучший val_loss! Сохранено: lean_best.pt")

    print(f"\nОбучение завершено. Лучший val_loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
