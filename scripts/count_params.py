import torch

from lean.config import load_config
from lean.model import Lean

cfg = load_config("configs/base.yaml")   # вернёт LeanConfig
m = Lean(cfg.model)
n = sum(p.numel() for p in m.parameters())
print(f"параметров: {n:,} (~{n/1e6:.1f}M)")
print(f"веса FP16: {n*2/1024**2:.0f} MB | ориентир обучения: {n*12/1024**3:.2f} ГБ")

V = cfg.model.vocab_size
x = torch.randint(0, V, (2, 64))
logits, cache = m(x)
print(f"forward ок: logits {tuple(logits.shape)}, кэш {'None (обучение)' if cache is None else len(cache)}")
nxt = torch.randint(0, V, (2, 1))
logits2, _ = m(nxt, cache=[], start_pos=64)   # ← [] для генерации, не None
print(f"KV-cache ок: logits {tuple(logits2.shape)}")
