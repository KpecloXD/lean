# src/lean/config.py
from dataclasses import dataclass

import yaml


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    kv_heads: int
    ffn_hidden: int
    dropout: float
    max_seq: int

    @property
    def head_dim(self): return self.d_model // self.n_heads
    @property
    def n_rep(self): return self.n_heads // self.kv_heads


@dataclass
class TrainingConfig:
    batch_size: int
    lr: float
    weight_decay: float
    max_steps: int
    grad_clip: float
    eval_every: int
    warmup_steps: int
    val_fraction: float = 0.1


@dataclass
class LeanConfig:
    model: ModelConfig
    training: TrainingConfig


def load_config(path, tokenizer_path=None):
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    m = ModelConfig(**raw["model"])
    t = TrainingConfig(**raw["training"])

    # архитектура
    assert m.d_model % m.n_heads == 0, "d_model должно делиться на n_heads"
    assert m.n_heads % m.kv_heads == 0, "n_heads должно делиться на kv_heads"
    assert m.d_model % 2 == 0, "d_model должно быть чётным (для RoPE)"
    assert m.max_seq > 0, "max_seq должен быть > 0"

    # тренировка
    assert t.batch_size > 0, "batch_size должен быть > 0"
    assert t.lr > 0, "lr должен быть > 0"
    assert t.grad_clip > 0, "grad_clip должен быть > 0"
    assert 0.0 <= t.val_fraction < 1.0, "val_fraction должен быть в [0, 1)"

    # сверка vocab_size с токенизатором
    if tokenizer_path is not None:
        from lean.tokenizer import LeanTokenizer
        real = LeanTokenizer(tokenizer_path).vocab_size
        assert m.vocab_size == real, (
            f"vocab_size в конфиге {m.vocab_size}, а у токенизатора {real}. "
            f"Поставь {real} в base.yaml."
        )

    return LeanConfig(model=m, training=t)
