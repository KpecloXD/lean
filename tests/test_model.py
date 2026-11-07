# tests/test_model.py
import math

import pytest
import torch

from lean.config import ModelConfig
from lean.model import Lean
from lean.tokenizer import LeanTokenizer


@pytest.fixture
def cfg():
    return ModelConfig(
        vocab_size=1000, d_model=64, n_layers=2, n_heads=4,
        kv_heads=2, ffn_hidden=128, dropout=0.0, max_seq=32
    )


def test_output_shape(cfg):
    model = Lean(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, _ = model(ids)
    assert logits.shape == (2, 16, cfg.vocab_size)


def test_init_loss_is_sane(cfg):
    """При random init loss должен быть ~ln(vocab), а не сотни."""
    torch.manual_seed(0)
    model = Lean(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    targets = torch.randint(0, cfg.vocab_size, (2, 16))

    with torch.no_grad():
        logits, _ = model(ids)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, cfg.vocab_size), targets.reshape(-1)
    )

    assert logits.std().item() < 1.0, f"logits std={logits.std().item():.2f}"
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 2.0, (
        f"loss={loss.item():.2f}, ожидали ~{math.log(cfg.vocab_size):.2f}"
    )


def test_causal_mask(cfg):
    """Модель не должна видеть будущее."""
    model = Lean(cfg).eval()
    ids1 = torch.tensor([[1, 2, 3, 4, 5]])
    ids2 = torch.tensor([[1, 2, 3, 99, 98]])

    with torch.no_grad():
        out1, _ = model(ids1)
        out2, _ = model(ids2)

    assert torch.allclose(out1[:, :3], out2[:, :3], atol=1e-5)


def test_kv_cache_equivalence(cfg):
    torch.manual_seed(0)
    model = Lean(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 10))

    with torch.no_grad():
        cache = []  # первый шаг генерации — пустой кэш
        for i in range(10):
            step_ids = ids[:, i:i + 1]
            out_cached, cache = model(step_ids, cache=cache, start_pos=i)
            out_full, _ = model(ids[:, :i + 1])

            assert out_cached.shape == out_full[:, -1:, :].shape
            assert torch.allclose(out_cached, out_full[:, -1:, :], atol=1e-3), (
                f"KV-cache сломан на шаге {i}: "
                f"max|diff|={(out_cached - out_full[:, -1:, :]).abs().max().item():.6e}"
            )


def test_kv_cache_multitoken_matches_full(cfg):
    """Кэш + S>1 не должен подсматривать в будущее."""
    torch.manual_seed(0)
    model = Lean(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 12))

    with torch.no_grad():
        full, _ = model(ids)
        _, cache = model(ids[:, :4], cache=[])
        tail, _ = model(ids[:, 4:], cache=cache, start_pos=4)  # S=8 при непустом кэше

    assert torch.allclose(full[:, 4:], tail, atol=1e-5), (
        f"max|diff|={(full[:, 4:] - tail).abs().max().item():.6e}"
    )


def test_padding_mask_with_cache(cfg):
    """padding_mask [B,S] вместе с непустым кэшем не должен падать."""
    torch.manual_seed(0)
    model = Lean(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 6))
    mask = torch.zeros(2, 3, dtype=torch.bool)

    with torch.no_grad():
        _, cache = model(ids[:, :3], cache=[])
        out, _ = model(ids[:, 3:], cache=cache, start_pos=3, padding_mask=mask)

    assert out.shape == (2, 3, cfg.vocab_size)


def test_fully_masked_row_has_no_nan(cfg):
    """Полностью замаскированная строка не должна давать NaN."""
    torch.manual_seed(0)
    model = Lean(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        out, _ = model(ids, padding_mask=mask)

    assert not torch.isnan(out).any()


def test_loss_mask_alignment():
    """mask=1 ровно на content ассистента + <end>; head ассистента = 0."""
    tok = object.__new__(LeanTokenizer)
    tok.encode = lambda text: [10, 11]

    ids, mask = tok.dialogue_to_ids([
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ])

    # user:  <user> 10 11 <end>   assistant: <assistant> 10 11 <end>
    assert ids == [1, 10, 11, 3, 2, 10, 11, 3]
    assert mask == [0, 0, 0, 0, 0, 1, 1, 1]
