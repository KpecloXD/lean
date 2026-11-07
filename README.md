# lean

Небольшая decoder-only языковая модель на PyTorch: GQA, RoPE, SwiGLU, KV-cache.
Учится на диалогах, loss считается только на ответах ассистента.

## Что уже есть

- `src/lean/model.py` — модель: RMSNorm, RoPE, GQA, SwiGLU, KV-cache, causal mask.
- `src/lean/config.py` — конфиги `ModelConfig` / `TrainingConfig` / `LeanConfig`.
- `src/lean/tokenizer.py` — ByteLevel-BPE + упаковка диалогов в `ids` с loss-маской.
- `scripts/train_tokenizer.py` — обучение BPE-токенизатора.
- `scripts/prepare_data.py` — сборка `ids` / `mask` / `offsets` из `data/raw/*.jsonl`.
- `scripts/train.py` — обучение, train/val split по диалогам, сохранение лучшей модели.
- `scripts/validate.py` — приёмка датасета.
- `scripts/count_params.py` — число параметров + проверка forward/KV-cache.
- `tests/` — pytest.

## Текущая конфигурация (`configs/base.yaml`)

| Параметр | Значение |
|---|---|
| `vocab_size` | 10795 |
| `d_model` | 256 |
| `n_layers` | 4 |
| `n_heads` / `kv_heads` | 4 / 2 |
| `ffn_hidden` | 512 |
| `max_seq` | 256 |
| параметров | ~5.1M |

## Установка

Python 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

## Данные

В репозитории уже лежат:

- `data/raw/lean.jsonl` — 1220 диалогов (244 группы × 5 стилей);
- `data/processed/tokenizer.json` — BPE-токенизатор на 10795 токенов;
- `data/processed/lean_data.npz` — 81771 токен, маска ответов 68.2%.

Пересобрать:

```powershell
python scripts/prepare_data.py            # max_seq берётся из configs/base.yaml
python scripts/validate.py                # приёмка датасета
```

## Обучение

```powershell
python scripts/train.py --config configs/base.yaml
```

Лучшая модель сохраняется в `checkpoints/lean_best.pt` по `val_loss`.
Доля диалогов на валидацию — `val_fraction` в `configs/base.yaml`.

## Проверки

```powershell
python scripts/count_params.py
pytest -q
ruff check src scripts tests
```

## Секреты

`.env` в git не попадает. Шаблон — `.env.example`:

```text
LEAN_API_BASE=https://text.pollinations.ai/openai
LEAN_API_KEY=none
LEAN_MODEL=openai-fast
```

## Статус

- [x] модель + KV-cache + тесты
- [x] токенизатор + датасет
- [x] обучение с train/val split
- [ ] генерация текста
- [ ] API-клиент
