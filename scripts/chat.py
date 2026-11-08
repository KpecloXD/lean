# scripts/chat.py
import argparse

import torch

from lean.config import load_config
from lean.model import Lean
from lean.tokenizer import LeanTokenizer


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=50, device="cpu", temp=1.0, top_k=50):
    model.eval()

    messages = [{"role": "user", "content": prompt}]
    ids, _ = tokenizer.dialogue_to_ids(messages)
    ids.append(2) # <assistant>

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    cache = []
    generated_ids = []

    for step in range(max_new_tokens):
        if step == 0:
            curr_ids = input_ids
            start_pos = 0
        else:
            curr_ids = torch.tensor([[generated_ids[-1]]], dtype=torch.long, device=device)
            start_pos = input_ids.shape[1] + step - 1

        logits, cache = model(curr_ids, cache=cache, start_pos=start_pos)
        next_token_logits = logits[0, -1, :].clone()

        # --- ТУТ МАГИЯ ТЕМПЕРАТУРЫ И TOP_K ---
        if temp > 0:
            # 1. Делим на температуру (чем больше temp, тем больше хаоса)
            next_token_logits = next_token_logits / temp

            # 2. Top-K: оставляем только K самых вероятных токенов
            if top_k > 0:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                # Все, что ниже K-го токена, превращаем в -бесконечность
                next_token_logits[next_token_logits < v[-1]] = float('-inf')

            # 3. Выбираем случайно с учетом вероятностей
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
        else:
            # Жадная генерация (как было)
            next_token = torch.argmax(next_token_logits).item()
        # -------------------------------------

        if next_token == 3: # <end>
            break

        generated_ids.append(next_token)

    return tokenizer.decode(generated_ids, skip_special=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--ckpt", default="checkpoints/lean_best.pt")
    ap.add_argument("--tokenizer", default="data/processed/tokenizer.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--temp", type=float, default=0.8, help="Температура (0 = жадно, >1 = хаос)")
    ap.add_argument("--top_k", type=int, default=50, help="Оставить только топ-K слов")
    args = ap.parse_args()

    device = args.device

    cfg = load_config(args.config)
    tok = LeanTokenizer(args.tokenizer)
    model = Lean(cfg.model).to(device)

    print(f"Загрузка весов из {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])

    val_loss = ckpt.get('best_val_loss', 0.0)
    print(f"Модель загружена (шаг {ckpt['step']}, val_loss {val_loss:.4f})")
    print(f"Температура: {args.temp} | Top-K: {args.top_k}")
    print("Введите 'выход', чтобы завершить.\n")

    while True:
        try:
            user_input = input("Вы: ")
            if user_input.lower() in ["выход", "exit", "quit"]:
                break
            if not user_input.strip():
                continue

            response = generate(
                model, tok, user_input,
                max_new_tokens=50,
                device=device,
                temp=args.temp,
                top_k=args.top_k
            )
            print(f"Lean: {response}\n")

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
