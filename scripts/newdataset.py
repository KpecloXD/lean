import json

from datasets import load_dataset

# Загружаем только тренировочную часть, но не всю сразу
print("Загрузка датасета (это может занять пару минут)...")
ds = load_dataset("ZeroAgency/ru-big-russian-dataset", split="train", streaming=True)

output_file = "lean.jsonl"
max_dialogs = 100000  # Возьмем 10к диалогов. Хватит за глаза.
count = 0

with open(output_file, "w", encoding="utf-8") as f:
    for example in ds:
        # У датасета есть поле 'conversation' - это список сообщений
        conv = example.get("conversation", [])
        if not conv:
            continue

        # Нам нужны только диалоги, где есть user и assistant
        messages = []
        for msg in conv:
            role = msg.get("role")
            content = msg.get("content", "").strip()
            if role in ["user", "assistant"] and content:
                messages.append({"role": role, "content": content})

        # Если диалог валидный (минимум 2 сообщения), берем его
        if len(messages) >= 2:
            # Приводим к нашему формату
            record = {
                "id": count + 1,
                "meta": {
                    "subject": example.get("classified_topic", "general"),
                    "style": "default",
                    "turns": len(messages)
                },
                "messages": messages
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

        if count >= max_dialogs:
            break

print(f"Готово! Сохранено {count} диалогов в {output_file}")
