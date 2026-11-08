# scripts/validate.py
import argparse
import json
from collections import Counter, defaultdict

STYLES = {"friendly", "serious", "playful", "concise", "teacher"}
ROLES = {"user", "assistant"}

# Словарь для перевода и склейки тем
TRANSLATE = {
    "география": "geography",
    "биология": "biology",
    "история": "history",
    "математика": "math",
    "физика": "physics",
    "химия": "chemistry",
    "литература": "literature",
    "программирование": "programming",
    "спорт": "sports",
    "музыка": "music",
    "кино": "movies",
    "искусство": "art",
    "кулинария": "cooking",
    "наука": "science",
    # Новые переводы
    "астрономия": "astronomy",
    "животные": "animals",
    "природа": "nature",
    "технологии": "technology",
    # Склейка дублей на английском
    "mathematics": "math",
    "tech": "technology",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/raw/lean.jsonl", help="Файл для проверки и очистки")
    args = ap.parse_args()

    groups = defaultdict(list)
    total_lines = 0
    errs = []

    # 1. Читаем и группируем по id
    print(f"Читаем {args.file}...")
    try:
        with open(args.file, encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                total_lines += 1

                try:
                    rec = json.loads(line)
                except Exception as e:
                    errs.append(f"L{ln}: битый JSON ({e})")
                    continue

                # Переводим и склеиваем subject
                meta = rec.get("meta", {})
                subj = str(meta.get("subject", "")).strip().lower()

                if subj in TRANSLATE:
                    meta["subject"] = TRANSLATE[subj]
                elif subj:
                    meta["subject"] = subj

                rec["meta"] = meta

                rid = str(rec.get("id", "")).strip()
                groups[rid].append((ln, rec))
    except FileNotFoundError:
        print(f"Ошибка: файл {args.file} не найден.")
        return

    # 2. Проверяем группы
    good_groups = []
    bad_groups_count = 0

    for rid, items in groups.items():
        bad = False

        if len(items) != 5:
            errs.append(f"id={rid}: {len(items)} строк вместо 5. Группа удалена.")
            bad = True

        sts = [str(r.get("meta", {}).get("style", "")).strip().lower() for _, r in items]
        if set(sts) != STYLES:
            errs.append(f"id={rid}: неполный набор стилей {sorted(set(sts))}. Группа удалена.")
            bad = True

        if not bad:
            for _, rec in items:
                msgs = rec.get("messages", [])
                roles = [m.get("role") for m in msgs]
                conts = [str(m.get("content", "")).strip() for m in msgs]

                if not msgs or roles[0] != "user" or any(r not in ROLES for r in roles):
                    bad = True
                elif any(not c for c in conts):
                    bad = True

            if bad:
                errs.append(f"id={rid}: ошибка в ролях или пустые реплики. Группа удалена.")

        if not bad:
            good_groups.append(items)
        else:
            bad_groups_count += 1

    # 3. Сортируем группы и перенумеровываем
    good_groups.sort(key=lambda items: items[0][0])

    # 4. Перезаписываем ТОТ ЖЕ файл и считаем темы
    print(f"Перезаписываем {args.file}...")
    subj_counter = Counter()

    with open(args.file, "w", encoding="utf-8") as f:
        new_id = 1
        for items in good_groups:
            # Берем тему из первой записи группы (они все одинаковые в группе)
            subj = str(items[0][1].get("meta", {}).get("subject", "")).strip()
            if subj:
                subj_counter[subj] += 1

            for _, rec in items:
                rec["id"] = new_id
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            new_id += 1

    # 5. Итоги
    print("\n--- РЕЗУЛЬТАТ ---")
    print(f"Было строк: {total_lines}")
    print(f"Было групп: {len(groups)}")
    print(f"Удалено групп: {bad_groups_count}")
    print(f"Осталось групп: {len(good_groups)}")
    print(f"Осталось строк: {len(good_groups) * 5}")

    if errs:
        print("\nНайдены и исправлены ошибки (удалены):")
        for e in errs:
            print(f"  - {e}")

    print(f"\n--- УНИКАЛЬНЫЕ ТЕМЫ ({len(subj_counter)}) ---")
    for s, count in sorted(subj_counter.items()):
        print(f"  - {s} - {count}")

    print(f"\nФайл {args.file} очищен, темы переведены, id выровнены!")

if __name__ == "__main__":
    main()
