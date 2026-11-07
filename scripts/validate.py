# scripts/validate.py — приёмка датасета lean.jsonl
import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict

STYLES = {"friendly", "serious", "playful", "concise", "teacher"}
ROLES = {"user", "assistant"}


def get_word_set(text: str) -> set:
    """Быстрое извлечение уникальных слов в нижнем регистре."""
    return set(re.findall(r"\w+", text.lower()))


def jaccard(a: set, b: set) -> float:
    """Сходство Жаккара между двумя множествами слов."""
    union_len = len(a | b)
    return len(a & b) / union_len if union_len else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/raw/lean.jsonl")
    ap.add_argument("--sim", type=float, default=0.85, help="порог сходства для near-дублей")
    ap.add_argument("--verbose", action="store_true", help="показать похожие пары")
    args = ap.parse_args()

    errs = []
    sim_pairs = []

    # Хранилища для проверок
    groups = defaultdict(list)          # id -> список записей
    seen_exact = {}                     # sha1(текст) -> (line, id, style)
    records = []                        # (line, id, style, word_set) для near-дублей
    subject_counts = Counter()
    style_counts = Counter()
    total = 0

    try:
        with open(args.file, encoding="utf-8") as f:
            for ln, raw in enumerate(f, 1):
                s = raw.strip()
                if not s:
                    continue
                total += 1

                try:
                    rec = json.loads(s)
                except json.JSONDecodeError as e:
                    errs.append(f"L{ln}: не JSON: {e}")
                    continue

                rid = str(rec.get("id", "")).strip()
                if not rid:
                    errs.append(f"L{ln}: пустой id")
                    rid = f"__missing_{ln}"

                meta = rec.get("meta") or {}
                subj = str(meta.get("subject", "")).strip()
                st = str(meta.get("style", "")).strip().lower()
                msgs = rec.get("messages")

                tag = f"L{ln} id={rid}.{st or '?'}"

                # --- Проверки метаданных ---
                if not subj:
                    errs.append(f"{tag}: нет или пустой subject")
                elif subj != subj.lower():
                    errs.append(f"{tag}: subject не в lowercase")

                if st not in STYLES:
                    errs.append(f"{tag}: style '{st}' не из {sorted(STYLES)}")

                # --- Проверки сообщений ---
                if not isinstance(msgs, list) or not msgs:
                    errs.append(f"{tag}: messages пустой")
                    groups[rid].append((ln, rec))
                    continue

                roles = [m.get("role") for m in msgs]
                if roles[0] != "user":
                    errs.append(f"{tag}: первая роль {roles[0]!r}, должна быть user")
                if any(r not in ROLES for r in roles):
                    errs.append(f"{tag}: неизвестная роль")

                # Проверка чередования ролей
                for j in range(1, len(roles)):
                    if roles[j] == roles[j - 1]:
                        errs.append(f"{tag}: нет чередования на {j}")
                        break

                cont = [str(m.get("content", "")).strip() for m in msgs]
                if any(not c for c in cont):
                    errs.append(f"{tag}: пустая реплика")

                # Проверка turns
                turns = meta.get("turns")
                if turns is not None and turns != len(msgs):
                    errs.append(f"{tag}: turns={turns} != {len(msgs)}")

                # --- Проверка дублей (exact) ---
                # sha1, а не hash(): не зависит от PYTHONHASHSEED
                digest = hashlib.sha1(repr(tuple(cont)).encode("utf-8")).hexdigest()
                if digest in seen_exact:
                    pl, pi, ps = seen_exact[digest]
                    errs.append(f"{tag}: точная копия L{pl} id={pi}.{ps}")
                else:
                    seen_exact[digest] = (ln, rid, st)

                # --- Сбор статистики ---
                groups[rid].append((ln, rec))
                if subj:
                    subject_counts[subj] += 1
                if st in STYLES:
                    style_counts[st] += 1
                records.append((ln, rid, st, get_word_set(" ".join(cont))))

    except FileNotFoundError:
        print(f"Файл {args.file} не найден.")
        sys.exit(1)

    # --- Проверка групп (1 subject = 5 стилей) ---
    for rid, items in groups.items():
        sts = [str(r.get("meta", {}).get("style", "")).strip().lower() for _, r in items]

        if len(items) != 5:
            errs.append(f"id={rid}: {len(items)} строк, ждём 5 (строки {[ln for ln, _ in items]})")

        miss = STYLES - set(sts)
        extra = set(sts) - STYLES
        if miss:
            errs.append(f"id={rid}: нет стилей {sorted(miss)}")
        if extra:
            errs.append(f"id={rid}: лишние стили {sorted(extra)}")
        if len(set(sts)) != len(sts):
            errs.append(f"id={rid}: повторы стилей")

    # --- Поиск near-дублей: сравниваем отдельные записи, а не subject-пулы ---
    # Записи одного id — это намеренные варианты стиля, их не сравниваем.
    for i in range(len(records)):
        ln1, rid1, st1, ws1 = records[i]
        for j in range(i + 1, len(records)):
            ln2, rid2, st2, ws2 = records[j]
            if rid1 == rid2:
                continue
            score = jaccard(ws1, ws2)
            if score >= args.sim:
                sim_pairs.append((score, ln1, rid1, st1, ln2, rid2, st2))

    # --- Вывод результатов ---
    for e in errs:
        print(e)

    if args.verbose:
        for sc, ln1, rid1, st1, ln2, rid2, st2 in sim_pairs:
            print(f"SIM {sc:.0%}: L{ln1} id={rid1}.{st1} <-> L{ln2} id={rid2}.{st2}")

    bal = ", ".join(f"{k} {v}" for k, v in sorted(subject_counts.items()))
    sty = "/".join(f"{s} {style_counts[s]}" for s in ["concise", "friendly", "playful", "serious", "teacher"])

    print(f"\n{args.file}: {total} записей, {len(groups)} групп, ошибок {len(errs)}, похожих пар {len(sim_pairs)}")
    print(f"subject: {bal}")
    print(f"style: {sty}")
    print("ВЕРДИКТ:", "ЧИСТ" if not errs else "ОШИБКИ")

    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
