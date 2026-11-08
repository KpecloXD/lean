import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIALS = ["<pad>", "<user>", "<assistant>", "<end>"]
PAD_ID, USER_ID, ASSISTANT_ID, END_ID = 0, 1, 2, 3


def iter_corpus(paths):
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                for m in rec["messages"]:
                    yield m["content"]


def train_tokenizer(paths, vocab_size, save_path):
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=SPECIALS,
        min_frequency=2, show_progress=True,
    )
    tok.train_from_iterator(iter_corpus(paths), trainer)

    for i, s in enumerate(SPECIALS):
        got = tok.token_to_id(s)
        assert got == i, f"спецтокен {s} получил id {got}, ожидали {i}"

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(save_path))
    print(f"токенизатор на {tok.get_vocab_size()} токенов сохранён: {save_path}")
    return tok


class LeanTokenizer:
    def __init__(self, path):
        self.tok = Tokenizer.from_file(str(path))
        for i, s in enumerate(SPECIALS):
            assert self.tok.token_to_id(s) == i, f"{s} не на id {i}"

    @property
    def vocab_size(self):
        return self.tok.get_vocab_size()

    def encode(self, text):
        return self.tok.encode(text).ids

    def decode(self, ids, skip_special=False):
        return self.tok.decode(ids, skip_special_tokens=skip_special)

    def dialogue_to_ids(self, messages):
        ids, mask = [], []
        for m in messages:
            is_asst = m["role"] == "assistant"
            head = ASSISTANT_ID if is_asst else USER_ID
            content = self.encode(m["content"])
            seg = [head] + content + [END_ID]
            if is_asst:
                seg_mask = [0] + [1] * (len(content) + 1)
            else:
                seg_mask = [0] * len(seg)
            ids += seg
            mask += seg_mask
        return ids, mask
