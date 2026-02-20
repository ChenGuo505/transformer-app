from __future__ import annotations

import csv
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch
from torch.utils.data import DataLoader, Dataset


PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


@dataclass(frozen=True)
class TranslationExample:
    src_text: str
    tgt_text: str


def basic_tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)


class Vocabulary:
    def __init__(self, stoi: dict[str, int], itos: list[str]) -> None:
        self.stoi = stoi
        self.itos = itos
        self.pad_id = self.stoi[PAD_TOKEN]
        self.bos_id = self.stoi[BOS_TOKEN]
        self.eos_id = self.stoi[EOS_TOKEN]
        self.unk_id = self.stoi[UNK_TOKEN]

    @classmethod
    def build(
        cls,
        tokenized_texts: Iterable[Sequence[str]],
        min_freq: int = 2,
        max_size: int | None = 20000,
    ) -> "Vocabulary":
        counter: Counter[str] = Counter()
        for tokens in tokenized_texts:
            counter.update(tokens)

        words = [w for w, c in counter.items() if c >= min_freq]
        words.sort(key=lambda w: (-counter[w], w))
        if max_size is not None:
            words = words[: max(0, max_size - len(SPECIAL_TOKENS))]

        itos = SPECIAL_TOKENS + words
        stoi = {token: idx for idx, token in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    def encode(
        self, tokens: Sequence[str], add_bos: bool = False, add_eos: bool = False
    ) -> list[int]:
        ids = [self.stoi.get(token, self.unk_id) for token in tokens]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, token_ids: Sequence[int]) -> list[str]:
        return [self.itos[idx] if 0 <= idx < len(self.itos) else UNK_TOKEN for idx in token_ids]

    def __len__(self) -> int:
        return len(self.itos)


def load_iwslt_pairs(
    csv_path: str | Path,
    lp: str = "en-de",
    src_column: str = "src",
    tgt_column: str = "ref",
    min_quality: int | None = None,
) -> list[TranslationExample]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    examples: list[TranslationExample] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_cols = {"lp", src_column, tgt_column}
        if reader.fieldnames is None or not required_cols.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"CSV must contain columns {sorted(required_cols)}, got {reader.fieldnames}"
            )

        for row in reader:
            if row["lp"] != lp:
                continue
            if min_quality is not None:
                raw_score = row.get("raw")
                if raw_score is None or not raw_score.strip():
                    continue
                try:
                    if int(raw_score) < min_quality:
                        continue
                except ValueError:
                    continue

            src = (row.get(src_column) or "").strip()
            tgt = (row.get(tgt_column) or "").strip()
            if not src or not tgt:
                continue
            examples.append(TranslationExample(src_text=src, tgt_text=tgt))

    if not examples:
        raise ValueError(f"No rows found for lp='{lp}' in {csv_path}")
    return examples


class TranslationDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[TranslationExample],
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
        tokenizer: Callable[[str], list[str]] = basic_tokenize,
        max_src_len: int | None = None,
        max_tgt_len: int | None = None,
    ) -> None:
        self.examples = list(examples)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        src_tokens = self.tokenizer(ex.src_text)
        tgt_tokens = self.tokenizer(ex.tgt_text)

        if self.max_src_len is not None:
            src_tokens = src_tokens[: self.max_src_len]
        if self.max_tgt_len is not None:
            tgt_tokens = tgt_tokens[: self.max_tgt_len]

        src_ids = self.src_vocab.encode(src_tokens, add_bos=True, add_eos=True)
        tgt_full = self.tgt_vocab.encode(tgt_tokens, add_bos=True, add_eos=True)

        tgt_in_ids = tgt_full[:-1]
        tgt_out_ids = tgt_full[1:]

        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_in_ids": torch.tensor(tgt_in_ids, dtype=torch.long),
            "tgt_out_ids": torch.tensor(tgt_out_ids, dtype=torch.long),
        }


def _pad_sequences(sequences: Sequence[torch.Tensor], pad_id: int) -> torch.Tensor:
    max_len = max(seq.size(0) for seq in sequences)
    out = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(sequences):
        out[i, : seq.size(0)] = seq
    return out


def make_collate_fn(src_pad_id: int, tgt_pad_id: int):
    def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        src_ids = _pad_sequences([x["src_ids"] for x in batch], src_pad_id)
        tgt_in_ids = _pad_sequences([x["tgt_in_ids"] for x in batch], tgt_pad_id)
        tgt_out_ids = _pad_sequences([x["tgt_out_ids"] for x in batch], tgt_pad_id)

        return {
            "src_ids": src_ids,
            "src_key_padding_mask": src_ids.eq(src_pad_id),
            "tgt_in_ids": tgt_in_ids,
            "tgt_out_ids": tgt_out_ids,
            "tgt_key_padding_mask": tgt_in_ids.eq(tgt_pad_id),
        }

    return collate_fn


def _train_val_split(
    examples: Sequence[TranslationExample],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[TranslationExample], list[TranslationExample]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    examples = list(examples)
    random.Random(seed).shuffle(examples)
    val_size = int(len(examples) * val_ratio)
    val_set = examples[:val_size]
    train_set = examples[val_size:]
    return train_set, val_set


def _grouped_train_val_split(
    examples: Sequence[TranslationExample],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[TranslationExample], list[TranslationExample]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")

    groups: dict[tuple[str, str], list[TranslationExample]] = {}
    for ex in examples:
        key = (ex.src_text, ex.tgt_text)
        groups.setdefault(key, []).append(ex)

    group_keys = list(groups.keys())
    random.Random(seed).shuffle(group_keys)

    val_group_size = int(len(group_keys) * val_ratio)
    val_group_keys = set(group_keys[:val_group_size])

    train_set: list[TranslationExample] = []
    val_set: list[TranslationExample] = []
    for key, grouped_examples in groups.items():
        if key in val_group_keys:
            val_set.extend(grouped_examples)
        else:
            train_set.extend(grouped_examples)

    return train_set, val_set


def build_en_de_dataloaders(
    csv_path: str | Path,
    batch_size: int = 32,
    min_freq: int = 2,
    max_vocab_size: int = 20000,
    max_src_len: int | None = 128,
    max_tgt_len: int | None = 128,
    val_ratio: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
    min_quality: int | None = None,
    unique_pairs_only: bool = True,
) -> dict[str, object]:
    examples = load_iwslt_pairs(
        csv_path=csv_path,
        lp="en-de",
        src_column="src",
        tgt_column="ref",
        min_quality=min_quality,
    )
    if unique_pairs_only:
        # Keep only one copy of each (src, tgt) pair to reduce memorization bias.
        examples = list(dict.fromkeys(examples))
    train_examples, val_examples = _train_val_split(
        examples,
        val_ratio=val_ratio,
        seed=seed,
    )

    src_tokens = (basic_tokenize(ex.src_text) for ex in train_examples)
    tgt_tokens = (basic_tokenize(ex.tgt_text) for ex in train_examples)
    src_vocab = Vocabulary.build(src_tokens, min_freq=min_freq, max_size=max_vocab_size)
    tgt_vocab = Vocabulary.build(tgt_tokens, min_freq=min_freq, max_size=max_vocab_size)

    train_ds = TranslationDataset(
        train_examples,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
    )
    val_ds = TranslationDataset(
        val_examples,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
    )

    collate_fn = make_collate_fn(src_pad_id=src_vocab.pad_id, tgt_pad_id=tgt_vocab.pad_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
    }


if __name__ == "__main__":
    artifacts = build_en_de_dataloaders(
        csv_path=Path(__file__).parent / "dataset" / "iwslt2023da.csv",
        batch_size=8,
    )
    print(f"train_size={artifacts['train_size']}, val_size={artifacts['val_size']}")
    print(
        f"src_vocab={len(artifacts['src_vocab'])}, tgt_vocab={len(artifacts['tgt_vocab'])}"
    )
    batch = next(iter(artifacts["train_loader"]))
    print(
        "batch shapes:",
        batch["src_ids"].shape,
        batch["tgt_in_ids"].shape,
        batch["tgt_out_ids"].shape,
    )
