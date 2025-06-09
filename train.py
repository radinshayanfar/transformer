import torch
import torch.nn as nn
import datasets
from datasets import load_dataset
from tokenizers import Tokenizer

from transformer import Transformer

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


class WMT14TranslationDataset(Dataset):
    def __init__(self, split, pair, source_lang=None, target_lang=None):
        """
        Args:
            split (str): Split name ('train', 'validation', or 'test')
            pair (str): Language pair string, e.g., "en-de"
            source_lang (str, optional): Source language. If None, defaults to first in pair.
            target_lang (str, optional): Target language. If None, defaults to second in pair.
        """
        self.dataset = load_dataset("wmt/wmt14", pair, split=split)
        self.source_lang, self.target_lang = pair.split("-")
        if source_lang:
            self.source_lang = source_lang
        if target_lang:
            self.target_lang = target_lang

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]["translation"]
        return {
            "source": row[self.source_lang],
            "target": row[self.target_lang],
        }


def collate_fn(batch, tokenizer, pad_token="[PAD]"):
    # batch: list of dicts with "source", "target" (both raw strings)
    src_texts = [item["source"] for item in batch]
    tgt_texts = [item["target"] for item in batch]

    src_enc = tokenizer.encode_batch(src_texts)
    tgt_enc = tokenizer.encode_batch(tgt_texts)

    def pad_and_mask(encodings):
        ids = [torch.tensor(e.ids, dtype=torch.long) for e in encodings]
        max_len = max(len(x) for x in ids)
        pad_id = tokenizer.token_to_id(pad_token)
        input_ids = torch.stack([
            torch.cat([x, torch.full((max_len - len(x),), pad_id, dtype=torch.long)])
            for x in ids
        ])
        attention_mask = torch.stack([
            torch.cat([torch.ones(len(x), dtype=torch.long),
                       torch.zeros(max_len - len(x), dtype=torch.long)])
            for x in ids
        ])
        return input_ids, attention_mask

    input_ids, input_attention_mask = pad_and_mask(src_enc)
    labels, labels_attention_mask = pad_and_mask(tgt_enc)

    return {
        "input_ids": input_ids,
        "input_attention_mask": input_attention_mask,
        "labels": labels,
        "labels_attention_mask": labels_attention_mask,
    }


if __name__ == "__main__":
    tokenizer = Tokenizer.from_file("wmt_bpe_tokenizer.json")
    dataset = WMT14TranslationDataset(
        split="train",
        pair="de-en",
        source_lang="en",
        target_lang="de"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, tokenizer)
    )

    transformer = Transformer(6, 8, 512, 64, 2048, 30_000, 1024, arch="both")
    for batch in dataloader:
        print(batch["input_ids"].shape)
        print(batch["input_attention_mask"].shape)
        print(batch["labels"].shape)
        print(batch["labels_attention_mask"].shape)

        print(batch["input_ids"])
        print(batch["input_attention_mask"])
        
        output = transformer(
            batch["input_ids"],
            batch["labels"],
            padding_mask_x=batch["input_attention_mask"],
            padding_mask_y=batch["labels_attention_mask"],
        )
        print(output.shape)

        break
