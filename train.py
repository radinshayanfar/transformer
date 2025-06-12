import json
import torch
import torch.nn as nn
import datasets
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm

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


    @staticmethod
    def collate_fn(batch, tokenizer, pad_token="[PAD]"):
        # batch: list of dicts with "source", "target" (both raw strings)
        src_texts = [item["source"] for item in batch]
        tgt_texts = [item["target"] for item in batch]
        tgt_texts = ["[SEP]" + text + "[EOS]" for text in tgt_texts]

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
            "source_ids": input_ids,
            "source_attention_mask": input_attention_mask,
            "target_ids": labels,
            "target_attention_mask": labels_attention_mask,
        }


if __name__ == "__main__":
    tokenizer = Tokenizer.from_file("wmt_bpe_tokenizer.json")
    tokenizer.enable_truncation(max_length=256)
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
        collate_fn=lambda x: WMT14TranslationDataset.collate_fn(x, tokenizer)
    )

    if torch.backends.mps.is_available():  # for mac
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    transformer = Transformer(6, 8, 512, 64, 2048, tokenizer.get_vocab_size(), 1024, arch="both")
    transformer = transformer.to(device)

    loss_fn = nn.CrossEntropyLoss(reduction="none")  # Set reduction to 'none' to get element-wise loss
    optim = torch.optim.Adam(transformer.parameters(), lr=0.0001)

    history_log = []

    for i, batch in enumerate(pbar := tqdm(dataloader)):
        source_ids = batch["source_ids"].to(device)
        source_attention_mask = batch["source_attention_mask"].to(device)
        target_ids = batch["target_ids"].to(device)
        target_attention_mask = batch["target_attention_mask"].to(device)

        output = transformer(
            encoder_x=source_ids,
            decoder_x=target_ids[:, :-1],  # dropping EOS token as input
            enc_pad_mask=source_attention_mask,
            dec_pad_mask=target_attention_mask[:, :-1],
        )

        labels, labels_mask = target_ids[:, 1:], target_attention_mask[:, 1:]  # dropping SEP for labels
        # labels = nn.functional.one_hot(labels, num_classes=tokenizer.get_vocab_size()).to(torch.float)
        labels = labels.long()  # this seems to be more efficient

        labels = labels.reshape(-1)
        output = output.reshape(-1, output.shape[-1])
        labels_mask = labels_mask.reshape(-1)

        loss = loss_fn(output, labels)
        loss = loss * labels_mask
        loss = loss.sum() / labels_mask.sum()

        loss.backward()

        optim.step()
        optim.zero_grad()

        history_log.append({
            "epoch": 0,
            "loss": loss.item()
        })
        pbar.set_description(f"loss: {loss.item():.3f}")

    torch.save(transformer.state_dict(), "wmt_de-en.pt")
    with open("history.json", 'w') as fp:
        json.dump(history_log, fp)
