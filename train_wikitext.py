import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm
import argparse

from transformer import Transformer
from utils import pad_and_mask, save_args
from preprocess import write_wikitext_to_file, train_bpe


class WikiTextDataset(Dataset):
    def __init__(self, split, subset):
        """
        Args:
            split (str): Split name ('train', 'validation', or 'test')
            subset (str): Subset name (e.g., "wikitext-103-raw-v1" or "wikitext-2-v1")
        """
        self.dataset = load_dataset("Salesforce/wikitext", subset, split=split)
        self.dataset = self.dataset.filter(lambda x: x["text"] != "")  # only training on single paragraphs

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]
        return {
            "text": text,
        }

    @staticmethod
    def collate_fn(batch, tokenizer, pad_token="[PAD]"):
        # batch: list of dicts with "text" (in raw string)
        texts = [item["text"] for item in batch]
        texts = ["[BOS]" + text + "[EOS]" for text in texts]

        txt_enc = tokenizer.encode_batch(texts)

        texts, attention_mask = pad_and_mask(txt_enc, tokenizer, pad_token)

        return {
            "texts": texts,
            "attention_mask": attention_mask,
        }


@torch.no_grad()
def evaluate(model, test_dataloader, device):
    model.eval()
    
    loss_fn = nn.CrossEntropyLoss(reduction="none")  # Set reduction to 'none' to get element-wise loss
    running_loss_sum = 0.0

    for batch in tqdm(test_dataloader, desc="Evaluating"):
        text_ids = batch["texts"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        output = model(
            decoder_x=text_ids[:, :-1],  # dropping EOS token as input
            dec_pad_mask=attention_mask[:, :-1],
        )

        labels, labels_mask = text_ids[:, 1:], attention_mask[:, 1:]  # dropping BOS for labels
        # labels = nn.functional.one_hot(labels, num_classes=tokenizer.get_vocab_size()).to(torch.float)
        labels = labels.long()  # this seems to be more efficient

        labels = labels.reshape(-1)
        output = output.reshape(-1, output.shape[-1])
        labels_mask = labels_mask.reshape(-1)

        loss = loss_fn(output, labels)
        loss = loss * labels_mask
        loss = loss.sum() / labels_mask.sum()

        running_loss_sum += loss.item()

    model.train()

    loss = running_loss_sum / len(test_dataloader)

    return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser("train text generation model on WikiText (wikipedia dump)")
    parser.add_argument("--subset", default="wikitext-103-raw-v1", type=str, help="WikiText subset to use (e.g., 'wikitext-103-raw-v1' or 'wikitext-2-v1')")
    parser.add_argument("--output_dir", "-o", default="wikitext", type=str, help="output directory")
    parser.add_argument("--n_vocab", default=30_000, type=int, help="model vocab size")
    parser.add_argument("--max_length", default=256, type=int, help="max model context length")
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("--eval_steps", default=400_000, type=int, help="number of steps between evaluations")
    parser.add_argument("--learning_rate", "--lr", default=1e-4, type=float, help="adam learning rate")
    parser.add_argument("--device", "-d", type=str, help="pytorch device")
    parser.add_argument("--skip_prep", action="store_true", help="skip preprocessing and tokenizer training - this will OVERWRITE existing files!")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=args.skip_prep)
    save_args(args, os.path.join(args.output_dir, "args.txt"))

    tokenizer_filepath = os.path.join(args.output_dir, "bpe_tokenizer.json")
    if not args.skip_prep:
        filepath = write_wikitext_to_file("train", args.subset, dirpath=args.output_dir)
        train_bpe([filepath], args.n_vocab, tokenizer_filepath)

    tokenizer = Tokenizer.from_file(tokenizer_filepath)
    tokenizer.enable_truncation(max_length=args.max_length)

    train_dataset = WikiTextDataset(
        split="train",
        subset=args.subset,
    )
    val_dataset = WikiTextDataset(
        split="validation",
        subset=args.subset,
    )
    test_dataset = WikiTextDataset(
        split="test",
        subset=args.subset,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: WikiTextDataset.collate_fn(x, tokenizer)
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: WikiTextDataset.collate_fn(x, tokenizer)
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: WikiTextDataset.collate_fn(x, tokenizer)
    )

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():  # for mac
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    transformer = Transformer(12, 8, 768, 64, 2048, tokenizer.get_vocab_size(), args.max_length, arch="decoder")
    transformer = transformer.to(device)

    loss_fn = nn.CrossEntropyLoss(reduction="none", label_smoothing=0.1)  # Set reduction to 'none' to get element-wise loss
    optim = torch.optim.Adam(transformer.parameters(), lr=args.learning_rate)

    history_log = []
    eval_loss = None

    for i, batch in enumerate(pbar := tqdm(train_dataloader)):
        text_ids = batch["texts"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        output = transformer(
            decoder_x=text_ids[:, :-1],  # dropping EOS token as input
            dec_pad_mask=attention_mask[:, :-1],
        )

        labels, labels_mask = text_ids[:, 1:], attention_mask[:, 1:]  # dropping BOS for labels
        # labels = nn.functional.one_hot(labels, num_classes=tokenizer.get_vocab_size()).to(torch.float)
        labels = labels.long()  # this seems to be more efficient

        labels = labels.reshape(-1)
        output = output.reshape(-1, output.shape[-1])
        labels_mask = labels_mask.reshape(-1)

        loss = loss_fn(output, labels)
        loss = loss * labels_mask.detach()
        loss = loss.sum() / labels_mask.sum()

        loss.backward()

        optim.step()
        optim.zero_grad()

        if i % (args.eval_steps // args.batch_size) == 0:
            print(f"Evaluating at step {i * args.batch_size}...")
            val_loss = evaluate(transformer, val_dataloader, device)
            print(f"Validation loss: {val_loss:.3f}")

            history_log.append({
                "epoch": 0,
                "step": i * args.batch_size,
                "loss": loss.item(),
                "val_loss": val_loss,
            })
        
        pbar.set_description(f"loss: {loss.item():.3f}")

    test_loss = evaluate(transformer, test_dataloader, device)
    print(f"Test loss: {test_loss:.3f}")

    torch.save(transformer.state_dict(), os.path.join(args.output_dir, "wiki.pt"))
    with open(os.path.join(args.output_dir, f"history.json"), 'w') as fp:
        json.dump(history_log, fp)
    with open(os.path.join(args.output_dir, f"results.json"), 'w') as fp:
        json.dump({
            "test_loss": test_loss,
            "val_loss": history_log[-1]["val_loss"],
        }, fp)
