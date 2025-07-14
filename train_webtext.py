import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
from transformers import GPT2Tokenizer
from tqdm import tqdm
import argparse

from transformer import Transformer
from utils import save_args, set_seed, load_checkpoint, save_checkpoint


class WebTextDataset(Dataset):
    def __init__(self, path):
        """
        Args:
            path (str): Path to the dataset on disk.
        """
        self.dataset = load_from_disk(path)["train"]

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
        texts = ["<|endoftext|>" + text + "<|endoftext|>" for text in texts]

        txt_enc = tokenizer(texts, truncation=True, padding=True, max_length=tokenizer.model_max_length, return_tensors="pt")

        texts, attention_mask = txt_enc["input_ids"], txt_enc["attention_mask"]

        return {
            "texts": texts,
            "attention_mask": attention_mask,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser("train text generation model on WebText")
    parser.add_argument("--ds_path", "-p", required=True, type=str, help="dataset path on disk")
    parser.add_argument("--output_dir", "-o", default="webtext", type=str, help="output directory")
    parser.add_argument("--max_length", default=256, type=int, help="max model context length")
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("--learning_rate", "--lr", default=1e-5, type=float, help="adam learning rate")
    parser.add_argument("--checkpoint_steps", default=400_000, type=int, help="number of steps between checkpoint saves")
    parser.add_argument("--device", "-d", type=str, help="pytorch device")
    parser.add_argument("--override", action="store_true", help="override output directory if it exists")
    parser.add_argument("--load_checkpoint", action="store_true", help="load model checkpoint from output_dir if exists")

    args = parser.parse_args()

    set_seed(0)  # for reproducibility

    os.makedirs(args.output_dir, exist_ok=args.override)
    save_args(args, os.path.join(args.output_dir, "args.txt"))

    # we use GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.model_max_length = args.max_length

    train_dataset = WebTextDataset(path=args.ds_path)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: WebTextDataset.collate_fn(x, tokenizer)
    )

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():  # for mac
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    transformer = Transformer(12, 8, 768, 64, 2048, tokenizer.total_vocab_size, args.max_length, arch="decoder")
    transformer = transformer.to(device)

    loss_fn = nn.CrossEntropyLoss(reduction="none", label_smoothing=0.1)  # Set reduction to 'none' to get element-wise loss
    optim = torch.optim.Adam(transformer.parameters(), lr=args.learning_rate)

    checkpoint_path = os.path.join(args.output_dir, "checkpoint.pt")
    start_batch = None
    if args.load_checkpoint:
        start_batch = load_checkpoint(checkpoint_path, transformer, optim, device)
        print(f"Resuming training from batch {start_batch}...")

    history_log = []
    eval_loss = None

    for i, batch in enumerate(pbar := tqdm(train_dataloader)):
        if start_batch and i < start_batch:
            continue
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
        
        if i % (args.checkpoint_steps // args.batch_size) == 0:
            history_log.append({
                "step": i * args.batch_size,
                "loss": loss.item(),
            })

            print(f"Saving checkpoint to {checkpoint_path}...")
            save_checkpoint(checkpoint_path, transformer, optim, i+1)
        
        pbar.set_description(f"loss: {loss.item():.3f}")

    torch.save(transformer.state_dict(), os.path.join(args.output_dir, "web.pt"))
    with open(os.path.join(args.output_dir, f"history.json"), 'w') as fp:
        json.dump(history_log, fp)
    with open(os.path.join(args.output_dir, f"results.json"), 'w') as fp:
        json.dump({
            "loss": history_log[-1]["loss"],
        }, fp)
