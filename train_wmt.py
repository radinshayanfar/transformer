import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm
import argparse
from nltk.translate.bleu_score import corpus_bleu

from transformer import Transformer
from utils import pad_and_mask, tensor_to_sequence, ids_to_tokens, save_args, set_seed, load_checkpoint, save_checkpoint
from preprocess import write_wmt_to_file, train_bpe


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

        input_ids, input_attention_mask = pad_and_mask(src_enc, tokenizer, pad_token)
        labels, labels_attention_mask = pad_and_mask(tgt_enc, tokenizer, pad_token)

        return {
            "source_ids": input_ids,
            "source_attention_mask": input_attention_mask,
            "target_ids": labels,
            "target_attention_mask": labels_attention_mask,
        }


@torch.no_grad()
def evaluate(model, test_dataloader, device, generation_eval=False):
    model.eval()
    
    loss_fn = nn.CrossEntropyLoss(reduction="none")  # Set reduction to 'none' to get element-wise loss
    running_loss_sum = 0.0

    generations_refs = []
    generations_hyps = []
    for batch in tqdm(test_dataloader, desc="Evaluating"):
        source_ids = batch["source_ids"].to(device)
        source_attention_mask = batch["source_attention_mask"].to(device)
        target_ids = batch["target_ids"].to(device)
        target_attention_mask = batch["target_attention_mask"].to(device)

        output = model(
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

        running_loss_sum += loss.item()

        if generation_eval:
            dec_tokens, dec_mask = transformer.generate(
                tokenizer.token_to_id("[EOS]"),
                tokenizer.token_to_id("[PAD]"),
                encoder_x=source_ids,
                decoder_x=target_ids[:, :1],  # keeping the first token
                enc_pad_mask=source_attention_mask,
                dec_pad_mask=target_attention_mask[:, :1],
            )

            generated_ids = tensor_to_sequence(dec_tokens, dec_mask)
            target_ids_seq = tensor_to_sequence(target_ids, target_attention_mask)
            generations_hyps.extend([[ids_to_tokens(seq, tokenizer)] for seq in target_ids_seq])
            generations_refs.extend([ids_to_tokens(seq, tokenizer) for seq in generated_ids])
    
    model.train()

    loss = running_loss_sum / len(test_dataloader)

    if generation_eval:
        bleu = corpus_bleu(
            generations_hyps,
            generations_refs
        )
        return loss, bleu

    return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser("train translation model on WMT")
    parser.add_argument("--lang_pair", "-p", required=True, type=str, help="WMT dataset language pair")
    parser.add_argument("--source", "-s", type=str, help="source language. If None, defaults to first in pair")
    parser.add_argument("--target", "-t", type=str, help="target language. If None, defaults to second in pair")
    parser.add_argument("--output_dir", "-o", default="wmt", type=str, help="output directory")
    parser.add_argument("--n_vocab", default=30_000, type=int, help="model vocab size")
    parser.add_argument("--max_length", default=256, type=int, help="max model context length")
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("--eval_steps", default=400_000, type=int, help="number of steps between evaluations")
    parser.add_argument("--learning_rate", "--lr", default=1e-4, type=float, help="adam learning rate")
    parser.add_argument("--device", "-d", type=str, help="pytorch device")
    parser.add_argument("--skip_prep", action="store_true", help="skip preprocessing and tokenizer training - this will OVERWRITE existing files!")
    parser.add_argument("--load_checkpoint", action="store_true", help="load model checkpoint from output_dir if exists")

    args = parser.parse_args()

    set_seed(0)  # for reproducibility

    os.makedirs(args.output_dir, exist_ok=args.skip_prep)
    save_args(args, os.path.join(args.output_dir, "args.txt"))

    tokenizer_filepath = os.path.join(args.output_dir, "bpe_tokenizer.json")
    if not args.skip_prep:
        filepath = write_wmt_to_file("train", args.lang_pair, dirpath=args.output_dir)
        train_bpe([filepath], args.n_vocab, tokenizer_filepath)

    tokenizer = Tokenizer.from_file(tokenizer_filepath)
    tokenizer.enable_truncation(max_length=args.max_length)

    train_dataset = WMT14TranslationDataset(
        split="train",
        pair=args.lang_pair,
        source_lang=args.source,
        target_lang=args.target,
    )
    val_dataset = WMT14TranslationDataset(
        split="validation",
        pair=args.lang_pair,
        source_lang=args.source,
        target_lang=args.target,
    )
    test_dataset = WMT14TranslationDataset(
        split="test",
        pair=args.lang_pair,
        source_lang=args.source,
        target_lang=args.target,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: WMT14TranslationDataset.collate_fn(x, tokenizer)
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: WMT14TranslationDataset.collate_fn(x, tokenizer)
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: WMT14TranslationDataset.collate_fn(x, tokenizer)
    )

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():  # for mac
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    transformer = Transformer(6, 8, 512, 64, 2048, tokenizer.get_vocab_size(), args.max_length, arch="both")
    transformer = transformer.to(device)

    loss_fn = nn.CrossEntropyLoss(reduction="none", label_smoothing=0.1)  # Set reduction to 'none' to get element-wise loss
    optim = torch.optim.Adam(transformer.parameters(), lr=args.learning_rate)

    checkpoint_path = os.path.join(args.output_dir, f"checkpoint_{args.source}-{args.target}.pt")
    start_batch = None
    if args.load_checkpoint:
        start_batch = load_checkpoint(checkpoint_path, transformer, optim, device)
        print(f"Resuming training from batch {start_batch}...")

    history_log = []
    eval_loss = None

    for i, batch in enumerate(pbar := tqdm(train_dataloader)):
        if start_batch and i < start_batch:
            continue
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

            print(f"Saving checkpoint to {checkpoint_path}...")
            save_checkpoint(checkpoint_path, transformer, optim, i+1)
        
        pbar.set_description(f"loss: {loss.item():.3f}")
    
    test_loss, test_bleu = evaluate(transformer, test_dataloader, device, generation_eval=True)
    print(f"Test loss: {test_loss:.3f}, Test BLEU: {test_bleu:.3f}")

    torch.save(transformer.state_dict(), os.path.join(args.output_dir, f"wmt_{args.source}-{args.target}.pt"))
    with open(os.path.join(args.output_dir, f"history_{args.source}-{args.target}.json"), 'w') as fp:
        json.dump(history_log, fp)
    with open(os.path.join(args.output_dir, f"results_{args.source}-{args.target}.json"), 'w') as fp:
        json.dump({
            "test_loss": test_loss,
            "test_bleu": test_bleu,
            "val_loss": history_log[-1]["val_loss"],
        }, fp)
