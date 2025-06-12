import os
from datasets import load_dataset
from tqdm import tqdm

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase, NFD, StripAccents, Sequence


def write_wmt_to_file(split, pair, dirpath="."):
    filepath = os.path.join(dirpath, f"wmt_{pair}_{split}.txt")
    if os.path.exists(filepath):
        print("Skipping dataset prep - file already exists")
        return
    
    ds = load_dataset("wmt/wmt14", pair, split=split)
    lang1, lang2 = pair.split("-")

    os.makedirs(dirpath, exist_ok=True)
    with open(filepath, 'w') as fp:
        for row in tqdm(ds):
            sent1 = row["translation"][lang1]
            sent2 = row["translation"][lang2]
            fp.write(sent1)
            fp.write("\n")
            fp.write(sent2)
            fp.write("\n")
    
    return filepath


def train_bpe(corpus_files, vocab_size, save_path="bpe_tokenizer.json", unk_token="[UNK]", special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[EOS]"]):
    # 1. Initialize a BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token=unk_token))

    # 2. Set normalizer and pre-tokenizer
    tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()

    # 3. Define a trainer with special tokens
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )

    # 4. Train the tokenizer on your corpus
    files = corpus_files  # Replace with your actual file(s)
    tokenizer.train(files, trainer)

    # 5. (Optional) Post-processing to make it behave like a standard BERT tokenizer
    # tokenizer.post_processor = TemplateProcessing(
    #     single="[CLS] $A [SEP]",
    #     pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    #     special_tokens=[("[CLS]", tokenizer.token_to_id("[CLS]")), ("[SEP]", tokenizer.token_to_id("[SEP]"))],
    # )

    # 6. Save the trained tokenizer
    tokenizer.save(save_path)
