import argparse
import os
from openwebtext import Openwebtext
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Download and save OpenWebText dataset")
parser.add_argument("--save_dir", required=True, 
                    help="Directory to save the processed dataset")
parser.add_argument("--cache_dir", default=None,
                    help="Cache directory for downloading (optional)")
parser.add_argument("--prepare_tokenizer_corpus", action="store_true",
                    help="Also write a text file for tokenizer training")

args = parser.parse_args()

# Initialize OpenWebText dataset
openwebtext = Openwebtext(cache_dir=args.cache_dir)
print(openwebtext.info)

# Download and prepare the dataset
openwebtext.download_and_prepare()

# Convert to dataset format and save
ds = openwebtext.as_dataset()
ds.save_to_disk(args.save_dir)

print(f"Dataset saved to: {args.save_dir}")

# Optionally write text file for tokenizer training
if args.prepare_tokenizer_corpus:
    corpus_file = os.path.join(args.save_dir, "tokenizer_corpus.txt")
    print(f"Writing tokenizer corpus to: {corpus_file}")
    
    with open(corpus_file, 'w') as fp:
        for row in tqdm(ds["train"], desc="Writing corpus"):
            text = row["text"].strip()
            if text:  # Only write non-empty texts
                fp.write(text)
                fp.write("\n")
    
    print(f"Tokenizer corpus saved to: {corpus_file}")
