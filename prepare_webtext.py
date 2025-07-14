import argparse
from openwebtext import Openwebtext

parser = argparse.ArgumentParser(description="Download and save OpenWebText dataset")
parser.add_argument("--save_dir", required=True, 
                    help="Directory to save the processed dataset")
parser.add_argument("--cache_dir", default=None,
                    help="Cache directory for downloading (optional)")

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
