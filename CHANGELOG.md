# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Custom BPE tokenizer training support for WebText pipeline
  - Added `--prepare_tokenizer_corpus` flag to `prepare_webtext.py` to write dataset text to file for tokenizer training
  - Added `--use_custom_tokenizer` flag to `train_webtext.py` to enable custom BPE tokenizer instead of pre-trained GPT-2 tokenizer
  - Added `--vocab_size` parameter to `train_webtext.py` for configuring tokenizer vocabulary size (default: 50,257 to match GPT-2)
  - Added `--skip_tokenizer_training` flag to `train_webtext.py` to reuse existing tokenizer files
  - Modified `WebTextDataset.collate_fn()` to support both custom BPE tokenizers and GPT-2 tokenizers
  - Added validation warning for unusually small vocabulary sizes (< 256)
- Test suite for tokenizer training functionality
  - Added `test_tokenizer.py` with comprehensive unit tests for BPE tokenizer training
  - Tests cover tokenizer creation, loading, special tokens, encoding, and vocabulary size validation
  - Added `TESTS.md` documentation for running and understanding tests

### Changed
- Updated `train_webtext.py` to support custom tokenizer training while maintaining backward compatibility (defaults to GPT-2 tokenizer)
- Enhanced `prepare_webtext.py` to optionally write tokenizer training corpus

### Documentation
- Updated README.md with examples for preparing tokenizer corpus
- Updated README.md with examples for training with custom BPE tokenizer
- Updated README.md notes section to reflect new tokenizer options
- Added testing section to README.md with instructions for running tests
- Added TESTS.md with detailed test documentation

## [0.1.0] - Initial Release

### Added
- Full Transformer architecture implementation from scratch based on "Attention Is All You Need" paper
  - Encoder-Decoder architecture (original Transformer)
  - Encoder-only architecture (BERT-style)
  - Decoder-only architecture (GPT-style)
- WMT translation task support (`train_wmt.py`)
  - Training on WMT14 dataset with multiple language pairs (EN↔DE, EN↔FR)
  - Custom BPE tokenizer training with HuggingFace Tokenizers
  - Evaluation with perplexity and BLEU score metrics
- WikiText text generation task support (`train_wikitext.py`)
  - Training on WikiText-103-raw-v1 dataset
  - Custom BPE tokenizer training
  - GPT-2 style decoder-only architecture
- OpenWebText text generation task support (`train_webtext.py`)
  - Training on OpenWebText dataset
  - Uses pre-trained GPT-2 tokenizer
  - GPT-2 style decoder-only architecture
- Dataset preparation utilities
  - `preprocess.py`: BPE tokenizer training utilities
  - `prepare_webtext.py`: OpenWebText dataset download and preparation
  - `write_wmt_to_file()`: WMT dataset export to text files
  - `write_wikitext_to_file()`: WikiText dataset export to text files
- Training utilities (`utils.py`)
  - Positional encoding
  - Padding and masking utilities
  - Checkpoint saving and loading
  - Seed management for reproducibility
- Core model (`transformer.py`)
  - Multi-head self-attention mechanism
  - Position-wise feed-forward networks
  - Layer normalization
  - Residual connections
  - Configurable architecture (encoder/decoder/both)

### Results
- WMT14 EN→DE: BLEU 22.6, Perplexity 2.30
- WMT14 DE→EN: BLEU 25.6, Perplexity 2.19
- WMT14 EN→FR: BLEU 39.2, Perplexity 1.33
- WMT14 FR→EN: BLEU 36.4, Perplexity 1.55
- WikiText-103: Test Perplexity 4.18

[Unreleased]: https://github.com/radinshayanfar/transformer/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/radinshayanfar/transformer/releases/tag/v0.1.0
