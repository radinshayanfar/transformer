# Tests

This directory contains unit tests for the transformer repository.

## Test Files

### `test_tokenizer.py`
Tests for tokenizer training functionality, including:
- BPE tokenizer training from `preprocess.py`
- Tokenizer loading and usage
- Special token handling
- Vocabulary size validation
- Multi-file corpus support

## Running Tests

To run all tests:
```bash
python test_tokenizer.py
```

To run with verbose output:
```bash
python test_tokenizer.py -v
```

To run with pytest (if installed):
```bash
pytest test_tokenizer.py -v
```

## Test Coverage

The test suite covers:
- ✅ Tokenizer training (`train_bpe` function)
- ✅ Tokenizer file creation
- ✅ Tokenizer loading and validation
- ✅ Special token handling
- ✅ Text encoding/decoding
- ✅ Vocabulary size constraints
- ✅ Custom special tokens
- ✅ Multi-file corpus training

## Requirements

The tests require the following packages:
- `tokenizers` - For BPE tokenizer functionality
- `datasets` - For dataset loading utilities

Install test dependencies:
```bash
pip install tokenizers datasets
```

## Notes

Some tests are skipped if optional dependencies (torch, openwebtext) are not available. These tests are marked with `@unittest.skipIf` decorators.
