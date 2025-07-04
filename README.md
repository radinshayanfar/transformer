# 🧠 Transformer from Scratch

This repository contains a **from-scratch PyTorch implementation** of the original **Transformer architecture** from [_Attention Is All You Need_](https://arxiv.org/abs/1706.03762).

> ⚠️ No pre-existing code or reference implementations were used. This project is built solely from the original paper and [Jay Alammar’s visual guide](https://jalammar.github.io/illustrated-transformer/).

---

## 🧱 Transformer Architecture

Full Transformer architecture, modularized and extensible:
- ✅ **Encoder–Decoder** (original)
- ✅ **Encoder-only** (e.g., BERT-style)
- ✅ **Decoder-only** (e.g., GPT-style)

Each variant is supported via the `Transformer` class in `transformer.py`.

---

## 🗣️ Translation Task (WMT)

To train a translation model using the WMT dataset (e.g., English to German):

```bash
python train_wmt.py --pair de-en --source en --target de
```

📌 More tasks training script will be coming soon, following the same architectural foundation.

---

## 📂 Code Structure

- `transformer.py`  
  Core Transformer implementation supporting encoder-decoder, encoder-only, and decoder-only variants.

- `train_wmt.py`  
  End-to-end training script for WMT translation — handles dataset loading, batching, masking, training loop, and evaluation.

- `preprocess.py`  
  Preprocessing utilities, including Byte Pair Encoding (BPE) tokenizer training using HuggingFace Tokenizers.

- `utils.py`  
  Miscellaneous helper functions for architecture and training (e.g., positional encodings, padding masks).

---

## 📚 Resources

This implementation is intentionally grounded in theory and built to demonstrate a deep understanding of the architecture. The only two resources referenced are:

1. [_Attention Is All You Need_ (Vaswani et al.)](https://arxiv.org/abs/1706.03762)
2. [Jay Alammar's Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

---

## 🎯 Why This Project?

The goal of this project is to show:
- A deep understanding of Transformer internals
- The ability to implement research-grade models from primary sources
- Practical application to large-scale NLP tasks

