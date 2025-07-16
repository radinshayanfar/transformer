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

**📊 Results**

| Language Pair | Perplexity   | BLEU Score   | Reference BLEU (Vaswani et al., 2017)  |
|---------------|--------------|--------------|----------------------------------------|
| EN → DE       | 2.30         | 22.6         | 27.3                                   |
| DE → EN       | 2.19         | 25.6         | —                                      |
| EN → FR       | 1.33         | 39.2         | 41.0                                   |
| FR → EN       | 1.55         | 36.4         | —                                      |

This result is achieved under significantly minimal tuning:
- No learning rate scheduling or warm-up tuning
- Naive greedy decoding (vs. beam search with length penalty in the paper)
- Only 1 training epoch
- No checkpoint averaging or fine-grained validation

Despite the simplicity, the implementation reaches a strong baseline close to the original paper’s result, validating the correctness and robustness of the codebase.

---

## ✍️ Text Generation (WikiText)

The repository also supports training a decoder-only Transformer for text generation, using the WikiText-103-raw-v1 dataset. The architecture closely follows the smallest GPT-2 configuration described in the [original paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), but it is trained directly on WikiText, not via zero-shot transfer.

To train the model:
```bash
python train_wikitext.py
```

**📊 Results**
- Test Perplexity: 4.18
- (GPT-2 was trained on WebText and evaluated in a zero-shot setup, therefore, not directly comparable here)

This experiment demonstrates the extensibility of the core Transformer implementation to causal language modeling tasks. The model learns to generate Wikipedia-like English text purely from scratch training.

---

## 🧾 Text Generation (OpenWebText, GPT-2)

This task trains a GPT-2-style decoder-only Transformer on the OpenWebText dataset — a public replication of the WebText corpus used in the original GPT-2 paper:
[Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

**🛠 Dataset Preparation**

Download and preprocess the dataset with:

```bash
python prepare_webtext.py --save_dir "<dataset_save_dir>"
```

**🚀 Training**

Once prepared, train the model using:

```bash
python train_webtext.py --ds_path "<dataset_save_dir>"
```

**🔧 Notes**
- Uses the GPT-2 tokenizer (via HuggingFace) for simplicity and speed.
- Trains only on the first 256 tokens of each example (for efficiency).
- Architecture mirrors the smallest GPT-2 model described in the original paper.
- Focuses on proof-of-concept and scalability, not direct reproduction of results.

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

