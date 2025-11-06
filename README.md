# Masked-Word-Prediction-using-Transformers

A compact, from-scratch masked-word prediction project implementing a Transformer architecture (no pretrained prediction models). The repo documents an iterative approach that progressed from LSTM → BiLSTM → Transformer, with custom evaluation using semantic and fluency metrics.

---

## Table of Contents

* Overview
* Highlights
* Model development timeline
* Training & optimization
* Hyperparameters
* Evaluation
* Outputs

---

## Overview

This project explores masked-word prediction using architectures built entirely from scratch. The main goal: evaluate how well a custom Transformer (self-attention) can predict missing tokens compared to simpler recurrent baselines, while running on constrained hardware.

---

## Highlights

* **No pretrained models** used for prediction; all training from scratch.
* **Progression:** LSTM baseline → BiLSTM → Transformer.
* **Custom evaluation:** Cosine similarity (Sentence-BERT) + Perplexity (GPT-2).
* **Resource-aware training:** Mixed precision, batch splitting, and manual checkpointing.

---

## Model development timeline

1. **LSTM baseline**

   * Slow training, weak long-range context capture.

2. **BiLSTM**

   * Improved bidirectional context, still limited on complex sentences.

3. **Transformer**

   * Implemented multi-head self-attention. Better global context, higher accuracy, but increased compute.

4. **Training constraints**

   * GPU and memory limits addressed with precision reduction and batch management.

---

## Training & optimization

Common optimizations used during training:

* Mixed precision (`torch.float16`) to reduce memory.
* Batch splitting / gradient accumulation for effective batch sizes.
* Manual checkpointing and session restarts on Colab/Kaggle.
* Optimizer: **AdamW** for stable convergence.

---

## Hyperparameters (example)

* Embedding dim: 256
* Transformer layers: 4
* Attention heads: 8
* Feedforward dim: 1024
* Dropout: 0.1
* Optimizer: AdamW
* Learning rate: 3e-4 (scheduler recommended)
* Batch size: 32 (effective via gradient accumulation)
* Mixed precision: enabled

> Tune these depending on dataset size and available memory.

---

## Evaluation

Two complementary metrics were used:

* **Cosine Similarity** (Sentence-BERT)

  * Measures semantic closeness between model-predicted word and BERT prediction.
  * Average similarity: **54.7** (dataset-dependent)
  * Note: Penalizes semantically valid but lexically different predictions.

* **Perplexity** (GPT-2)

  * Measures sentence fluency after inserting the predicted token.
  * Average perplexity: **488**
  * Note: Tends to favor common phrasing; not a pure correctness metric.

---

## Outputs

* Predictions and evaluation scores exported as CSVs.
* Merged result files for easy visualization and downstream analysis.
* Saved checkpoints for reproducibility.

---


