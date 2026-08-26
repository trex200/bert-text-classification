# BERT-style text classification

From-scratch **Transformer encoder** + **[CLS]** head for binary sentiment.

English movie-review sentences → `negative` / `positive`. Blocks written by hand — no Hugging Face `BertForSequenceClassification`, no `nn.Transformer`.

> Status: **v1 landed.** Data pipeline, encoder + CLS, 10-epoch train (avg loss 0.49 → 0.067), 6.95M params, sample preds.

## Language + task

| | |
|---|---|
| Language | **English** |
| Task | binary text classification (sentiment) |
| Labels | `0` = negative, `1` = positive |
| Domain | movie-review phrases (SST-2 / GLUE) |

## What's in here

| Step | Where |
|---|---|
| Normalize + vocab + CLS tensors + Dataset | [`src/data.py`](src/data.py) |
| Attention, PE, encoder, `Bert` CLS head | [`src/model.py`](src/model.py) |
| Train loop | [`src/train.py`](src/train.py) |
| Sentence → sentiment | [`src/predict.py`](src/predict.py) |
| Data notes | [`data/README.md`](data/README.md) |
| 10-epoch loss | [`results/training_notes.md`](results/training_notes.md) |
| Sample preds | [`results/sample_preds.md`](results/sample_preds.md) |

## Data

`sst2_train.csv`, **67349** English rows × 2 columns (`sentence`, `label`). Word vocab after normalize: **14819**. CSV is not committed. See [`data/README.md`](data/README.md).

Same SST-2 train split as [Hugging Face `stanfordnlp/sst2`](https://huggingface.co/datasets/stanfordnlp/sst2) (67349 / 872 / 1821). Original: [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/).

| token | id |
|---|---|
| PAD | 0 |
| CLS | 1 |
| UNK | 2 |

Each row is `[CLS] + word ids`, clipped/padded to `MAX_LENGTH = 64`. Batch = 32.

## Model

| | |
|---|---|
| d_model | 256 |
| layers | 4 encoder (no decoder) |
| heads | 8 |
| d_ff | 1024 |
| dropout | 0.1 |
| PE | sinusoidal buffer, max_len 64 |
| classes | 2 |
| params | 6,953,218 |

Post-norm residual blocks. GELU FFN. Pad mask `[B, 1, 1, S]`.

```
ids [B, 64] → embed + PE → encoder × 4 → hidden[:, 0] (CLS) → Linear(256, 2) → [B, 2]
```

## Train

AdamW `2e-4`, `weight_decay=0.01`, `CrossEntropyLoss()` on 2 class scores, grad clip `1.0`.

| epoch | avg loss |
|---|---|
| 1 | 0.4894 |
| 5 | 0.1120 |
| 10 | 0.0673 |

~42–46s / epoch at ~46–49 it/s on CUDA. 2105 batches/epoch.

## Sample (softmax on CLS)

```
EN: we all will fail
→ negative  (0.99)

EN: painting a girl
→ positive  (0.82)

EN: its harsh but its true.
→ negative  (0.96)
```

## License

MIT
