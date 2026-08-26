# BERT-style text classification

Transformer **encoder** + **[CLS]** → 2-way sentiment on SST-2.

> Status: **v1 landed.** 10-epoch train (loss 0.49 → 0.067), 6.95M params, sample preds.

## Data

`sst2_train.csv` — 67349 rows. Vocab 14819. CLS=1 at index 0, pad to 64.

## Model

4 layers, d_model 256, 8 heads, d_ff 1024. [`src/model.py`](src/model.py)

## Train

AdamW 2e-4. Loss table: [`results/training_notes.md`](results/training_notes.md)

## Predict

[`src/predict.py`](src/predict.py)

```
we all will fail          → negative  0.99
painting a girl           → positive  0.82
its harsh but its true.   → negative  0.96
```

## License

MIT
