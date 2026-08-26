# BERT-style text classification

Transformer **encoder** + **[CLS]** → 2-way sentiment on SST-2.

> Status: **v1 train logged.** Loss 0.49 → 0.067 over 10 epochs. Eval / sample preds next if you have them.

## Data

`sst2_train.csv` — 67349 rows, `sentence` + `label` (0/1). Vocab **14819**. Tensors `[N, 64]` with CLS=1 at position 0.

## Model

4 layers, d_model 256, 8 heads, d_ff 1024. [`src/model.py`](src/model.py)

## Train

AdamW 2e-4, weight decay 0.01, CE, grad clip 1.0. [`src/train.py`](src/train.py)

| epoch | avg loss |
|---|---|
| 1 | 0.49 |
| 5 | 0.11 |
| 10 | 0.067 |

Full table: [`results/training_notes.md`](results/training_notes.md)

## License

MIT
