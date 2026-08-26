# BERT-style text classification

Transformer **encoder** + **[CLS]** → 2-way sentiment on SST-2.

> Status: **encoder + CLS head + SST-2 notes in.** Vocab / CLS-prepending / train loop next.

## Data

`sst2_train.csv` — 67349 rows, `sentence` + `label` (0/1).

## Model

[`src/model.py`](src/model.py): embed + PE + encoder → `hidden[:, 0]` (CLS) → Linear to 2 classes.

`max_seq_len = 64`, PAD = 0.

## License

MIT
