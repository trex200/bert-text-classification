# Data

Exact file used in the notebook: **`sst2_train.csv`**

| | |
|---|---|
| Language | English |
| Task | binary sentiment |
| Rows | 67349 |
| Columns | `sentence`, `label` |
| Labels | `0` negative, `1` positive |
| First rows | `hide new secretions from the parental units` / 0 |

## Source

**SST-2** (Stanford Sentiment Treebank, binary) — the GLUE train split.

- Site: [nlp.stanford.edu/sentiment](https://nlp.stanford.edu/sentiment/)
- Same 67349-row train split: [Hugging Face `stanfordnlp/sst2`](https://huggingface.co/datasets/stanfordnlp/sst2)
- GLUE zip: https://dl.fbaipublicfiles.com/glue/data/SST-2.zip

A short peek is in [`sst2.sample.csv`](sst2.sample.csv). Drop the full CSV at `data/sst2_train.csv` when training.

## Vocab from the notebook run

| | |
|---|---|
| lang | English only |
| `n_words` | 14819 |

PAD=0, CLS=1, UNK=2. Sequence = `[CLS] + tokens`, pad to 64. Batch = 32.
