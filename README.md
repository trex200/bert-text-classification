# BERT-style text classification

Transformer **encoder** + **[CLS]** token → class logits.

Not `BertForSequenceClassification` from Hugging Face. The encoder and the classification head are written in this repo.

> Status: **repo is up.** Model / train loop / dataset / screenshots incoming.

## Idea

BERT-style models pack the whole sentence into one vector: a learned `[CLS]` token at the front. After the encoder stack, that position is the sentence embedding. A linear layer turns it into class scores.

```
[CLS] token1 token2 ... tokenN [SEP?]
  |
  v
encoder (self-attn + FFN, × N layers)
  |
  v
hidden[0]  →  linear  →  logits
```

## Layout

| path | |
|---|---|
| `src/` | encoder, CLS head, data |
| `notebooks/` | walkthrough |
| `data/` | dataset notes |
| `results/` | loss, accuracy, sample preds |
| `assets/screenshots/` | notebook / train captures |

## License

MIT
