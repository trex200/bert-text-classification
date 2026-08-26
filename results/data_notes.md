# Data-prep numbers (notebook)

- vocab size: **14819** (includes PAD=0, CLS=1, UNK=2)
- input_tensors: `[67349, 64]`
- first sequence starts `[1, 3, 4, 5, 6, 7, 8, 9, 0, …]` — CLS then words then PAD
- first label: `0`
- batch size 32

Model build:

```
Bert(num_layers=4, d_model=256, num_heads=8, d_feedforward=1024,
     vocablen=14819, max_seq_len=64, dropout=0.1)
```
