"""BERT-style encoder + CLS classifier (notebook transcription)."""

import math

import torch
import torch.nn as nn


def scaled_attention(q, k, v, mask=None):
    dk = q.size(-1)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dk)
    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
    attn_probs = torch.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_probs, v)
    return output, attn_probs


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.linear_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = scaled_attention(q, k, v, mask)
        values = values.permute(0, 2, 1, 3).contiguous()
        values = values.reshape(batch_size, seq_len, self.d_model)
        return self.linear_proj(values)


class Position_encoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, seq_len):
        return self.pe[:, :seq_len, :]


class encoder_feedforward(nn.Module):
    def __init__(self, d_model, d_feedforward, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_feedforward)
        self.linear_2 = nn.Linear(d_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.linear_2(self.dropout(self.activation(self.linear_1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_feedforward, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.feedforward = encoder_feedforward(d_model, d_feedforward, dropout)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        residual = x
        x = self.norm1(self.dropout1(self.attention(x, mask)) + residual)
        residual = x
        x = self.norm2(self.dropout2(self.feedforward(x)) + residual)
        return x


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_feedforward, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_feedforward, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Bert(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_feedforward, vocablen, max_seq_len, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocablen, d_model, padding_idx=0)
        self.pos_encoding_module = Position_encoding(d_model, max_seq_len)
        self.encoder = Encoder(num_layers, d_model, num_heads, d_feedforward, dropout)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, 2)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = x + self.pos_encoding_module(x.size(1))
        x = self.encoder(x, mask)
        cls = self.dropout(x[:, 0, :])
        return self.linear(cls)
