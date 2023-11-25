from dataclasses import dataclass

import torch
from configsys.config import ConfigMixin
from torch import nn


class Transformer(nn.Module):
    @dataclass(kw_only=True)
    class Config(ConfigMixin):
        vocab_size: int
        block_size: int
        n_embd: int
        n_heads: int
        n_blocks: int
        dropout: float
        _target_ = 'ai.models.nlp.transformer.Transformer'

    def __init__(self, config: Config):
        super().__init__()
        c = self.config = config


        self.token_embedding_table = nn.Embedding(c.vocab_size, c.n_embd)
        self.position_embedding_table = nn.Embedding(c.block_size, c.n_embd)
        self.blocks = nn.Sequential(
            *[TransformerBlock(n_heads=c.n_heads, n_embd=c.n_embd, in_channels=c.n_embd, dropout=c.dropout)
              for i in range(c.n_blocks)]
        )
        self.ln_final = nn.LayerNorm(c.n_embd)
        self.project = nn.Linear(c.n_embd, c.vocab_size)

        self.register_buffer('positions', torch.arange(c.block_size))

    def forward(self, x) -> (torch.Tensor, torch.Tensor):
        c = self.config
        B, T, C, H, V = x.shape[0], c.block_size, c.n_embd, c.n_embd, c.vocab_size
        assert x.shape == (B, T)

        emb = self.token_embedding_table(x)  # B,T,C
        assert emb.shape == (B, T, C)
        assert not x.isnan().any()

        pos = self.position_embedding_table(self.positions)  # T,C
        assert pos.shape == (T, C)
        assert not x.isnan().any()

        x = emb + pos  # B,T,C
        assert x.shape == (B, T, C)
        assert not x.isnan().any()

        x = self.blocks(x)  # B,T,H
        x = self.ln_final(x)  # B,T,H
        x = self.project(x)  # B,T,V
        assert x.shape == (B, T, V)

        logits = x
        return logits


class AttentionHead(nn.Module):
    def __init__(self, n_channels, head_size, dropout):
        super().__init__()
        C, H = n_channels, head_size

        self.H = H

        self.q = nn.Linear(C, H, bias=False)
        self.k = nn.Linear(C, H, bias=False)
        self.v = nn.Linear(C, H, bias=False)

        self.register_buffer('scale', torch.tensor(H ** -0.5))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        T = x.shape[1]  # time dim
        mask = ~torch.tril(torch.ones(T, T).to(torch.bool)).to(x.device)  # causal mask

        # forward
        q, k, v = self.q(x), self.k(x), self.v(x)  # B,T,C
        w = q @ k.transpose(-2, -1)  # B,T,C @ B,C,T = B,T,T
        w = torch.masked_fill(w, mask, value=float('-inf'))
        w = w * self.scale  # causal mask, normalize
        w = torch.softmax(w, dim=-1)
        w = self.dropout(w)
        assert w.isnan().sum() == 0
        x = w @ v  # B,T,T @ B,T,C = B,T,C
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, n_embd, in_channels, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.n_embd = n_embd
        self.in_channels = in_channels
        self.heads = nn.ModuleList(
            [AttentionHead(n_channels=in_channels, head_size=n_embd // n_heads, dropout=dropout) for _ in
             range(n_heads)])
        self.project = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        o = torch.cat([head(x) for head in self.heads], dim=-1)
        o = self.dropout(self.project(o))
        return o


class TransformerBlock(nn.Module):
    def __init__(self, n_heads, n_embd, in_channels, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(in_channels)
        self.mha = MultiHeadAttention(n_heads=n_heads, n_embd=n_embd, in_channels=in_channels, dropout=dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class FeedForward(nn.Module):
    def __init__(self, n_features, dropout):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(n_features, n_features * 4),
                                    nn.ReLU(),
                                    nn.Linear(n_features * 4, n_features),
                                    nn.Dropout(dropout))

    def forward(self, x):
        return self.layers(x)
