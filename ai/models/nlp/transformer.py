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


class AttentionHeads(nn.Module):
    def __init__(self, n_channels, n_heads, dropout):
        super().__init__()
        assert n_channels % n_heads == 0, 'invalid settings'

        head_size = n_channels // n_heads
        C, Hs, N = n_channels, head_size, n_heads

        self.head_size = head_size
        self.n_heads = n_heads

        self.q = nn.Linear(C, C, bias=False)
        self.k = nn.Linear(C, C, bias=False)
        self.v = nn.Linear(C, C, bias=False)

        self.register_buffer('scale', torch.tensor(Hs ** -0.5))

        self.dropout = nn.Dropout(dropout)

        self.last_attn_map = None

    def forward(self, x):
        B, T, C = x.shape

        T = x.shape[1]  # time dim
        mask = ~torch.tril(torch.ones(T, T).to(torch.bool)).to(x.device)  # causal mask

        # forward
        q, k, v = self.q(x), self.k(x), self.v(x)  # B,T,C
        q = q.view(B, T, self.n_heads, self.head_size).transpose(1, 2)  # B, nh, T, hs
        k = k.view(B, T, self.n_heads, self.head_size).transpose(1, 2)  # B, nh, T, hs
        v = v.view(B, T, self.n_heads, self.head_size).transpose(1, 2)  # B, nh, T, hs

        attn = q @ k.transpose(-2, -1)  # B,nh,T,C @ B,N,C,T = B,nh,T,T
        attn = torch.masked_fill(attn, mask, value=float('-inf'))
        attn = attn * self.scale  # causal mask, normalize
        attn = torch.softmax(attn, dim=-1)
        dattn = self.dropout(attn)
        self.last_attn_map = dattn.detach().cpu().clone()
        assert dattn.isnan().sum() == 0
        x = dattn @ v  # B,nh,T,T @ B,nh,T,hs = B,nh,T,hs

        x = x.transpose(1, 2).contiguous()  # B,T,nh,hs
        x = x.view(B, T, C)  # B,T,C
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, n_embd, in_channels, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.n_embd = n_embd
        self.in_channels = in_channels
        self.heads = AttentionHeads(n_channels=in_channels, n_heads=n_heads, dropout=dropout)
        self.project = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        o = self.heads(x)
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
