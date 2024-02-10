import math

import torch
from torch import nn


class AttentionHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config
        self.config = config
        self.q = nn.Linear(c.n_embd, c.n_embd)
        self.k = nn.Linear(c.n_embd, c.n_embd)
        self.v = nn.Linear(c.n_embd, c.n_embd)
        self.do = nn.Dropout(c.dropout)

        attn_mask = torch.ones(c.block_size, c.block_size)
        attn_mask.masked_fill(~torch.tril(attn_mask).to(bool), -torch.inf)
        self.register_buffer('attn_mask', attn_mask)

    def forward(self, x):
        c = self.config
        B = x.shape[0]
        T = c.block_size
        C = c.n_embd
        nh = c.n_head

        q = self.q(x)  # B,T,C
        k = self.k(x)  # B,T,C
        v = self.v(x)  # B,T,C
        q = q.view(B, T, nh, C // nh).permute(0, 2, 1, 3)  # B,nh,T,C//nh
        k = k.view(B, T, nh, C // nh).permute(0, 2, 3, 1)  # B,nh,C//nh,T
        v = v.view(B, T, nh, C // nh).permute(0, 2, 1, 3)  # B,nh,T,C//nh

        attn = q @ k * (math.sqrt(1 / (C // nh)))  # B,nh,T,T
        attn *= self.attn_mask
        attn = torch.softmax(attn, dim=-1)  # B,nh,T,T
        dattn = self.dropout(attn)
        o = dattn @ v  # B,nh,T,C//nh

        return o.permute(0, 2, 1, 3).contiguous().view(B, T, C)


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config
        self.l1 = nn.Linear(c.n_embd, c.n_embd * 4)
        self.l2 = nn.Linear(c.n_embd * 4, c.n_embd)
        self.do = nn.Dropout(c.dropout)

    def forward(self, x):
        o = self.l1(x)
        o = torch.relu(o)
        o = self.l2(o)
        o = self.do(o)
        return o


class MultiHeadedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config
        self.config = config

        self.heads = AttentionHeads(c)
        self.project = nn.Linear(c.n_embd, c.n_embd)
        self.dropout = nn.Dropout(c.n_embd)

    def forward(self, x):
        x = self.heads(x)
        x = self.project(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config
        self.config = config

        self.attn = MultiHeadedAttention(config)
        self.ln1 = nn.LayerNorm(c.n_embd)

        self.ln2 = nn.LayerNorm(c.n_embd)

        self.ff = FeedForward(c)

    def forward(self, x):
        o = x + self.attn(self.ln1(x))
        o = x + self.ff(self.ln2(o))
        return o


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config
        self.config = config
        self.token_emb = nn.Embedding(c.vocab_size, c.n_embd)
        self.pos_emb = nn.Embedding(c.block_size, c.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(c.n_layer)])
        self.ln_final = nn.LayerNorm(c.n_embd)
        self.proj = nn.Linear(c.n_embd, c.vocab_size)

        self.register_buffer('pos_id', torch.arange(0, c.block_size))

    def forward(self, x):
        tok_emb = self.token_emb(x)
        pos_emb = self.pos_emb(self.pos_id)
        o = pos_emb + tok_emb

        for block in self.blocks:
            o = block(o)
        self.ln_final(o)
        o = self.proj(o)
        return o

