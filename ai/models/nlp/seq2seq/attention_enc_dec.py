import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import torch.nn as nn
import math


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Attention(nn.Module):
    def __init__(self, n_embd, block_size):
        super().__init__()
        self.n_embd = n_embd
        self.wq = nn.Linear(n_embd, n_embd)
        self.wk = nn.Linear(n_embd, n_embd)
        self.wv = nn.Linear(n_embd, n_embd)

        causal_mask = torch.ones(block_size, block_size)
        causal_mask = torch.tril(causal_mask).to(bool)
        self.register_buffer('causal_mask', causal_mask)

    def forward(self, q, k, v, causal: bool):
        # q: B,T1,C
        # k: B,T2,C
        # v: B,T2,C
        q = self.wq(q)  # B,T1,C
        k = self.wk(k)  # B,T2,C
        v = self.wv(v)  # B,T2,C
        T1 = q.shape[1]
        T2 = k.shape[1]

        weights = q @ k.transpose(-1, -2) / math.sqrt(self.n_embd)  # B,T1,T2
        if causal:
            weights = weights.masked_fill(~self.causal_mask, -torch.inf)
        attn = torch.softmax(weights, axis=-1)  # B,T1,T2
        assert not torch.isnan(attn).any()
        x = attn @ v  # B,T1,C
        return x


class Encoder(nn.Module):
    def __init__(self, input_size, n_embd, max_block_size):
        super().__init__()

        self.register_buffer('pos_id', torch.arange(0, max_block_size))
        self.pos_enc = nn.Embedding(max_block_size, n_embd)
        self.embedding = nn.Embedding(input_size, n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = Attention(n_embd, max_block_size)

        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd)
        )

    def forward(self, x):
        # note i need positional encoding
        o = self.embedding(x) + self.pos_enc(self.pos_id)
        o = self.ln1(o)
        o = self.attn(o, o, o, causal=False)
        assert not torch.isnan(o).any()
        o = self.mlp(self.ln2(o))
        return o


class Decoder(nn.Module):
    def __init__(self, output_vocab_size, n_embd, block_size):
        super().__init__()
        self.embedding = nn.Embedding(output_vocab_size, n_embd)
        self.pos_enc = nn.Embedding(block_size, n_embd)
        self.s_ln1 = nn.LayerNorm(n_embd)
        self.s_attn = Attention(n_embd, block_size)

        self.x_ln1 = nn.LayerNorm(n_embd)
        self.x_attn = Attention(n_embd, block_size)

        # self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd)
        )

        self.proj = nn.Linear(n_embd, output_vocab_size)

        self.register_buffer('pos_id', torch.arange(0, block_size))

    def forward(self, x, decoder_outputs):
        x = self.embedding(x) + self.pos_enc(self.pos_id)

        # self attention
        o = self.s_attn(x, x, x, causal=True)
        x = self.s_ln1(x + o)

        # cross attention
        o = self.x_attn(x, decoder_outputs, decoder_outputs, causal=False)
        o = self.x_ln1(x + o)

        o = self.mlp(o)
        o = self.proj(o)
        return o
