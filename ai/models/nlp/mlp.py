from dataclasses import dataclass

from configsys.config import ConfigMixin
from torch import nn


class MLP(nn.Module):
    @dataclass(kw_only=True)
    class Config(ConfigMixin):
        vocab_size: int
        block_size: int
        n_embd: int
        n_hidden: int
        n_layers: int

        act: nn.Module = nn.Tanh

        _target_: str = "ai.models.nlp.mlp.MLP"

    def __init__(self, config: Config):
        super().__init__()
        c = self.config = config

        layers = [nn.Embedding(c.vocab_size, c.n_embd), nn.Flatten()]

        for i in range(c.n_layers - 1):
            layers += [
                nn.Linear(c.n_embd * c.block_size if i == 0 else c.n_hidden, c.n_hidden),
                nn.BatchNorm1d(c.n_hidden),
                c.act()
            ]

        layers += [
            nn.Linear(c.n_hidden, c.vocab_size)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.layers(x)  # (B,T,C)

        return logits
