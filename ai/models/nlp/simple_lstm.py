from dataclasses import dataclass

from configsys.config import ConfigMixin
from torch import nn


class SimpleLSTM(nn.Module):
    @dataclass(kw_only=True)
    class Config(ConfigMixin):
        vocab_size: int
        n_embd: int
        hidden_size: int
        num_layers: int
        _target_ = 'ai.models.nlp.simple_lstm.SimpleLSTM'

    def __init__(self, config: Config):
        super().__init__()
        c = config
        self.config = config
        self.embedding = nn.Embedding(c.vocab_size, c.n_embd)
        self.lstm = nn.LSTM(c.n_embd, hidden_size=c.hidden_size, num_layers=c.num_layers, batch_first=True)
        self.proj = nn.Linear(c.hidden_size, c.vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, (hn, cn) = self.lstm(x)
        x = self.proj(x)
        return x
