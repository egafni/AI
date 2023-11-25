from dataclasses import dataclass

from configsys.config import ConfigMixin
from torch import nn
import torch


class WaveBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates, kernel_size):
        super(WaveBlock, self).__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        dilation_rates = [2 ** i for i in range(dilation_rates)]
        for dilation_rate in dilation_rates:
            self.filter_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                          padding=int((dilation_rate * (kernel_size - 1)) / 2), dilation=dilation_rate))
            self.gate_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                          padding=int((dilation_rate * (kernel_size - 1)) / 2), dilation=dilation_rate))
            self.convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=1))

    def forward(self, x):
        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            x = torch.tanh(self.filter_convs[i](x)) * torch.sigmoid(self.gate_convs[i](x))
            x = self.convs[i + 1](x)
            res = res + x
        return res


class WaveNet(nn.Module):
    @dataclass(kw_only=True)
    class Config(ConfigMixin):
        kernel_size: int
        vocab_size: int
        n_embd: int
        _target_ = 'ai.models.nlp.wavenet.WaveNet'

    def __init__(self, config: Config):
        c = config
        super().__init__()
        # self.LSTM = nn.GRU(input_size=input_size, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        self.emb = nn.Embedding(c.vocab_size, c.n_embd)

        self.wave_block1 = WaveBlock(c.n_embd, 16, 12, c.kernel_size)
        self.wave_block2 = WaveBlock(16, 32, 8, c.kernel_size)
        self.wave_block3 = WaveBlock(32, 64, 4, c.kernel_size)
        self.wave_block4 = WaveBlock(64, 128, 1, c.kernel_size)
        self.fc = nn.Linear(128, c.vocab_size)

    def forward(self, x, y=None):
        x = self.emb(x)
        x = x.permute(0, 2, 1)

        x = self.wave_block1(x)
        x = self.wave_block2(x)
        x = self.wave_block3(x)

        x = self.wave_block4(x)
        x = x.permute(0, 2, 1)
        # x, _ = self.LSTM(x)
        x = self.fc(x)
        return x
