from dataclasses import dataclass

import torch
from configsys.config import ConfigMixin
from torch import nn


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel_size,
                              padding=self.padding,
                              dilation=dilation,
                              bias=False,
                              **kwargs)

    def forward(self, input_):
        return self.conv(input_)[:, :, :-self.padding] if self.padding else self.conv(input_)


class WaveBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates, kernel_size, causal):
        super(WaveBlock, self).__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        dilation_rates = [2 ** i for i in range(dilation_rates)]
        for dilation_rate in dilation_rates:
            if causal:
                conv = lambda: CausalConv1d(out_channels, out_channels, kernel_size=kernel_size, dilation=dilation_rate)
            else:
                conv = lambda: nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                                         padding=(dilation_rate * (kernel_size - 1)) // 2, dilation=dilation_rate)

            self.filter_convs.append(conv())
            self.gate_convs.append(conv())
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
        causal: bool  # i don't think this really does anything... the convs are already causal
        _target_ = 'ai.models.nlp.wavenet.WaveNet'

    def __init__(self, config: Config):
        super().__init__()
        c = config
        # self.LSTM = nn.GRU(input_size=input_size, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        self.emb = nn.Embedding(c.vocab_size, c.n_embd)

        self.wave_block1 = WaveBlock(c.n_embd, 16, 12, c.kernel_size, causal=c.causal)
        self.wave_block2 = WaveBlock(16, 32, 8, c.kernel_size, causal=c.causal)
        self.wave_block3 = WaveBlock(32, 64, 4, c.kernel_size, causal=c.causal)
        self.wave_block4 = WaveBlock(64, 128, 1, c.kernel_size, causal=c.causal)
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
        # x = x.flatten(1, -1)
        # raise
        x = self.fc(x)
        return x
