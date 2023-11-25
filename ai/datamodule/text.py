from dataclasses import dataclass

import torch
from configsys.config import ConfigMixin
from loguru import logger
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    def __init__(self, fname, split, block_size):
        self.name = fname
        self.split = split
        self.block_size = block_size

        with open(fname) as fp:
            data = fp.read()

        vocab = sorted(set(data))
        self.vocab = vocab

        i1, i2 = int(len(data) * .8), int(len(data) * .9)
        if split == 'train':
            data = data[:i1]
        elif split == 'val':
            data = data[i1:i2]
        elif split == 'test':
            data = data[i2:]
        else:
            raise ValueError(f'invalid {split}')

        self.itos = {i: c for i, c in enumerate(vocab)}
        self.stoi = {c: i for i, c in enumerate(vocab)}

        data_i = torch.as_tensor([self.stoi[c] for c in data], dtype=torch.long)

        self.data = data
        self.data_i = data_i

    def decode(self, x: list[int] | torch.Tensor):
        if torch.is_tensor(x):
            x = x.tolist()
        return [self.itos[i] for i in x]

    def encode(self, x: list[str] | torch.Tensor):
        if torch.is_tensor(x):
            x = x.tolist()
        return [self.stoi[c] for c in x]

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx: int):
        assert idx < len(self)
        return self.data_i[idx:idx + self.block_size], self.data_i[idx + 1:idx + self.block_size + 1]


class TextDataModule(LightningDataModule):
    @dataclass
    class Config(ConfigMixin):
        fname: str
        block_size: int
        batch_size: int
        num_workers: int = 0
        _target_: str = "ai.datamodule.text.TextDataModule"

    def __init__(self, config: Config):
        super().__init__()

        self.config = config
        self.ds_train = TextDataset(self.config.fname, 'train', self.config.block_size)
        self.ds_val = TextDataset(self.config.fname, 'val', self.config.block_size)
        self.ds_test = TextDataset(self.config.fname, 'test', self.config.block_size)

    def prepare_data(self) -> None:
        logger.info("Creating datasets...")

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.config.batch_size, num_workers=self.config.num_workers)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.config.batch_size, num_workers=self.config.num_workers)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.config.batch_size, num_workers=self.config.num_workers)
