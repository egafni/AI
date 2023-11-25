# type: ignore
from collections.abc import Mapping
from dataclasses import dataclass

import pytorch_lightning as pl
import torch
from configsys.config import ConfigMixin
from configsys.utils import import_and_instantiate
from torch import nn
from torchmetrics import Metric

from ai.models.nlp.mlp import MLP
from ai.models.nlp.simple_lstm import SimpleLSTM
from ai.models.nlp.transformer import Transformer
from ai.models.nlp.wavenet import WaveNet

MetricType = Metric | Mapping[str, nn.Module] | None


class NextToken(pl.LightningModule):
    @dataclass(kw_only=True)
    class Config(ConfigMixin):
        model: MLP.Config | WaveNet.Config | SimpleLSTM.Config | Transformer.Config
        # ex: "torch.optim.AdamW"
        optimizer_class: str
        # ex: dict(weight_decay=1e-4, lr=1e-3)
        optimizer_init_params: dict
        # ex: "torch.optim.lr_scheduler.ReduceLROnPlateau"
        scheduler_class: str | None = None
        # ex: dict(mode="min", factor=0.2, patience=10)
        scheduler_init_params: dict | None = None
        # ex: dict(monitor="train/loss", interval="step")
        scheduler_lightning_cfg: dict | None = None

        _target_: str = "ai.lightning_modules.next_token.NextToken"

    def __init__(self, config: Config):
        super().__init__()
        self.save_hyperparameters("config")
        self.config = config
        self.model = self.config.model.instantiate_target()

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, batch_idx, stage):
        x, y = batch
        y_hat = self(x)
        # loss = nn.functional.cross_entropy(y_hat, y[:, -1])
        # loss = nn.functional.cross_entropy(y_hat[:, -1], y[:, -1])
        if y_hat.ndim == 3:
            b, t, c = y_hat.shape
            # autoregressive prediction of next token
            loss = nn.functional.cross_entropy(y_hat.view(b * t, c), y.view(b * t))
        else:
            # only predicted the next token
            loss = nn.functional.cross_entropy(y_hat, y[:, -1])

        self.log(f"{stage}/loss", loss, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # just generate some text
        return self.generate(batch[0][0], max_tokens=1000)

    @torch.inference_mode()
    def generate(self, X, max_tokens, prefix_inputs=False):
        # x is a batch
        y_hat = []
        for i in range(max_tokens):
            y_logit = self(X)
            assert isinstance(y_logit, torch.Tensor)
            if y_logit.ndim == 2:
                y_proba = torch.softmax(y_logit, axis=1)
                next_token = torch.multinomial(y_proba, num_samples=1)
            elif y_logit.ndim == 3:
                y_proba = torch.softmax(y_logit, axis=2)
                next_token = torch.multinomial(y_proba[:, -1, :], num_samples=1)

            X = torch.cat([X[:, 1:], next_token], axis=1)
            y_hat += [next_token]
        y_hat = torch.cat(y_hat, 1)
        if prefix_inputs:
            y_hat = torch.cat([X, y_hat], axis=1)
        return y_hat

    def configure_optimizers(self):
        """
        Instantiates the optimizer and the scheduler from the classes and parameters specified in the config.
        """
        if hasattr(self.model, 'configure_optimizers'):
            # run the model's .configure_optimizers() method
            return self.model.configure_optimizers(self.config.optimizer_class, self.config.optimizer_init_params)
        else:
            optimizer = import_and_instantiate(
                self.config.optimizer_class, self.parameters(), **self.config.optimizer_init_params
            )

            if self.config.scheduler_class is None:
                return optimizer

            scheduler = import_and_instantiate(self.config.scheduler_class, optimizer,
                                               **self.config.scheduler_init_params)
            scheduler_dict = {"scheduler": scheduler}

            if self.config.scheduler_lightning_cfg is not None:
                scheduler_dict.update(**self.config.scheduler_lightning_cfg)

            return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
