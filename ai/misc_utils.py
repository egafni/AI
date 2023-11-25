import json
from json import JSONDecodeError

import fsspec
import pytorch_lightning as pl
from loguru import logger
from optuna.integration import PyTorchLightningPruningCallback


def write_json_file(fpath, data):
    with fsspec.open(fpath, "w") as fp:
        try:
            fp.write(json.dumps(data))
        except JSONDecodeError as ex:
            logger.error(f"error processing: {data}")
            raise ex


class OptunaPruningCallback(PyTorchLightningPruningCallback):
    def on_init_start(self, trainer: pl.Trainer) -> None:
        pass


def get_vocab_size(fname):
    with open(fname) as fp:
        data = fp.read()

    vocab = sorted(set(data))
    return len(vocab)
