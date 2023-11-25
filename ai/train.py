"""
Train an Neural Net model
"""
import argparse
import json
import os
from dataclasses import dataclass, field
from getpass import getuser
from typing import Any, Literal

import optuna
import pytorch_lightning as pl
import torch
import wandb
from configsys.config import ConfigMixin
from loguru import logger
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers import CSVLogger, Logger, TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.profilers import SimpleProfiler

from ai.constants import DEFAULT_EXPERIMENT_DIR
from ai.datamodule.text import TextDataModule
from ai.lightning_modules.next_token import NextToken
from ai.misc_utils import write_json_file, OptunaPruningCallback


@dataclass(kw_only=True)
class TrainerConfig(ConfigMixin):
    """pytorch_lightning.Trainer parameters, for detailed help see the pytorch_lightning documentation."""

    accelerator: str
    max_epochs: int = field(metadata=dict(help="max epochs"))

    limit_train_batches: int | float | None = field(
        default=None,
        metadata=dict(help="limit the number of training batches, can be useful for quick testing"),
    )
    limit_val_batches: int | float | None = field(
        default=None,
        metadata=dict(help="limit the number of validation batches, can be useful for quick testing"),
    )
    limit_test_batches: int | float | None = field(
        default=None,
        metadata=dict(help="limit the number of tests batches, can be useful for quick testing"),
    )
    val_check_interval: int | float | None = field(default=None, metadata=dict(help="how often to check validation"))
    devices: int = field(
        default=1,
        metadata=dict(help="number of devices specified by accelerator to use"),
    )

    accumulate_grad_batches: int = field(
        default=1,
        metadata=dict(help="accumulate gradients over multiple batches."),
    )


@dataclass(kw_only=True)
class TrainNNConfig(ConfigMixin):
    """Top level config for training a model"""

    datamodule: TextDataModule.Config
    trainer: TrainerConfig
    lightning_module: NextToken.Config

    name: str = field(metadata=dict(help="experiment name"))
    unique_id: str

    experiment_group: str = field(
        metadata=dict(help="a name to group experiments by, for example in wandb"),
    )

    model_checkpoint: dict = field(
        metadata=dict(help="ModelCheckpoint params"),
    )

    # ex: dict(monitor="val/acc_macro", min_delta=0.00, patience=20, verbose=True, mode="max")
    early_stopping: dict | None = field(
        metadata=dict(help="Early stopping kwargs, if None, do not do early stopping."),
    )

    # ex: dict(swa_epoch_start=.8,swa_lrs=None,annealing_epochs=10,annealing_strategy='cos')
    swa: dict | None = field(default=None, metadata=dict(help="params to StochasticWeightAveraging"))

    # Experiment params
    wandb_user: str = field(
        default=getuser(),
        metadata=dict(help="username of the person running this experiment, defaults to current unix user"),
    )
    logger: str = field(
        default="wandb",
        metadata=dict(help="lightning logger to use", choices=["wandb", "csv", "tensorboard"]),
    )
    resume: bool = False
    fit_ckpt_path: str | None = field(
        default=None, metadata=dict(help="load trainer and model state from this checkpoint. "
                                         " set to 'best' to load best checkpoint")
    )
    predict: bool = field(default=False, metadata=dict(help="if True saves predictions to output folder"))
    matmul_precision: Literal["highest", "high", "medium"] = "high"

    seed: int = 1337
    base_output_dir: str = field(
        default=DEFAULT_EXPERIMENT_DIR, metadata=dict(help="output directory to store results in")
    )

    @property
    def output_dir(self):
        return os.path.join(self.base_output_dir, self.experiment_group, self.name, self.unique_id)

    @property
    def group_output_dir(self):
        return os.path.join(self.base_output_dir, self.experiment_group, self.name)

    def get_logger(self, trial_id: int | None = None) -> Logger:
        name = f"{self.name}/{self.unique_id}" + (f"/trial-{trial_id}" if trial_id is not None else "")
        if self.logger == "wandb":
            if "WANDB_API_KEY" not in os.environ:
                raise OSError("WANDB_API_KEY env variable must be set if logger=wandb")

            # note: init wandb early to capture all stdout/err
            # if "WANDB_PROJECT_NAME" not in os.environ:
            project = f"ai.{self.experiment_group}"
            logger: Logger = WandbLogger(
                name=name,
                project=project,
                id=name.replace('/', '__'),
                resume='must' if self.resume else 'never'
                # entity="",
            )
            logger.experiment.config.update(self.to_dict())  # type: ignore[attr-defined]
        elif self.logger == "tensorboard":
            logger = TensorBoardLogger(save_dir=f"{self.output_dir}/logs", name=name)
            # logger.experiment._get_file_writer().add_summary(self.to_dict().update(name="config"))
            # experiment._get_file_writer() returns None in some settings (multi-gpu) and add_summary fails
            # instead convert config to string for logging with add_text
            # logger.experiment._get_file_writer().add_summary(self.to_dict().update(name="config"))
            cfg = {key: str(val) for key, val in self.to_dict().items()}
            logger.experiment.add_text("config", json.dumps(cfg), 0)
        else:
            logger = CSVLogger(save_dir=f"{self.output_dir}/logs", name=name)
        return logger

    def train(self, trial=None):
        return train_model(self, trial=trial)


def train_model(config: TrainNNConfig, trial: optuna.Trial | None = None) -> dict[str, Any]:
    seed_everything(config.seed)

    torch.set_float32_matmul_precision(config.matmul_precision)

    logger.info(f"Training, output_dir: {config.output_dir}")
    config.to_yaml_file(f"{config.output_dir}/train.config")

    # ### Setup Logger ####
    os.makedirs(config.output_dir, exist_ok=True)
    trial_id = None if trial is None else trial.number
    pl_logger = config.get_logger(trial_id)

    lightning_module = config.lightning_module.instantiate_target()

    # ## Setup Callbacks

    checkpoint_dirpath = f"{config.output_dir}/checkpoints"
    if trial:
        checkpoint_dirpath += f"/trial{trial.number}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dirpath,
        **config.model_checkpoint,
    )

    callbacks = [
        checkpoint_callback,
        LearningRateMonitor(logging_interval="step"),
    ]
    if trial:
        assert config.optuna is not None
        callbacks.append(OptunaPruningCallback(trial, config.optuna.optimized_metric_name))
    if config.early_stopping:
        callbacks.append(EarlyStopping(**config.early_stopping))
    if config.swa:
        callbacks.append(StochasticWeightAveraging(**config.swa))
    # callbacks.append(DeviceStatsMonitor())
    callbacks.append(ModelSummary())

    profiler = SimpleProfiler(dirpath=config.output_dir, filename="perf_logs")

    trainer = pl.Trainer(
        devices=config.trainer.devices,
        deterministic=True,
        max_time={"days": 4},
        detect_anomaly=True,
        val_check_interval=config.trainer.val_check_interval,
        limit_train_batches=config.trainer.limit_train_batches,
        limit_val_batches=config.trainer.limit_val_batches,
        limit_test_batches=config.trainer.limit_test_batches,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        max_epochs=config.trainer.max_epochs,
        default_root_dir=config.output_dir,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        logger=pl_logger,
        log_every_n_steps=30,
        accelerator=config.trainer.accelerator,
        profiler=profiler,
    )

    datamodule = config.datamodule.instantiate_target()

    # collect metrics
    metrics = {}
    trainer.fit(lightning_module, datamodule=datamodule, ckpt_path=config.fit_ckpt_path)
    metrics.update(trainer.callback_metrics)
    trainer.test(ckpt_path="best", datamodule=datamodule)
    metrics.update(trainer.callback_metrics)

    results = dict(
        best_model_path=str(checkpoint_callback.best_model_path),
        output_dir=config.output_dir,
        checkpoint_dirpath=checkpoint_dirpath,
        callback_metrics={k: float(v) for k, v in metrics.items()},
        trainer_interrupted=trainer.interrupted,
        trainer_state_fn=trainer.state.fn,
        trainer_status=trainer.state.status,
        trainer_stage=trainer.state.stage,
    )

    # save results
    if not trial:
        write_json_file(f"{config.output_dir}/results.json", results)

    if config.predict:
        y_pred_val, y_pred_test = trainer.predict(ckpt_path="best", datamodule=datamodule)  # type: ignore[assignment]
        val_pred = torch.concat(y_pred_val).squeeze().cpu().numpy()
        test_pred = torch.concat(y_pred_test).squeeze().cpu().numpy()

        torch.save(val_pred, f"{config.output_dir}/val_pred.pt")
        torch.save(test_pred, f"{config.output_dir}/test_pred.pt")

    logger.info(f"output_dir: {config.output_dir}")
    pl_logger.finalize("successful" if not trainer.interrupted else "interrupted")
    if config.logger == "wandb":
        wandb.finish()

    return results


# def optimize(config: TrainConfig):
#     assert config.optuna, "No optuna config passed for hyperparameter optimization"
#     study = optuna.load_study(study_name=config.optuna.study_name, storage=config.optuna.storage)
#     while len(study.trials) < config.optuna.n_trials:
#         trial = study.ask(config.optuna.hyperparameter_distributions)
#         config.replace_fields(trial.params)
#         try:
#             results, _, _ = train_model(config, trial)
#             study.tell(trial, results["callback_metrics"][config.optuna.optimized_metric_name])
#         except optuna.TrialPruned as e:
#             study.tell(trial, state=optuna.trial.TrialState.PRUNED)
#             logging.info(str(e))
#             if config.logger == "wandb":
#                 wandb.finish()


def objective(config: TrainNNConfig, trial: optuna.Trial):
    assert config.optuna is not None
    params = {}
    for name, distribution in config.optuna.hyperparameter_distributions.items():
        assert isinstance(distribution, optuna.distributions.BaseDistribution)  # assert for mypy
        params[name] = trial._suggest(name, distribution)  # noqa
    logger.info(f"\nTrial parameters:\n{params}\n")
    config.replace_fields(params, in_place=True)
    try:
        results = train_model(config, trial)
    except optuna.TrialPruned:
        if config.logger == "wandb":
            wandb.finish()
        raise
    assert "callback_metrics" in results and isinstance(results["callback_metrics"], dict)  # pyright
    return results["callback_metrics"][config.optuna.optimized_metric_name]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="path to a TrainConfig yaml file")
    args = p.parse_args()
    train_config = TrainNNConfig.from_yaml_file(args.config)
    if train_config.optuna:
        assert train_config.optuna, "No optuna config passed for hyperparameter optimization"
        assert isinstance(train_config.optuna.storage, str)
        study = optuna.load_study(study_name=train_config.optuna.study_name, storage=train_config.optuna.storage)
        study.optimize(
            lambda trial: objective(train_config, trial),  # type: ignore[no-any-return]
            callbacks=[MaxTrialsCallback(train_config.optuna.n_trials, (TrialState.COMPLETE, TrialState.PRUNED))],
            gc_after_trial=True,
        )
    else:
        train_model(train_config)
