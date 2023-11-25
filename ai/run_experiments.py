from ai import train
from ai.constants import REPO_DIR

# dm = TextDataModule.Config(fname='data/names.txt', block_size=8, batch_size=32, num_workers=0)
# dm.prepare_data()
# dm.train_data
# X, y = next(iter(dm.train_dataloader()))
from ai.datamodule.text import TextDataModule
from ai.lightning_modules.next_token import NextToken
from ai.misc_utils import get_vocab_size
from ai.models.nlp.wavenet import WaveNet
from ai.train import TrainerConfig, TrainNNConfig

fname = f'{REPO_DIR}/data/names.txt'
vocab_size = get_vocab_size(fname)
block_size = 32
lr = 1e-3
n_embd = 256
n_hidden = 256
batch_size = 64

config = TrainNNConfig(
    name='wavenet',
    experiment_group='next_token',
    unique_id='v1',
    datamodule=TextDataModule.Config(fname=fname, block_size=block_size, batch_size=batch_size, num_workers=0),
    trainer=TrainerConfig(
        accelerator='cpu',
        max_epochs=1,
        val_check_interval=200,
    ),
    lightning_module=NextToken.Config(
        # model=MLP.Config(vocab_size=vocab_size, block_size=block_size, n_embd=n_embd, n_hidden=n_hidden, n_layers=4),
        model=WaveNet.Config(vocab_size=vocab_size, kernel_size=3, n_embd=n_embd, causal=True),
        optimizer_class='torch.optim.AdamW',
        optimizer_init_params=dict(lr=lr),
    ),
    early_stopping=None,
    model_checkpoint=dict(
        monitor="val/loss",
        mode="min",
        filename="epoch{epoch}__step{step}",
        auto_insert_metric_name=False,
        save_top_k=5,
        verbose=True,
    ),
    logger='wandb'

)
train.train_model(config)
