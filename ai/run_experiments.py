import argparse

from ai import train
from ai.constants import REPO_DIR
# dm = TextDataModule.Config(fname='data/names.txt', block_size=8, batch_size=32, num_workers=0)
# dm.prepare_data()
# dm.train_data
# X, y = next(iter(dm.train_dataloader()))
from ai.datamodule.text import TextDataModule
from ai.lightning_modules.next_token import NextToken
from ai.misc_utils import get_vocab_size
from ai.models.nlp.transformer import Transformer
from ai.train import TrainerConfig, TrainNNConfig


def main(unique_id, resume):
    fname = f'{REPO_DIR}/data/shakespeare.txt'
    vocab_size = get_vocab_size(fname)
    lr = 2e-4
    n_embd = 256
    max_epochs = 10

    # batch_size = 64
    # block_size = 32
    # n_hidden = 256
    # model = MLP.Config(vocab_size=vocab_size, block_size=block_size, n_embd=n_embd, n_hidden=n_hidden, n_layers=4)

    # block_size = 1024
    # batch_size = 32
    # model = WaveNet.Config(vocab_size=vocab_size, kernel_size=2, n_embd=n_embd, causal=True)

    # block_size = 256
    # batch_size = 32
    # model = SimpleLSTM.Config(vocab_size=vocab_size, n_embd=n_embd, hidden_size=256, num_layers=4)

    block_size = 128
    n_embd = 512
    batch_size = 128
    accumulate_grad_batches = 4
    model = Transformer.Config(vocab_size=vocab_size, n_embd=n_embd, n_heads=16, n_blocks=16, dropout=0.2,
                               block_size=block_size)

    config = TrainNNConfig(
        name=f'{model._target_.split(".")[-1]}',
        experiment_group='next_token',
        unique_id=unique_id,
        datamodule=TextDataModule.Config(fname=fname, block_size=block_size, batch_size=batch_size, num_workers=0),
        trainer=TrainerConfig(
            accelerator='gpu',
            max_epochs=max_epochs,
            val_check_interval=200,
            limit_val_batches=.05 if 'shakespeare' in fname else 1.0,
            accumulate_grad_batches=4,
        ),
        lightning_module=NextToken.Config(
            model=model,
            optimizer_class='torch.optim.AdamW',
            optimizer_init_params=dict(lr=lr),
        ),
        early_stopping=dict(monitor="val/loss", min_delta=0.00, patience=25, verbose=True, mode="min"),
        model_checkpoint=dict(
            monitor="val/loss",
            mode="min",
            filename="epoch{epoch}__step{step}",
            auto_insert_metric_name=False,
            save_top_k=5,
            verbose=True,
        ),
        # resume=True,
        # fit_ckpt_path='/mnt/ssd3/user/spock/projects/AI/experiments/next_token/Transformer/v5//checkpoints/epoch0__step7250.ckpt',
        # resume=True,
        # fit_ckpt_path='last',
        logger='wandb'

    )
    train.train_model(config)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-u', '--unique_id', required=True)
    p.add_argument('-c', '--resume', action='store_true', help='resume training')
    args = p.parse_args()
    main(unique_id=args.unique_id, resume=args.resume)
