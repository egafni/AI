import torch

from ai.models.nlp.mlp import MLP


def test_mlp():
    # make sure we can forward through a model
    b, t, v = 3, 10, 5
    x = torch.randint(0, v, size=(b, t))
    model = MLP.Config(vocab_size=v, block_size=t, n_embd=7, n_hidden=3, n_layers=2).i()
    model(x)
