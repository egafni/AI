from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VQVAE(nn.Module):
    @dataclass
    class Config:
        n_embd: int

    def __init__(self, config: Config):
        self.config = c = config
        super(VQVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, c.n_embd, kernel_size=1)  # project down to embedding size
        )

        self.embedding = nn.Embedding(num_embeddings=3, embedding_dim=c.n_embd)

        # Commitment Loss Beta
        self.beta = 0.2

        self.decoder = nn.Sequential(
            nn.Conv2d(c.n_embd, 4, kernel_size=1),  # project up to original channel size
            nn.ConvTranspose2d(4, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # B, C, H, W
        z_e_of_x = self.encoder(x)

        ## Quantization
        B, C, H, W = z_e_of_x.shape
        z_e_of_x = rearrange(z_e_of_x, 'b c h w -> b (h w) c')

        # B,H*W,3; Compute pairwise distances to each embedding
        dist = torch.cdist(z_e_of_x, self.embedding.weight.repeat((B, 1, 1)))

        # (B,H*W) Find index of nearest embedding
        tokens = torch.argmin(dist, dim=-1)

        # Select the embedding weights
        e = self.embedding.weight[tokens.view(-1)]

        # flatten
        z_e_of_x = rearrange(z_e_of_x, 'b (h w) c -> (b h w) c', h=H, w=W)

        # Compute losses
        commitment_loss = torch.mean((z_e_of_x - e.detach()) ** 2)
        # train codebook
        codebook_loss = torch.mean((e - z_e_of_x.detach()) ** 2)
        # mix losses
        quantize_losses = codebook_loss + self.beta * commitment_loss

        # Ensure straight through gradient
        e = z_e_of_x + (e - z_e_of_x).detach()

        # Reshaping back to original input shape
        e = rearrange(e, '(b h w) c -> b c h w', h=H, w=W)
        tokens = rearrange(tokens, 'b (h w) -> b h w', h=H, w=W)

        # Decoder part
        output = self.decoder(e)

        return output, quantize_losses, tokens
