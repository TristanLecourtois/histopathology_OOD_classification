import random

import torch
import torch.nn as nn


class MixStyle(nn.Module):
    """
    MixStyle applied to 1D feature vectors.
    Mixes instance-level statistics (mean, std) between random pairs of samples,
    simulating domain style transfer in feature space.

    p     : probability of applying MixStyle to a batch
    alpha : Beta distribution parameter (controls mixing strength)
    eps   : numerical stability for std computation
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.p     = p
        self.alpha = alpha
        self.eps   = eps
        self.beta  = torch.distributions.Beta(alpha, alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or random.random() > self.p:
            return x

        B = x.size(0)

        mu  = x.mean(dim=1, keepdim=True)
        sig = (x.var(dim=1, keepdim=True) + self.eps).sqrt()

        x_normed = (x - mu) / sig

        lam  = self.beta.sample((B, 1)).to(x.device)
        perm = torch.randperm(B, device=x.device)

        mu_mix  = lam * mu  + (1 - lam) * mu[perm]
        sig_mix = lam * sig + (1 - lam) * sig[perm]

        return x_normed * sig_mix + mu_mix
