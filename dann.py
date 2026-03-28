import numpy as np
import torch
import torch.nn as nn


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


class DANNModel(nn.Module):
    def __init__(self, feat_dim: int, num_domains: int):
        super().__init__()
        self.label_clf = nn.Linear(feat_dim, 2)
        self.domain_clf = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_domains),
        )

    def forward(self, x, alpha: float = 1.0):
        label_logits  = self.label_clf(x)
        rev_x         = GradientReversalFunction.apply(x, alpha)
        domain_logits = self.domain_clf(rev_x)
        return label_logits, domain_logits


def get_alpha(epoch: int, num_epochs: int, gamma: float = 10.0) -> float:
    p = epoch / max(num_epochs - 1, 1)
    return 2.0 / (1.0 + np.exp(-gamma * p)) - 1.0
