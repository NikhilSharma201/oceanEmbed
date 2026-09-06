"""
model.py — OceanEmbed model

Architecture (matches the pipeline described in the proposal):
  Surface inputs (SST, SSH, lat, lon)
    -> Embedding layer (stands in for a pretrained satellite-embedding
       encoder like Clay/Prithvi; in the full system, replace `Embedder`
       with a frozen pretrained encoder + this same small head)
    -> Prediction head
    -> For each depth: (low, median, high) quantiles
       This IS the uncertainty quantification differentiator — instead of
       one number, the model outputs a calibrated range at each depth.

We use quantile regression (pinball loss) rather than full simulation-based
inference (SBI) for this prototype: it gives genuine, calibrated uncertainty
bands with a much smaller implementation surface, which matters given the
timeline. The proposal can (and should) still cite SBI as the more powerful
extension of this same idea — the architecture below is built so an `sbi`
posterior-estimation head could later replace `PredictionHead` without
touching the rest of the pipeline.
"""

import torch
import torch.nn as nn

QUANTILES = [0.1, 0.5, 0.9]  # 80% confidence interval + median


class Embedder(nn.Module):
    """
    Stands in for a pretrained satellite-embedding encoder (Clay/Prithvi).
    Takes raw surface features and produces a dense embedding.
    In production: freeze a real pretrained encoder here and only train
    the head below (much less data needed, per the proposal's approach).
    """

    def __init__(self, in_features=4, embed_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class PredictionHead(nn.Module):
    """Predicts (low, median, high) quantiles of temperature at each depth."""

    def __init__(self, embed_dim=32, n_depths=6, n_quantiles=3):
        super().__init__()
        self.n_depths = n_depths
        self.n_quantiles = n_quantiles
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_depths * n_quantiles),
        )

    def forward(self, embedding):
        out = self.net(embedding)
        return out.view(-1, self.n_depths, self.n_quantiles)


class OceanEmbedModel(nn.Module):
    def __init__(self, in_features=4, embed_dim=32, n_depths=6, n_quantiles=3):
        super().__init__()
        self.embedder = Embedder(in_features, embed_dim)
        self.head = PredictionHead(embed_dim, n_depths, n_quantiles)

    def forward(self, x):
        emb = self.embedder(x)
        return self.head(emb)  # (batch, n_depths, n_quantiles)


def pinball_loss(preds, target, quantiles=QUANTILES):
    """
    Quantile (pinball) loss — trains the model to produce calibrated
    quantiles rather than a single point estimate.
    preds: (batch, n_depths, n_quantiles)
    target: (batch, n_depths)
    """
    losses = []
    for i, q in enumerate(quantiles):
        errors = target - preds[:, :, i]
        losses.append(torch.max((q - 1) * errors, q * errors))
    return torch.stack(losses, dim=-1).mean()
