"""The complete AGAP-Net model, with no alternative architectures."""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    n_layers: int = 3
    input_dim: int = 1920
    width: int = 256
    input_dropout: float = 0.3
    embedding_dropout: float = 0.3
    moment_ratio: float = 0.0

    def __post_init__(self):
        if self.n_layers < 1 or self.input_dim < 1 or self.width < 1 or self.width % 4:
            raise ValueError("Use positive input dimensions and a width divisible by four.")
        if not 0 <= self.moment_ratio <= 1:
            raise ValueError("moment_ratio must lie in [0, 1].")
        if any(not 0 <= p < 1 for p in [self.input_dropout, self.embedding_dropout]):
            raise ValueError("Dropout probabilities must lie in [0, 1).")


class LayerFusion(nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_layers))
        # Retained for state-dictionary compatibility; every supplied layer participates.
        self.register_buffer("mask", torch.ones(n_layers))

    def forward(self, x):
        weights = torch.softmax(self.logits, dim=0)
        return torch.einsum("l,bltd->btd", weights, x)


class InputProj(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fuse = LayerFusion(config.n_layers)
        self.proj = nn.Linear(config.input_dim, config.width)
        self.norm = nn.LayerNorm(config.width)
        self.drop = nn.Dropout(config.input_dropout)

    def forward(self, x):
        return self.drop(self.norm(self.proj(self.fuse(x))))


class AttentiveStatPool(nn.Module):
    """Attention over all valid vectors, concatenated with their moments."""

    def __init__(self, d, moment_ratio=0.0):
        super().__init__()
        self.score = nn.Linear(d, 1)
        self.moment_ratio = moment_ratio

    def forward(self, x, valid):
        logits = self.score(x).squeeze(-1).masked_fill(~valid, float("-inf"))
        attention = (torch.softmax(logits, -1).unsqueeze(-1) * x).sum(1)
        selected = valid
        if self.moment_ratio > 0:
            norms = x.norm(dim=-1).masked_fill(~valid, -1e9)
            k = (valid.sum(1).float() * self.moment_ratio).long().clamp(1, x.shape[1])
            threshold = norms.topk(x.shape[1], dim=-1).values.gather(1, (k - 1).unsqueeze(1))
            selected = valid & (norms >= threshold)
        mask = selected.unsqueeze(-1)
        count = mask.sum(1).clamp(min=1)
        mean = (x * mask).sum(1) / count
        variance = ((x - mean.unsqueeze(1)) ** 2 * mask).sum(1) / count
        return torch.cat([attention, mean, variance.clamp(min=1e-6).sqrt()], -1)


class GatedAgentAttention(nn.Module):
    """Gated values, agent aggregation, then context broadcast to queries."""

    def __init__(self, d, n_agents, heads=4):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.n_agents = n_agents
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.g = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(0.1)
        self.agents = nn.Parameter(torch.randn(n_agents, d) * 0.02)

    def _split(self, x):
        batch, time, width = x.shape
        return x.view(batch, time, self.h, self.dh).transpose(1, 2)

    def forward(self, x, valid):
        batch, time, width = x.shape
        q, k = self._split(self.q(x)), self._split(self.k(x))
        values = self._split(torch.sigmoid(self.g(x)) * self.v(x))
        agents = self.agents.view(self.n_agents, self.h, self.dh).permute(1, 0, 2)
        scores = torch.einsum("hmd,bhtd->bhmt", agents, k) / math.sqrt(self.dh)
        scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
        context = torch.softmax(scores, -1) @ values
        scores = torch.einsum("bhtd,hmd->bhtm", q, agents) / math.sqrt(self.dh)
        output = torch.softmax(scores, -1) @ context
        output = output.transpose(1, 2).reshape(batch, time, width)
        return self.drop(self.o(output))


class AgentBlock(nn.Module):
    def __init__(self, d, n_agents):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = GatedAgentAttention(d, n_agents)
        self.ffn = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(0.1), nn.Linear(4 * d, d)
        )
        self.drop = nn.Dropout(0.1)

    def forward(self, x, valid):
        x = x + self.attn(self.n1(x), valid)
        return x + self.drop(self.ffn(self.n2(x)))


class PrototypeHead(nn.Module):
    def __init__(self, d, anchors):
        super().__init__()
        self.register_buffer("anchors", anchors)
        self.mlp = nn.Sequential(nn.Linear(3, 64), nn.GELU(), nn.Linear(64, d))
        self.scale = nn.Parameter(torch.tensor(8.0))

    def prototypes(self):
        return self.mlp(self.anchors)

    def forward(self, z):
        return self.scale * (F.normalize(z, dim=-1) @ F.normalize(self.prototypes(), dim=-1).t())


class AGAPNet(nn.Module):
    """Two temporal agent blocks, one spectral block, and shared VAD prototypes.

    Input: features [batch, layers, padded_time, channels], valid [batch, time].
    Output: class logits and the VAD/embedding tensors used by the objective.
    """

    def __init__(self, n_cls, anchors, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        c, d = self.config, self.config.width
        anchors = torch.as_tensor(anchors, dtype=torch.float32)
        if anchors.shape != (n_cls, 3) or not torch.isfinite(anchors).all():
            raise ValueError("Provide one finite three-dimensional VAD prior per class.")
        self.register_buffer("vad_anchors", anchors)
        self.n_cls = n_cls
        self.inp = InputProj(c)
        n_agents = n_cls + 2
        self.time_blocks = nn.ModuleList([AgentBlock(d, n_agents) for _ in range(2)])
        self.pool_t = AttentiveStatPool(d, c.moment_ratio)
        self.freq_proj = nn.Linear(d, d)
        self.freq_block = AgentBlock(d, n_agents)
        self.pool_f = AttentiveStatPool(d, c.moment_ratio)
        self.head = nn.Sequential(nn.Linear(6 * d, d), nn.GELU(), nn.Dropout(c.embedding_dropout))
        self.proto = PrototypeHead(d, anchors)
        self.vad_head = nn.Linear(d, 3)
        with torch.no_grad():
            prototypes = self.proto.prototypes()[:n_cls]
            for block in [*self.time_blocks, self.freq_block]:
                block.attn.agents[:n_cls] = prototypes / math.sqrt(d)
                block.attn.agents[n_cls:] = torch.randn(2, d) * 0.02 / math.sqrt(d)

    def forward(self, features, valid):
        c = self.config
        if features.ndim != 4 or features.shape[1] != c.n_layers or features.shape[-1] != c.input_dim:
            raise ValueError("Feature dimensions must match the model configuration.")
        if valid.dtype != torch.bool or valid.shape != (features.shape[0], features.shape[2]):
            raise ValueError("valid must be a boolean [batch, time] tensor.")
        if not valid.any(dim=1).all():
            raise ValueError("Each utterance must contain at least one valid frame.")
        h = self.inp(features)
        for block in self.time_blocks:
            h = block(h, valid)
        temporal = self.pool_t(h, valid)
        magnitude = torch.fft.rfft(h, dim=1).abs()
        magnitude = magnitude / (magnitude.mean(dim=1, keepdim=True) + 1e-5)
        frequency_valid = torch.ones(h.shape[0], magnitude.shape[1], dtype=torch.bool, device=h.device)
        frequency = self.freq_block(self.freq_proj(torch.log1p(10 * magnitude)), frequency_valid)
        spectral = self.pool_f(frequency, frequency_valid)
        z = self.head(torch.cat([temporal, spectral], -1))
        return self.proto(z), {"vad": self.vad_head(z), "z": z}

    def aux_losses(self, aux, y_gold, vad_targets):
        regression = F.l1_loss(aux["vad"], vad_targets[y_gold])
        prototypes = self.proto.prototypes()
        target = (prototypes / prototypes.norm(dim=-1, keepdim=True)).detach()
        pull = torch.zeros((), device=target.device)
        for block in [*self.time_blocks, self.freq_block]:
            agents = block.attn.agents[:self.n_cls]
            agents = agents / agents.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            pull = pull + F.mse_loss(agents, target)
        return {"anchor_pull": pull / 3, "vad_reg": regression}
