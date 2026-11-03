"""The AGAP-Net training objective, fixed-schedule fitting, and prediction."""

from dataclasses import asdict, dataclass
import copy
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from .data import LAYERS, batch_indices, make_batch, training_moments
from .model import AGAPNet, ModelConfig


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 100
    seed: int = 1
    batch_size: int = 32
    frame_budget: int = 8192
    crop_frames: int = 400
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    ema_decay: float = 0.998

    def __post_init__(self):
        if min(self.epochs, self.batch_size, self.frame_budget, self.crop_frames, self.warmup_epochs) < 1:
            raise ValueError("Epoch, batch, frame and warmup counts must be positive.")
        if self.learning_rate <= 0 or self.weight_decay < 0 or not 0 <= self.ema_decay < 1:
            raise ValueError("Invalid optimizer or EMA configuration.")


def read_config(path=None):
    data = json.loads(Path(path).read_text()) if path else {}
    if set(data) - {"model", "training"}:
        raise ValueError("The configuration accepts only model and training sections.")
    return ModelConfig(**data.get("model", {})), TrainConfig(**data.get("training", {}))


def choose_device(value="auto"):
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def soft_label_matrix(anchors, epsilon=0.3, temperature=0.25):
    anchors = torch.as_tensor(anchors, dtype=torch.float32)
    kernel = torch.softmax(-(torch.cdist(anchors, anchors)**2) / temperature, -1)
    return (1 - epsilon) * torch.eye(len(anchors), device=anchors.device) + epsilon * kernel


def objective(model, logits, auxiliary, labels, targets):
    soft_targets = soft_label_matrix(targets)[labels]
    categorical = -(soft_targets * F.log_softmax(logits, -1)).sum(-1).mean()
    auxiliary_losses = model.aux_losses(auxiliary, labels, targets)
    total = categorical + 0.3 * auxiliary_losses["vad_reg"] + 0.01 * auxiliary_losses["anchor_pull"]
    return total


class EMA:
    def __init__(self, model, decay=0.998):
        self.decay, self.steps = decay, 0
        self.shadow = copy.deepcopy({k: v.detach().clone() for k, v in model.state_dict().items()})

    @torch.no_grad()
    def update(self, model):
        self.steps += 1
        decay = min(self.decay, (1 + self.steps) / (10 + self.steps))
        for name, value in model.state_dict().items():
            if value.is_floating_point():
                self.shadow[name].mul_(decay).add_(value.detach(), alpha=1 - decay)
            else:
                self.shadow[name].copy_(value)


def metrics(predicted, gold, n_classes):
    predicted, gold = np.asarray(predicted), np.asarray(gold)
    if predicted.shape != gold.shape or predicted.ndim != 1 or not len(gold):
        raise ValueError("Predictions and labels must be aligned, non-empty one-dimensional arrays.")
    if (predicted < 0).any() or (predicted >= n_classes).any() or (gold < 0).any() or (gold >= n_classes).any():
        raise ValueError("Class indices lie outside the label vocabulary.")
    recalls, f1 = [], []
    for label in range(n_classes):
        true = gold == label
        if true.any():
            recalls.append(float((predicted[true] == label).mean()))
        tp = int(((predicted == label) & true).sum())
        fp = int(((predicted == label) & ~true).sum())
        fn = int(((predicted != label) & true).sum())
        denominator = 2 * tp + fp + fn
        f1.append(2 * tp / denominator if denominator else 0.0)
    return {"WA": float((predicted == gold).mean()), "UAR": float(np.mean(recalls)), "macroF1": float(np.mean(f1))}


def _moments_tensors(mean, std, device):
    return [torch.as_tensor(x, device=device).view(1, *x.shape[:1], 1, x.shape[-1]) for x in [mean, std]]


@torch.inference_mode()
def predict(model, store, indices, mean, std, device, batch_size=32, frame_budget=8192):
    """Return probabilities in the requested utterance order, with no test-time shifts."""
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("Prediction indices must be non-empty and unique.")
    model.eval()
    mean_t, std_t = _moments_tensors(mean, std, device)
    position = {index: row for row, index in enumerate(indices)}
    probabilities = np.empty((len(indices), len(store.labels)), np.float32)
    for batch in batch_indices(store, indices, batch_size, frame_budget):
        x, valid = make_batch(store, batch, mean_t, std_t, device)
        logits, _ = model(x, valid)
        values = torch.softmax(logits.float(), -1).cpu().numpy()
        if not np.isfinite(values).all():
            raise FloatingPointError("Prediction produced non-finite probabilities.")
        for row, index in enumerate(batch):
            probabilities[position[index]] = values[row]
    return probabilities


def train_fold(store, held_out, output, model_config=None, config=None, device="cpu"):
    model_config, config = model_config or ModelConfig(), config or TrainConfig()
    if model_config.n_layers != len(LAYERS) or model_config.input_dim != store.width:
        raise ValueError("The model configuration must match the three selected cache layers and their width.")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty output directory for each training run.")
    training, held = store.split(held_out)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    mean, std = training_moments(store, training)
    mean_t, std_t = _moments_tensors(mean, std, device)
    model = AGAPNet(len(store.labels), store.anchors, model_config).to(device)
    targets = torch.as_tensor(store.anchors, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    ema = EMA(model, config.ema_decay)
    nominal_steps = max(1, len(training) // config.batch_size + 1)
    warm = nominal_steps * config.warmup_epochs
    horizon = nominal_steps * config.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: step / warm if step < warm else
        0.5 * (1 + np.cos(np.pi * (step - warm) / max(1, horizon - warm)))
    )
    batches = batch_indices(store, training, config.batch_size, config.frame_budget, config.crop_frames)
    steps = 0
    for epoch in range(config.epochs):
        model.train()
        order = np.random.default_rng(config.seed * 1000 + epoch).permutation(len(batches))
        total_loss = 0.0
        for batch_position in order:
            indices = batches[int(batch_position)]
            x, valid = make_batch(store, indices, mean_t, std_t, device, training=True, crop=config.crop_frames)
            y = torch.tensor([store.utts[i]["y"] for i in indices], device=device)
            logits, auxiliary = model(x, valid)
            loss = objective(model, logits, auxiliary, y, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss is non-finite.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            ema.update(model)
            steps += 1
            total_loss += float(loss.detach())
        print(f"Epoch {epoch+1}/{config.epochs}: loss={total_loss/len(batches):.6f}", flush=True)
    model.load_state_dict(ema.shadow)
    probabilities = predict(model, store, held, mean, std, device, config.batch_size, config.frame_budget)
    gold = np.asarray([store.utts[i]["y"] for i in held], dtype=np.int64)
    score = metrics(probabilities.argmax(-1), gold, len(store.labels))
    output.mkdir(parents=True, exist_ok=True)
    metadata_hash = hashlib.sha256((store.directory / "meta.json").read_bytes()).hexdigest()
    provenance = {"cache_metadata_sha256": metadata_hash, "torch": str(torch.__version__),
                  "numpy": np.__version__, "device": str(device), "test_time_shifts": 0,
                  "padding_multiple": 32, "normalization_max_frames": 120000,
                  "completed_optimizer_steps": steps,
                  "extraction": store.meta.get("extraction", {})}
    checkpoint = {"format_version": 1, "model_config": asdict(model_config), "training_config": asdict(config),
                  "labels": store.labels, "anchors": store.anchors.tolist(), "layers": list(LAYERS),
                  "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                  "mean": torch.from_numpy(mean), "std": torch.from_numpy(std), "held_out": held_out,
                  "provenance": provenance}
    torch.save(checkpoint, output / "model.pt")
    ids = np.asarray([store.utts[i]["utt_id"] for i in held])
    np.savez_compressed(output / "predictions.npz", utterance_ids=ids, probabilities=probabilities, gold=gold)
    summary = {"held_out": held_out, "seed": config.seed, "epochs": config.epochs,
               "n_train": len(training), "n_test": len(held), "labels": store.labels,
               "parameters": sum(p.numel() for p in model.parameters()), "metrics": score,
               "model_config": asdict(model_config), "training_config": asdict(config), "provenance": provenance}
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("layers") != list(LAYERS):
        raise ValueError("Unsupported AGAP-Net checkpoint format or layer selection.")
    model = AGAPNet(len(checkpoint["labels"]), checkpoint["anchors"], ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint
