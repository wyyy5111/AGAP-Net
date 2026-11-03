"""Portable manifests, memory-mapped feature caches, and training-fold moments."""

import json
import math
from pathlib import Path
import random

import numpy as np
import torch

LAYERS = (16, 24, 32)
VAD_PRIORS = {
    "anger": (-0.62, 0.78, 0.42), "happiness": (0.81, 0.60, 0.44),
    "neutral": (0.00, 0.00, 0.00), "sadness": (-0.63, -0.27, -0.36),
    "fear": (-0.60, 0.40, -0.42), "surprise": (0.30, 0.62, 0.02),
    "boredom": (-0.43, -0.52, -0.32), "disgust": (-0.62, 0.20, 0.28),
}


def read_manifest(path):
    """Read one JSON object per utterance; relative audio paths use the manifest directory."""
    path = Path(path).resolve()
    rows = []
    seen = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not all(isinstance(row.get(k), str) and row[k] for k in ["id", "audio", "label", "group"]):
            raise ValueError("Each manifest row needs non-empty id, audio, label, and group strings.")
        if row["id"] in seen:
            raise ValueError(f"Duplicate utterance ID: {row['id']}")
        seen.add(row["id"])
        audio = Path(row["audio"])
        row["audio"] = str((audio if audio.is_absolute() else path.parent / audio).resolve())
        if not Path(row["audio"]).is_file():
            raise FileNotFoundError(row["audio"])
        rows.append(row)
    if not rows:
        raise ValueError("The manifest contains no utterances.")
    return rows


def load_audio(path, target_sr=16000):
    import soundfile as sf
    from scipy.signal import resample_poly

    waveform, sr = sf.read(path, dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    if not len(waveform) or not np.isfinite(waveform).all():
        raise ValueError("Audio must be non-empty and finite.")
    if sr != target_sr:
        divisor = math.gcd(sr, target_sr)
        waveform = resample_poly(waveform, target_sr // divisor, sr // divisor).astype("float32")
    return (waveform - waveform.mean()) / (waveform.std() + 1e-7)


class FeatureStore:
    """Read both newly extracted and existing [layers, frames, channels] caches."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.meta = json.loads((self.directory / "meta.json").read_text())
        self.array = np.load(self.directory / "feats.npy", mmap_mode="r", allow_pickle=False)
        if self.array.ndim != 3 or not np.issubdtype(self.array.dtype, np.floating):
            raise ValueError("feats.npy must be a floating-point [layers, frames, channels] array.")
        self.labels = self.meta["labels"]
        if len(self.labels) != len(set(self.labels)) or len(self.labels) < 2:
            raise ValueError("Provide at least two distinct emotion labels.")
        cached_layers = self.meta["layers"]
        if self.array.shape[0] != len(cached_layers) or len(set(cached_layers)) != len(cached_layers):
            raise ValueError("Cached layer metadata does not match the feature array.")
        self.positions = [cached_layers.index(layer) for layer in LAYERS]
        self.width = self.array.shape[-1]
        self.utts = self.meta["utts"]
        self.anchors = np.asarray([self.meta["vad_anchors"][s] for s in self.labels], dtype=np.float32)
        if self.anchors.shape != (len(self.labels), 3) or not np.isfinite(self.anchors).all():
            raise ValueError("Provide a finite VAD prior for every label.")
        seen = set()
        for utterance in self.utts:
            uid = utterance["utt_id"]
            if not isinstance(uid, str) or not uid or uid in seen:
                raise ValueError("Utterance IDs must be unique non-empty strings.")
            seen.add(uid)
            a, n = utterance["start"], utterance["length"]
            if not isinstance(a, int) or not isinstance(n, int) or a < 0 or n < 1 or a + n > self.array.shape[1]:
                raise ValueError(f"Invalid feature interval for {uid}.")
            if utterance["label"] not in self.labels or utterance["y"] != self.labels.index(utterance["label"]):
                raise ValueError(f"Inconsistent label for {uid}.")
            if not isinstance(utterance["group"], str) or not utterance["group"]:
                raise ValueError(f"Missing held-out group for {uid}.")
        if not self.utts:
            raise ValueError("The cache contains no utterances.")

    def read(self, index, offset=0, length=None):
        utterance = self.utts[index]
        n = utterance["length"] - offset if length is None else length
        if offset < 0 or n < 1 or offset + n > utterance["length"]:
            raise ValueError("Requested frames lie outside the utterance interval.")
        start = utterance["start"] + offset
        return np.stack([self.array[layer, start:start+n, :] for layer in self.positions])

    def groups(self):
        return sorted({u["group"] for u in self.utts})

    def split(self, held_out):
        if held_out not in self.groups():
            raise ValueError(f"Unknown held-out group: {held_out}")
        training = [i for i, u in enumerate(self.utts) if u["group"] != held_out]
        test = [i for i, u in enumerate(self.utts) if u["group"] == held_out]
        if not training or not test:
            raise ValueError("Both the training and held-out partitions must be non-empty.")
        return training, test


def training_moments(store, indices, max_frames=120000, chunk=8000):
    """Accumulate the selected training-frame grid without materializing a whole fold."""
    lengths = np.asarray([store.utts[i]["length"] for i in indices], dtype=np.int64)
    if not len(lengths) or (lengths < 1).any():
        raise ValueError("Moment estimation requires non-empty training utterances.")
    ends = np.cumsum(lengths)
    starts = np.concatenate([[0], ends[:-1]])
    selected = np.arange(0, ends[-1], max(1, int(ends[-1]) // max_frames))
    total = np.zeros((len(LAYERS), store.width), np.float64)
    squares = np.zeros_like(total)
    for first in range(0, len(selected), chunk):
        positions = selected[first:first+chunk]
        membership = np.searchsorted(ends, positions, side="right")
        x = np.empty((len(LAYERS), len(positions), store.width), dtype=np.float32)
        for member in np.unique(membership):
            where = np.flatnonzero(membership == member)
            utterance = store.utts[indices[member]]
            frames = utterance["start"] + positions[where] - starts[member]
            for layer, cached in enumerate(store.positions):
                x[layer, where, :] = store.array[cached, frames, :]
        total += x.sum(axis=1)
        squares += (x.astype(np.float64) ** 2).sum(axis=1)
    mean = total / len(selected)
    std = np.sqrt(np.maximum(squares / len(selected) - mean**2, 1e-10)) + 1e-5
    return mean.astype(np.float32), std.astype(np.float32)


def batch_indices(store, indices, batch_size=32, frame_budget=8192, crop=None):
    """Sort by length and respect the budget after rounding padding to 32 frames."""
    batches, current, maximum = [], [], 0
    for index in sorted(indices, key=lambda i: store.utts[i]["length"]):
        length = store.utts[index]["length"]
        if crop:
            length = min(length, crop)
        rounded = ((length + 31) // 32) * 32
        limit = max(maximum, rounded)
        if current and (len(current) >= batch_size or (len(current) + 1) * limit > frame_budget):
            batches.append(current)
            current, maximum = [], 0
        current.append(index)
        maximum = max(maximum, rounded)
    if current:
        batches.append(current)
    return batches


def make_batch(store, indices, mean, std, device, training=False, crop=400):
    lengths = [min(store.utts[i]["length"], crop) if training else store.utts[i]["length"] for i in indices]
    maximum = ((max(lengths) + 31) // 32) * 32
    x = np.zeros((len(indices), len(LAYERS), maximum, store.width), dtype=np.float16)
    for row, (index, length) in enumerate(zip(indices, lengths)):
        available = store.utts[index]["length"]
        start = random.randint(0, available - length) if training and available > length else 0
        x[row, :, :length, :] = store.read(index, start, length)
    tensor = torch.from_numpy(x).to(device).float()
    tensor = (tensor - mean) / std
    valid = torch.arange(maximum, device=device)[None, :] < torch.tensor(lengths, device=device)[:, None]
    if training:
        time_mask = torch.ones(len(indices), maximum, device=device)
        for row, length in enumerate(lengths):
            width = max(1, int(0.1 * length))
            for _ in range(2):
                start = random.randint(0, max(0, length - width))
                time_mask[row, start:start+width] = 0
        tensor = tensor * time_mask[:, None, :, None]
        keep = (torch.rand(len(indices), len(LAYERS), 1, store.width, device=device) > 0.05).float()
        tensor = tensor * keep + 0.05 * torch.randn_like(tensor)
    return tensor, valid
