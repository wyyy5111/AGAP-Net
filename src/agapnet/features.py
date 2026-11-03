"""Frozen XLS-R feature extraction with sample-level attention masks."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch

from .data import LAYERS, VAD_PRIORS, load_audio, read_manifest


def frames_of(samples, kernels=(10, 3, 3, 3, 3, 2, 2), strides=(5, 2, 2, 2, 2, 2, 2)):
    for kernel, stride in zip(kernels, strides):
        samples = (samples - kernel) // stride + 1
    return samples


def waveform_batch(waveforms, device, dtype):
    """Both the input tensor and its attention mask use waveform samples."""
    if not waveforms or any(w.ndim != 1 or not len(w) or not np.isfinite(w).all() for w in waveforms):
        raise ValueError("Provide non-empty, finite mono waveforms.")
    lengths = [len(w) for w in waveforms]
    batch = torch.zeros(len(waveforms), max(lengths), device=device, dtype=dtype)
    mask = torch.zeros(len(waveforms), max(lengths), device=device, dtype=torch.long)
    for row, waveform in enumerate(waveforms):
        batch[row, :len(waveform)] = torch.as_tensor(waveform, device=device, dtype=dtype)
        mask[row, :len(waveform)] = 1
    return batch, mask, lengths


@torch.inference_mode()
def encode_waveforms(encoder, waveforms, layers=LAYERS):
    encoder.requires_grad_(False).eval()
    parameter = next(encoder.parameters())
    batch, mask, lengths = waveform_batch(waveforms, parameter.device, parameter.dtype)
    output = encoder(batch, attention_mask=mask, output_hidden_states=True)
    result = []
    for row, samples in enumerate(lengths):
        count = frames_of(samples, encoder.config.conv_kernel, encoder.config.conv_stride)
        if count < 1:
            raise ValueError("The waveform is shorter than the encoder receptive field.")
        states = torch.stack([output.hidden_states[layer][row, :count] for layer in layers])
        if states.shape[1] != count or not torch.isfinite(states).all():
            raise ValueError("Invalid extracted feature shape or values.")
        result.append(states.float().cpu().numpy().astype(np.float16))
    return result


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_encoder(model_id, revision=None, device="cpu"):
    from transformers import Wav2Vec2Model

    dtype = torch.float32 if torch.device(device).type == "cpu" else torch.float16
    encoder = Wav2Vec2Model.from_pretrained(model_id, revision=revision, torch_dtype=dtype)
    if encoder.config.hidden_size != 1920 or encoder.config.num_hidden_layers != 48:
        raise ValueError("The core extractor expects the XLS-R-2B encoder (48 layers, 1920 channels).")
    return encoder.to(device).requires_grad_(False).eval()


def extract(manifest, output, model_id="facebook/wav2vec2-xls-r-2b", revision=None,
            device="cpu", batch_size=1):
    import soundfile as sf

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    rows = read_manifest(manifest)
    labels = sorted({row["label"] for row in rows})
    if len(labels) < 2 or any(label not in VAD_PRIORS for label in labels):
        raise ValueError("Use at least two labels defined in agapnet.data.VAD_PRIORS.")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty output directory; existing caches are never overwritten.")
    counts = []
    for row in rows:
        info = sf.info(row["audio"])
        samples = (info.frames * 16000 + info.samplerate - 1) // info.samplerate
        count = frames_of(samples)
        if count < 1:
            raise ValueError(f"Audio is too short: {row['id']}")
        counts.append(count)
    encoder = load_encoder(model_id, revision, device)
    output.mkdir(parents=True, exist_ok=True)
    partial = output / "feats.npy.partial"
    array = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float16,
                                     shape=(len(LAYERS), sum(counts), 1920))
    offset, entries = 0, []
    for first in range(0, len(rows), batch_size):
        selected = rows[first:first+batch_size]
        waves = [load_audio(row["audio"]) for row in selected]
        states = encode_waveforms(encoder, waves)
        for position, (row, wave, feature) in enumerate(zip(selected, waves, states), first):
            if feature.shape[1] != counts[position]:
                raise ValueError("Audio length changed while building the cache.")
            count = feature.shape[1]
            array[:, offset:offset+count] = feature
            entries.append({"utt_id": row["id"], "label": row["label"], "group": row["group"],
                            "y": labels.index(row["label"]), "start": offset, "length": count,
                            "valid_samples": len(wave), "audio_sha256": file_sha256(row["audio"])})
            offset += count
        print(f"Extracted {len(entries)}/{len(rows)} utterances", flush=True)
    array.flush()
    shape = list(array.shape)
    del array
    partial.rename(output / "feats.npy")
    metadata = {"format_version": 1, "labels": labels, "groups": sorted({r["group"] for r in rows}),
                "vad_anchors": {label: VAD_PRIORS[label] for label in labels},
                "layers": list(LAYERS), "feats_shape": shape, "n_utts": len(entries), "utts": entries,
                "extraction": {"model": model_id, "requested_revision": revision,
                               "resolved_revision": getattr(encoder.config, "_commit_hash", None),
                               "sample_rate": 16000, "attention_mask_units": "waveform_samples",
                               "torch": str(torch.__version__),
                               "transformers": importlib.metadata.version("transformers")}}
    (output / "meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "DONE").write_text("ok\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="facebook/wav2vec2-xls-r-2b")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    extract(args.manifest, args.output, args.model, args.revision, args.device, args.batch_size)


if __name__ == "__main__":
    main()
