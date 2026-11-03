"""Predict emotion from a saved AGAP-Net checkpoint and audio or cached features."""

import argparse
import json
from pathlib import Path

from .data import FeatureStore, load_audio
from .engine import choose_device, load_checkpoint, predict


class AudioFeatures:
    def __init__(self, audio_paths, features, checkpoint):
        self.features = features
        self.labels = checkpoint["labels"]
        self.width = features[0].shape[-1]
        self.utts = [{"utt_id": f"audio_{i:05d}_{Path(path).stem}", "length": f.shape[1]}
                     for i, (path, f) in enumerate(zip(audio_paths, features))]

    def read(self, index, offset=0, length=None):
        values = self.features[index]
        end = values.shape[1] if length is None else offset + length
        return values[:, offset:end]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--cache")
    source.add_argument("--audio", nargs="+")
    parser.add_argument("--group", help="For cached features, predict this group; default is the checkpoint's held-out group.")
    parser.add_argument("--output", required=True, help="Destination JSONL file.")
    parser.add_argument("--encoder", default="facebook/wav2vec2-xls-r-2b")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Choose a new prediction file.")
    device = choose_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    if args.cache:
        store = FeatureStore(args.cache)
        if store.labels != checkpoint["labels"]:
            raise ValueError("The cache label order must match the checkpoint vocabulary.")
        group = args.group or checkpoint["held_out"]
        if group not in store.groups():
            raise ValueError("The requested group is absent; specify an existing --group explicitly.")
        indices = [i for i, utterance in enumerate(store.utts) if utterance["group"] == group]
    else:
        if args.group:
            raise ValueError("--group is used only with --cache.")
        from .features import encode_waveforms, load_encoder

        encoder = load_encoder(args.encoder, args.revision, device)
        features = [encode_waveforms(encoder, [load_audio(path)])[0] for path in args.audio]
        store = AudioFeatures(args.audio, features, checkpoint)
        indices = list(range(len(features)))
        del encoder
    cfg = checkpoint["training_config"]
    values = predict(model, store, indices, checkpoint["mean"].numpy(), checkpoint["std"].numpy(),
                     device, cfg["batch_size"], cfg["frame_budget"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        for index, probabilities in zip(indices, values):
            result = {"utterance_id": store.utts[index]["utt_id"],
                      "predicted_label": store.labels[int(probabilities.argmax())],
                      "probabilities": {label: float(p) for label, p in zip(store.labels, probabilities)}}
            stream.write(json.dumps(result) + "\n")
    print(output)


if __name__ == "__main__":
    main()
