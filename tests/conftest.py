import json

import numpy as np
import pytest
import torch

from agapnet.data import FeatureStore, VAD_PRIORS


@pytest.fixture(scope="session", autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


@pytest.fixture
def cache(tmp_path):
    directory = tmp_path / "cache"
    directory.mkdir()
    labels = ["anger", "neutral"]
    lengths = [7, 13, 9, 17, 11, 15]
    values = np.random.default_rng(13).normal(size=(3, sum(lengths), 1920)).astype(np.float16)
    entries, start = [], 0
    for i, length in enumerate(lengths):
        label = labels[i % 2]
        entries.append({"utt_id": f"sample_{i}", "label": label, "y": i % 2,
                        "group": "speaker_01" if i < 3 else "speaker_02",
                        "start": start, "length": length})
        start += length
    np.save(directory / "feats.npy", values)
    (directory / "meta.json").write_text(json.dumps({"labels": labels, "layers": [16, 24, 32],
        "vad_anchors": {s: VAD_PRIORS[s] for s in labels}, "utts": entries}))
    return FeatureStore(directory)
