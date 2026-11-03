import math
import json
import sys

import numpy as np
import pytest
import torch

from agapnet.data import batch_indices, training_moments
from agapnet.engine import TrainConfig, load_checkpoint, metrics, objective, predict, soft_label_matrix, train_fold
from agapnet.features import encode_waveforms, frames_of, waveform_batch
from agapnet.model import AGAPNet, AttentiveStatPool, ModelConfig


@pytest.mark.parametrize("ratio", [0.0, 0.5])
def test_complete_model_gradients_and_class_initialization(ratio):
    torch.manual_seed(5)
    anchors = torch.tensor([[-0.62, 0.78, 0.42], [0.0, 0.0, 0.0]])
    model = AGAPNet(2, anchors, ModelConfig(moment_ratio=ratio))
    initial = model.proto.prototypes().detach() / math.sqrt(256)
    for block in [*model.time_blocks, model.freq_block]:
        torch.testing.assert_close(block.attn.agents[:2], initial)
        assert block.attn.agents.shape == (4, 256)
    valid = torch.arange(32)[None] < torch.tensor([12, 25])[:, None]
    logits, auxiliary = model(torch.randn(2, 3, 32, 1920), valid)
    assert logits.shape == (2, 2) and auxiliary["vad"].shape == (2, 3)
    loss = objective(model, logits, auxiliary, torch.tensor([0, 1]), anchors)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_moment_selection_preserves_full_attention():
    pool = AttentiveStatPool(1, moment_ratio=0.5)
    with torch.no_grad():
        pool.score.weight.zero_()
        pool.score.bias.zero_()
    result = pool(torch.tensor([[[1.0], [4.0], [3.0], [100.0]]]), torch.tensor([[True, True, True, False]]))
    torch.testing.assert_close(result[0, 0], torch.tensor(8 / 3))
    torch.testing.assert_close(result[0, 1], torch.tensor(4.0))


def test_soft_targets_and_missing_class_metrics():
    targets = soft_label_matrix([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    torch.testing.assert_close(targets.sum(-1), torch.ones(2))
    score = metrics(np.array([0, 0]), np.array([0, 0]), 2)
    assert score == {"WA": 1.0, "UAR": 1.0, "macroF1": 0.5}


def test_training_moments_ignore_held_out_frames(cache):
    train, held = cache.split("speaker_02")
    mean, std = training_moments(cache, train)
    values = np.concatenate([cache.read(i) for i in train], axis=1).astype(np.float32)
    reference_mean = values.sum(1, dtype=np.float32).astype(np.float64) / values.shape[1]
    reference_std = np.sqrt(np.maximum((values.astype(np.float64)**2).sum(1) / values.shape[1] - reference_mean**2, 1e-10)) + 1e-5
    np.testing.assert_allclose(mean, reference_mean.astype(np.float32), atol=1e-7)
    np.testing.assert_allclose(std, reference_std.astype(np.float32), atol=1e-7)
    assert set(train).isdisjoint(held)
    assert max(train) < min(held)
    batches = batch_indices(cache, list(range(6)), batch_size=2, frame_budget=64)
    assert sorted(i for batch in batches for i in batch) == list(range(6))
    assert all(len(batch) <= 2 for batch in batches)


def test_waveform_masks_use_samples():
    waves = [np.zeros(1600, np.float32), np.ones(960, np.float32)]
    batch, mask, lengths = waveform_batch(waves, "cpu", torch.float32)
    assert batch.shape == mask.shape == (2, 1600)
    assert mask.sum(1).tolist() == lengths == [1600, 960]
    assert mask[1, 960:].sum() == 0
    assert frames_of(1600) == 4


def test_tiny_frozen_encoder_lengths():
    transformers = pytest.importorskip("transformers")
    config = transformers.Wav2Vec2Config(hidden_size=16, num_hidden_layers=3, num_attention_heads=2,
        intermediate_size=32, conv_dim=(8, 8, 8), conv_stride=(2, 2, 2), conv_kernel=(5, 3, 3),
        num_conv_pos_embedding_groups=2, num_conv_pos_embeddings=16, feat_extract_norm="layer")
    encoder = transformers.Wav2Vec2Model(config)
    waves = [np.random.default_rng(i).normal(size=n).astype(np.float32) for i, n in enumerate([400, 640])]
    results = encode_waveforms(encoder, waves, layers=(1, 2, 3))
    assert [x.shape for x in results] == [(3, frames_of(len(w), config.conv_kernel, config.conv_stride), 16) for w in waves]
    assert all(x.dtype == np.float16 and np.isfinite(x).all() for x in results)
    assert all(not p.requires_grad for p in encoder.parameters())


def test_checkpoint_prediction_order_and_round_trip(cache, tmp_path):
    output = tmp_path / "run"
    summary = train_fold(cache, "speaker_02", output, config=TrainConfig(epochs=1), device="cpu")
    assert summary["n_train"] == summary["n_test"] == 3
    model, checkpoint = load_checkpoint(output / "model.pt")
    held = cache.split("speaker_02")[1]
    actual = predict(model, cache, held, checkpoint["mean"].numpy(), checkpoint["std"].numpy(), "cpu")
    with np.load(output / "predictions.npz", allow_pickle=False) as saved:
        np.testing.assert_allclose(actual, saved["probabilities"], atol=1e-7)
        assert saved["utterance_ids"].tolist() == [cache.utts[i]["utt_id"] for i in held]
        assert saved["gold"].tolist() == [cache.utts[i]["y"] for i in held]
    assert checkpoint["provenance"]["test_time_shifts"] == 0


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        ModelConfig(moment_ratio=1.5)
    model = AGAPNet(2, [[0, 0, 0], [0.5, 0.5, 0.5]])
    with pytest.raises(ValueError):
        model(torch.randn(1, 3, 32, 1920), torch.zeros(1, 32, dtype=torch.bool))


def test_training_and_prediction_commands(cache, tmp_path, monkeypatch):
    from agapnet.train import main as train_main
    from agapnet.predict import main as predict_main

    output = tmp_path / "commands"
    monkeypatch.setattr(sys, "argv", ["agapnet-train", "--cache", str(cache.directory),
                                     "--output", str(output), "--epochs", "1", "--device", "cpu"])
    train_main()
    assert len(json.loads((output / "summary.json").read_text())["folds"]) == 2
    checkpoint = sorted(output.glob("*/model.pt"))[0]
    destination = tmp_path / "predictions.jsonl"
    monkeypatch.setattr(sys, "argv", ["agapnet-predict", "--checkpoint", str(checkpoint),
                                     "--cache", str(cache.directory), "--output", str(destination),
                                     "--device", "cpu"])
    predict_main()
    rows = [json.loads(line) for line in destination.read_text().splitlines()]
    assert len(rows) == 3
    assert len({r["utterance_id"] for r in rows}) == 3
    assert all(abs(sum(r["probabilities"].values()) - 1) < 1e-6 for r in rows)
