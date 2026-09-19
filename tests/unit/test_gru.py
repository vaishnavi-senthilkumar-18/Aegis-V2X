"""Focused tests for `ai.gru.model.GRUPredictor`.

SYNTHETIC / CODE-CORRECTNESS VALIDATION ONLY. Every tensor here is randomly
generated purely to unit-test the architecture's shape/determinism/error
handling in isolation -- NOT real wireless data. No real CSI/SNR/beam_index
values exist anywhere in the Aegis-V2X Phase 2 dataset (confirmed in the
Phase 4 GRU audit: `csi`/`snr_db`/`rssi_dbm`/`path_loss_db`/`beam_index` are
`None` for every frame in all 5 real scenes), so no real-data validation is
performed or claimed for this module. These tests establish only that the
architecture is wired correctly, not that it predicts anything meaningful.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ai.gru.model import GRUPredictor, GRUPrediction, InvalidSequenceError, VALID_HORIZONS

TEST_CONFIG = {
    "perception": {
        "gru": {
            "hidden_size": 16,
            "num_layers": 2,
            "input_features": ["csi", "snr", "beam_index", "mobility"],
        }
    }
}

INPUT_DIM = 4  # arbitrary flat feature width for synthetic tests


def _make_model() -> GRUPredictor:
    return GRUPredictor(config=TEST_CONFIG, input_dim=INPUT_DIM, seed=0)


def _synthetic_sequence(batch: int = 1, seq_len: int = 10, seed: int = 0) -> torch.Tensor:
    """SYNTHETIC (code-correctness only) -- not real CSI/SNR/mobility data."""
    rng = np.random.default_rng(seed)
    arr = rng.normal(size=(batch, seq_len, INPUT_DIM)).astype(np.float32)
    return torch.from_numpy(arr)


def test_model_can_be_instantiated():
    model = _make_model()
    assert model.hidden_size == 16
    assert model.num_layers == 2
    assert model.input_features == ["csi", "snr", "beam_index", "mobility"]


def test_valid_synthetic_sequence_produces_prediction():
    """SYNTHETIC input -- validates architecture wiring only."""
    model = _make_model()
    seq = _synthetic_sequence(batch=1, seq_len=10)
    result = model.predict(seq, horizon=5)
    assert isinstance(result, GRUPrediction)


def test_batch_input_is_supported():
    """SYNTHETIC input -- validates batch handling only."""
    model = _make_model()
    seq = _synthetic_sequence(batch=8, seq_len=6)
    result = model.predict(seq, horizon=1)
    assert result.predicted_channel_state.shape == (8, INPUT_DIM)
    assert result.prediction_uncertainty.shape == (8, INPUT_DIM)


@pytest.mark.parametrize("horizon", VALID_HORIZONS)
def test_all_documented_horizons_are_accepted(horizon):
    """SYNTHETIC input -- exercises every horizon in the documented
    {1,2,3,5,8,10} set from ai/gru/base.py / TAHS discretization."""
    model = _make_model()
    seq = _synthetic_sequence(batch=2, seq_len=5)
    result = model.predict(seq, horizon=horizon)
    assert result.horizon == horizon


def test_different_horizons_produce_different_predictions():
    """SYNTHETIC input -- confirms horizon conditioning actually affects
    the output (not silently ignored)."""
    model = _make_model()
    seq = _synthetic_sequence(batch=1, seq_len=5)
    out_h1 = model.predict(seq, horizon=1)
    out_h10 = model.predict(seq, horizon=10)
    assert not torch.allclose(out_h1.predicted_channel_state, out_h10.predicted_channel_state)


def test_prediction_output_shape():
    model = _make_model()
    seq = _synthetic_sequence(batch=3, seq_len=7)
    result = model.predict(seq, horizon=3)
    assert result.predicted_channel_state.shape == (3, INPUT_DIM)


def test_uncertainty_output_shape_and_nonnegativity():
    model = _make_model()
    seq = _synthetic_sequence(batch=3, seq_len=7)
    result = model.predict(seq, horizon=3)
    assert result.prediction_uncertainty.shape == (3, INPUT_DIM)
    assert bool((result.prediction_uncertainty >= 0).all())


def test_output_fields_are_present_and_named():
    model = _make_model()
    seq = _synthetic_sequence(batch=1, seq_len=4)
    result = model.predict(seq, horizon=2)
    assert hasattr(result, "predicted_channel_state")
    assert hasattr(result, "prediction_uncertainty")
    assert hasattr(result, "horizon")


def test_outputs_are_finite():
    model = _make_model()
    seq = _synthetic_sequence(batch=4, seq_len=8)
    result = model.predict(seq, horizon=8)
    assert torch.isfinite(result.predicted_channel_state).all()
    assert torch.isfinite(result.prediction_uncertainty).all()


def test_deterministic_eval_mode_inference():
    model = _make_model()
    seq = _synthetic_sequence(batch=2, seq_len=6)
    out1 = model.predict(seq, horizon=5)
    out2 = model.predict(seq, horizon=5)
    assert torch.equal(out1.predicted_channel_state, out2.predicted_channel_state)
    assert torch.equal(out1.prediction_uncertainty, out2.prediction_uncertainty)


def test_empty_sequence_raises():
    model = _make_model()
    empty_seq = torch.zeros((1, 0, INPUT_DIM))
    with pytest.raises(InvalidSequenceError):
        model.predict(empty_seq, horizon=1)


def test_empty_batch_raises():
    model = _make_model()
    empty_batch = torch.zeros((0, 5, INPUT_DIM))
    with pytest.raises(InvalidSequenceError):
        model.predict(empty_batch, horizon=1)


def test_wrong_feature_dimension_raises():
    model = _make_model()
    wrong_dim = torch.zeros((1, 5, INPUT_DIM + 1))
    with pytest.raises(InvalidSequenceError):
        model.predict(wrong_dim, horizon=1)


def test_wrong_ndim_raises():
    model = _make_model()
    with pytest.raises(InvalidSequenceError):
        model.predict(torch.zeros((5, INPUT_DIM)), horizon=1)


def test_non_finite_input_raises():
    model = _make_model()
    seq = _synthetic_sequence(batch=1, seq_len=5)
    seq[0, 0, 0] = float("nan")
    with pytest.raises(InvalidSequenceError):
        model.predict(seq, horizon=1)


def test_non_tensor_input_raises():
    model = _make_model()
    with pytest.raises(InvalidSequenceError):
        model.predict([[1.0, 2.0, 3.0, 4.0]], horizon=1)


@pytest.mark.parametrize("bad_horizon", [0, 4, 7, 11, -1])
def test_invalid_horizon_raises(bad_horizon):
    model = _make_model()
    seq = _synthetic_sequence(batch=1, seq_len=5)
    with pytest.raises(InvalidSequenceError):
        model.predict(seq, horizon=bad_horizon)
