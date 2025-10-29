import numpy as np
import pytest
from collections.abc import Mapping

pytest.importorskip("torch")
import torch

from psann import PSANNRegressor, StateConfig, WaveResNetRegressor
from psann.episodes import multiplicative_return_reward
from psann.hisso import hisso_evaluate_reward, hisso_infer_series
from psann.utils import seed_all


def _make_dataset(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((240, 6)).astype(np.float32)
    y = (
        np.sin(X[:, 0]) * 0.4
        + 0.25 * X[:, 1]
        - 0.15 * X[:, 2] ** 2
        + 0.1 * X[:, 3] * X[:, 4]
    ).astype(np.float32)
    return X, y


def test_save_load_roundtrip_preserves_predictions(tmp_path):
    seed_all(7)
    X, y = _make_dataset(seed=11)

    X_tr, y_tr = X[:120], y[:120]
    X_va, y_va = X[120:160], y[120:160]
    X_te, y_te = X[160:], y[160:]

    model = PSANNRegressor(
        hidden_layers=1,
        hidden_units=48,
        epochs=40,
        batch_size=48,
        lr=5e-3,
        early_stopping=True,
        patience=8,
        random_state=5,
    )
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va), verbose=0)

    preds_before = model.predict(X_te)
    score_before = model.score(X_te, y_te)

    checkpoint_path = tmp_path / "psann_checkpoint.pt"
    model.save(str(checkpoint_path))

    restored = PSANNRegressor.load(str(checkpoint_path))
    preds_after = restored.predict(X_te)
    score_after = restored.score(X_te, y_te)

    np.testing.assert_allclose(preds_after, preds_before, rtol=1e-6, atol=1e-6)
    assert score_after == pytest.approx(score_before, rel=1e-6, abs=1e-6)


def test_predict_sequence_shapes_and_online_updates():
    seed_all(17)
    X, y = _make_dataset(seed=23)
    X_tr, y_tr = X[:120], y[:120]
    X_va, y_va = X[120:160], y[120:160]
    X_seq, y_seq = X[160:180], y[160:180]

    model = PSANNRegressor(
        hidden_layers=1,
        hidden_units=32,
        epochs=35,
        batch_size=48,
        lr=3e-3,
        early_stopping=True,
        patience=8,
        stateful=True,
        stream_lr=1e-3,
        state=StateConfig(rho=0.95, beta=1.0, max_abs=5.0, init=0.0, detach=True),
        random_state=13,
    )
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va), verbose=0)

    seq_last = model.predict_sequence(X_seq, reset_state=True, return_sequence=False)
    assert np.isscalar(seq_last), "Expected scalar prediction when return_sequence=False"

    seq_full = model.predict_sequence(X_seq, reset_state=True, return_sequence=True)
    assert isinstance(seq_full, np.ndarray)
    assert seq_full.shape == (X_seq.shape[0],)

    online_seq = model.predict_sequence_online(
        X_seq, y_seq, reset_state=True, return_sequence=True
    )
    assert isinstance(online_seq, np.ndarray)
    assert online_seq.shape == (X_seq.shape[0],)

    seq_last_no_state = model.predict_sequence(
        X_seq, reset_state=True, return_sequence=False, update_state=False
    )
    direct_last = model.predict(X_seq[-1:])
    assert np.allclose(seq_last_no_state, direct_last[-1], atol=1e-5)


def test_save_load_roundtrip_preserves_context_builder(tmp_path):
    seed_all(29)
    X, y = _make_dataset(seed=31)
    X_tr, y_tr = X[:120], y[:120]
    X_va, y_va = X[120:160], y[120:160]
    X_te, y_te = X[160:], y[160:]

    model = WaveResNetRegressor(
        hidden_layers=2,
        hidden_units=40,
        epochs=25,
        batch_size=32,
        lr=4e-3,
        early_stopping=True,
        patience=5,
        context_builder="cosine",
        context_builder_params={"frequencies": 2, "include_cos": False},
        random_state=17,
    )
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va), verbose=0)

    assert model._context_dim_ is not None and model._context_dim_ > 0

    preds_before = model.predict(X_te)
    score_before = model.score(X_te, y_te)

    checkpoint_path = tmp_path / "psann_context_checkpoint.pt"
    model.save(str(checkpoint_path))

    restored = WaveResNetRegressor.load(str(checkpoint_path))
    preds_after = restored.predict(X_te)
    score_after = restored.score(X_te, y_te)

    np.testing.assert_allclose(preds_after, preds_before, rtol=1e-6, atol=1e-6)
    assert score_after == pytest.approx(score_before, rel=1e-6, abs=1e-6)

    assert restored.context_builder == "cosine"
    assert restored.context_builder_params == model.context_builder_params
    assert restored._context_dim_ == model._context_dim_


def test_save_load_roundtrip_preserves_hisso_metadata(tmp_path):
    seed_all(41)
    X, y = _make_dataset(seed=43)
    X_tr, y_tr = X[:64], y[:64]
    X_va, y_va = X[64:88], y[64:88]
    X_seq = X[88:120]

    model = WaveResNetRegressor(
        hidden_layers=2,
        hidden_units=48,
        epochs=12,
        batch_size=16,
        lr=3e-3,
        early_stopping=True,
        patience=3,
        context_builder="cosine",
        context_builder_params={"frequencies": 3, "include_sin": True, "include_cos": True},
        random_state=23,
    )

    model.fit(
        X_tr,
        y_tr,
        validation_data=(X_va, y_va),
        verbose=0,
        hisso=True,
        hisso_window=4,
        hisso_reward_fn=multiplicative_return_reward,
        hisso_primary_transform="tanh",
        hisso_transition_penalty=0.05,
        hisso_context_extractor=_mean_context_extractor,
        hisso_supervised={"y": y_tr.reshape(-1, 1), "epochs": 1, "batch_size": 8},
    )

    cfg_before = getattr(model, "_hisso_cfg_", None)
    options_before = getattr(model, "_hisso_options_", None)
    assert cfg_before is not None
    assert options_before is not None
    assert getattr(model, "_hisso_trained_", False) is True

    hisso_preds_before = hisso_infer_series(model, X_seq)
    reward_fn_before = getattr(model, "_hisso_reward_fn_", None)
    assert reward_fn_before is not None

    checkpoint_path = tmp_path / "psann_hisso_checkpoint.pt"
    model.save(str(checkpoint_path))

    restored = WaveResNetRegressor.load(str(checkpoint_path))

    cfg_after = getattr(restored, "_hisso_cfg_", None)
    options_after = getattr(restored, "_hisso_options_", None)
    assert cfg_after is not None
    assert options_after is not None
    assert getattr(restored, "_hisso_trained_", False) is True

    assert cfg_after.episode_length == cfg_before.episode_length
    assert cfg_after.primary_dim == cfg_before.primary_dim
    assert cfg_after.primary_transform == cfg_before.primary_transform
    assert cfg_after.episodes_per_batch == cfg_before.episodes_per_batch

    assert options_after.primary_transform == options_before.primary_transform
    assert options_after.transition_penalty == pytest.approx(options_before.transition_penalty)
    if isinstance(options_after.supervised, Mapping):
        assert isinstance(options_before.supervised, Mapping)
        for key, value in options_before.supervised.items():
            if key == "y":
                np.testing.assert_allclose(
                    np.asarray(options_after.supervised[key], dtype=np.float32),
                    np.asarray(value, dtype=np.float32),
                    rtol=1e-6,
                    atol=1e-6,
                )
            else:
                assert options_after.supervised[key] == value
    else:
        assert options_after.supervised == options_before.supervised
    assert options_after.context_extractor is not None
    assert options_after.context_extractor is options_before.context_extractor

    hisso_preds_after = hisso_infer_series(restored, X_seq)
    reward_fn_after = getattr(restored, "_hisso_reward_fn_", None)
    assert reward_fn_after is not None

    np.testing.assert_allclose(hisso_preds_after, hisso_preds_before, rtol=1e-6, atol=1e-6)

    with torch.no_grad():
        dummy_actions = torch.zeros(2, 3, 1)
        dummy_context = torch.zeros(2, 3, 1)
        before_val = reward_fn_before(dummy_actions, dummy_context)
        after_val = reward_fn_after(dummy_actions, dummy_context)
    if isinstance(before_val, torch.Tensor):
        before_val = before_val.detach().cpu()
    if isinstance(after_val, torch.Tensor):
        after_val = after_val.detach().cpu()
    np.testing.assert_allclose(
        np.asarray(after_val, dtype=np.float32),
        np.asarray(before_val, dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def _mean_context_extractor(inputs: torch.Tensor) -> torch.Tensor:
    return inputs.mean(dim=-1)
