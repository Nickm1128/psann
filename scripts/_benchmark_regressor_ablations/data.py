# ruff: noqa: F403,F405
from __future__ import annotations

from .shared import *


@dataclass
class DatasetBundle:
    name: str
    task: str
    kind: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    meta: Dict[str, Any]
    y_train_labels: Optional[np.ndarray] = None
    X_val: Optional[np.ndarray] = None
    y_val: Optional[np.ndarray] = None
    y_val_labels: Optional[np.ndarray] = None
    y_test_labels: Optional[np.ndarray] = None


@dataclass
class DatasetSpec:
    name: str
    task: str
    kind: str
    builder: Callable[[int], DatasetBundle]


def _make_tabular_sine(seed: int) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    n_train, n_test, features = 1024, 256, 8
    X = rng.uniform(-2.0, 2.0, size=(n_train + n_test, features)).astype(np.float32)
    noise = 0.05 * rng.standard_normal(size=(n_train + n_test,))
    y = (
        np.sin(3.0 * X[:, 0])
        + 0.5 * np.cos(2.0 * X[:, 1])
        + 0.2 * X[:, 2] * X[:, 3]
        - 0.1 * (X[:, 4] ** 2)
        + noise
    )
    y = y.astype(np.float32).reshape(-1, 1)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    return DatasetBundle(
        name="tabular_sine",
        task="regression",
        kind="tabular",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        meta={"features": features, "train_size": n_train, "test_size": n_test},
    )


def _make_tabular_shifted(seed: int) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    n_train, n_test, features = 1024, 256, 8
    X_train = rng.normal(loc=0.0, scale=1.0, size=(n_train, features)).astype(np.float32)
    X_test = rng.normal(loc=0.6, scale=1.3, size=(n_test, features)).astype(np.float32)

    def _targets(X: np.ndarray) -> np.ndarray:
        noise = 0.2 * rng.standard_t(df=3, size=(X.shape[0],))
        heavy = rng.random(X.shape[0]) < 0.03
        noise = noise + heavy * rng.normal(scale=2.0, size=(X.shape[0],))
        base = np.where(
            X[:, 0] > 0,
            1.5 * X[:, 0] - 0.5 * X[:, 1],
            -1.2 * X[:, 0] + 0.3 * X[:, 2],
        )
        return (base + noise).astype(np.float32).reshape(-1, 1)

    y_train = _targets(X_train)
    y_test = _targets(X_test)
    return DatasetBundle(
        name="tabular_shifted",
        task="regression",
        kind="tabular",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        meta={
            "features": features,
            "train_size": n_train,
            "test_size": n_test,
            "train_shift": 0.0,
            "test_shift": 0.6,
        },
    )


def _make_classification_clusters(seed: int) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    n_train, n_test = 900, 300
    n_classes = 3
    total = n_train + n_test
    per_class = total // n_classes
    extras = total % n_classes
    centers = np.array([[0.0, 0.0], [2.0, 2.0], [-2.0, 2.0]], dtype=np.float32)

    features = []
    labels = []
    for idx in range(n_classes):
        count = per_class + (1 if idx < extras else 0)
        base = rng.normal(scale=0.5, size=(count, 2)).astype(np.float32) + centers[idx]
        extra1 = np.sin(base[:, :1]) + 0.1 * rng.standard_normal(size=(count, 1))
        extra2 = np.cos(base[:, 1:2]) + 0.1 * rng.standard_normal(size=(count, 1))
        feats = np.concatenate([base, extra1, extra2], axis=1)
        features.append(feats)
        labels.append(np.full((count,), idx, dtype=np.int64))

    X = np.concatenate(features, axis=0).astype(np.float32)
    y_labels = np.concatenate(labels, axis=0)
    order = rng.permutation(X.shape[0])
    X = X[order]
    y_labels = y_labels[order]
    y_onehot = np.eye(n_classes, dtype=np.float32)[y_labels]

    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y_onehot[:n_train], y_onehot[n_train:]
    y_train_labels = y_labels[:n_train]
    y_test_labels = y_labels[n_train:]
    return DatasetBundle(
        name="classification_clusters",
        task="classification",
        kind="tabular",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        meta={
            "features": X.shape[1],
            "train_size": n_train,
            "test_size": n_test,
            "classes": n_classes,
        },
        y_train_labels=y_train_labels,
        y_test_labels=y_test_labels,
    )


def _make_ts_periodic(seed: int) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    total_steps = 1600
    window = 32
    t = np.arange(total_steps, dtype=np.float32)
    daily = np.sin(2.0 * math.pi * t / 24.0)
    weekly = np.sin(2.0 * math.pi * t / (24.0 * 7.0))
    mid = np.cos(2.0 * math.pi * t / 12.0)
    noise = 0.05 * rng.standard_normal(size=total_steps)
    series = (daily + 0.6 * weekly + 0.3 * mid + noise).astype(np.float32)
    feats = np.stack([daily, np.cos(2.0 * math.pi * t / 24.0), weekly], axis=1).astype(np.float32)

    X, y = _windowed_series(feats, series, window=window)
    return _split_sequence_dataset(
        name="ts_periodic",
        X=X,
        y=y,
        window=window,
        kind="sequence",
        meta={"features": feats.shape[1], "window": window},
    )


def _make_ts_regime(seed: int) -> DatasetBundle:
    series, contexts = make_regime_switch_ts(1400, regimes=3, seed=seed)
    series_np = series.numpy().astype(np.float32)
    contexts_np = contexts.numpy().astype(np.float32)
    feats = np.concatenate([series_np[:, None], contexts_np], axis=1).astype(np.float32)
    window = 32
    X, y = _windowed_series(feats, series_np, window=window)
    return _split_sequence_dataset(
        name="ts_regime_switch",
        X=X,
        y=y,
        window=window,
        kind="sequence",
        meta={"features": feats.shape[1], "window": window, "regimes": 3},
    )


def _make_ts_drift(seed: int) -> DatasetBundle:
    X_raw, y_raw = make_drift_series(1600, drift=0.001, frequency=0.02, noise=0.02, seed=seed)
    series = torch.cat([X_raw.squeeze(-1), y_raw[-1].squeeze(-1).reshape(1)], dim=0)
    series_np = series.numpy().astype(np.float32)
    feats = series_np[:-1].reshape(-1, 1)
    targets = series_np[1:]
    window = 32
    X, y = _windowed_series(feats, targets, window=window)
    return _split_sequence_dataset(
        name="ts_drift",
        X=X,
        y=y,
        window=window,
        kind="sequence",
        meta={"features": feats.shape[1], "window": window, "drift": 0.001},
    )


def _make_ts_shock(seed: int) -> DatasetBundle:
    X_raw, y_raw = make_shock_series(
        1600, shock_prob=0.05, shock_scale=2.0, noise=0.05, mean_revert=0.85, seed=seed
    )
    series = torch.cat([X_raw.squeeze(-1), y_raw[-1].squeeze(-1).reshape(1)], dim=0)
    series_np = series.numpy().astype(np.float32)
    feats = series_np[:-1].reshape(-1, 1)
    targets = series_np[1:]
    window = 32
    X, y = _windowed_series(feats, targets, window=window)
    return _split_sequence_dataset(
        name="ts_shock",
        X=X,
        y=y,
        window=window,
        kind="sequence",
        meta={"features": feats.shape[1], "window": window, "shock_prob": 0.05},
    )


def _make_context_rotating_moons(seed: int) -> DatasetBundle:
    feats, labels, contexts = make_context_rotating_moons(1200, noise=0.05, seed=seed)
    X = torch.cat([feats, contexts], dim=1).numpy().astype(np.float32)
    y_labels = labels.numpy().astype(np.int64)
    y_onehot = np.eye(2, dtype=np.float32)[y_labels]
    order = np.random.default_rng(seed).permutation(X.shape[0])
    X = X[order]
    y_onehot = y_onehot[order]
    y_labels = y_labels[order]
    n_train = 900
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y_onehot[:n_train], y_onehot[n_train:]
    y_train_labels = y_labels[:n_train]
    y_test_labels = y_labels[n_train:]
    return DatasetBundle(
        name="context_rotating_moons",
        task="classification",
        kind="tabular",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        meta={
            "features": X.shape[1],
            "train_size": n_train,
            "test_size": X_test.shape[0],
            "classes": 2,
            "context_dim": 1,
        },
        y_train_labels=y_train_labels,
        y_test_labels=y_test_labels,
    )


def _windowed_series(
    feats: np.ndarray,
    target: np.ndarray,
    *,
    window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if feats.ndim != 2:
        raise ValueError("feats must be 2D (T, F)")
    if target.ndim != 1:
        raise ValueError("target must be 1D (T,)")
    if feats.shape[0] != target.shape[0]:
        raise ValueError("feats and target must share the time dimension")
    if window <= 0 or window >= target.shape[0]:
        raise ValueError("window must be positive and shorter than series length")

    windows = []
    targets = []
    for idx in range(window, target.shape[0]):
        windows.append(feats[idx - window : idx])
        targets.append(target[idx])
    X = np.stack(windows, axis=0).astype(np.float32)
    y = np.asarray(targets, dtype=np.float32).reshape(-1, 1)
    return X, y


def _split_sequence_dataset(
    *,
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    window: int,
    kind: str,
    meta: Dict[str, Any],
    train_ratio: float = 0.8,
) -> DatasetBundle:
    split = int(X.shape[0] * train_ratio)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    meta = dict(meta)
    meta.update({"train_size": X_train.shape[0], "test_size": X_test.shape[0], "window": window})
    return DatasetBundle(
        name=name,
        task="regression",
        kind=kind,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        meta=meta,
    )


DATASETS: Dict[str, DatasetSpec] = {
    "tabular_sine": DatasetSpec(
        name="tabular_sine", task="regression", kind="tabular", builder=_make_tabular_sine
    ),
    "tabular_shifted": DatasetSpec(
        name="tabular_shifted",
        task="regression",
        kind="tabular",
        builder=_make_tabular_shifted,
    ),
    "classification_clusters": DatasetSpec(
        name="classification_clusters",
        task="classification",
        kind="tabular",
        builder=_make_classification_clusters,
    ),
    "ts_periodic": DatasetSpec(
        name="ts_periodic", task="regression", kind="sequence", builder=_make_ts_periodic
    ),
    "ts_regime_switch": DatasetSpec(
        name="ts_regime_switch",
        task="regression",
        kind="sequence",
        builder=_make_ts_regime,
    ),
    "ts_drift": DatasetSpec(
        name="ts_drift", task="regression", kind="sequence", builder=_make_ts_drift
    ),
    "ts_shock": DatasetSpec(
        name="ts_shock", task="regression", kind="sequence", builder=_make_ts_shock
    ),
    "context_rotating_moons": DatasetSpec(
        name="context_rotating_moons",
        task="classification",
        kind="tabular",
        builder=_make_context_rotating_moons,
    ),
}
