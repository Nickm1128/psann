# ruff: noqa: F403,F405
from __future__ import annotations

from .shared import *


@dataclass
class DatasetBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    meta: Dict[str, Any]


@dataclass(frozen=True)
class StandardScaler:
    mean: np.ndarray
    scale: np.ndarray
    eps: float

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.scale).astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return (x * self.scale + self.mean).astype(np.float32)


def _fit_standard_scaler(x: np.ndarray, *, eps: float = 1e-6) -> StandardScaler:
    mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    scale = x.std(axis=0, keepdims=True).astype(np.float32)
    scale = np.where(scale < float(eps), 1.0, scale).astype(np.float32)
    return StandardScaler(mean=mean, scale=scale, eps=float(eps))


def _parse_shape(text: str) -> Tuple[int, int]:
    for sep in ("x", "X", ","):
        if sep in text:
            parts = text.split(sep)
            if len(parts) == 2:
                return int(parts[0]), int(parts[1])
    raise ValueError("shape must be formatted as HxW or H,W")


def _print_header(args: argparse.Namespace, out_dir: Path, device: torch.device) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(
        "[bench] start",
        f"time={ts}",
        f"device={device}",
        f"seed={args.seed}",
        f"shape={args.shape}",
        f"depth={args.depth}",
        f"k={args.k}",
        f"task={args.task}",
        f"out={out_dir}",
        flush=True,
    )
    print(
        f"[bench] torch={torch.__version__} numpy={np.__version__} tf32={args.tf32} amp={args.amp}",
        flush=True,
    )


def _build_tabular_sine(
    *,
    seed: int,
    n_train: int,
    n_test: int,
    features: int,
) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    total = n_train + n_test
    X = rng.uniform(-2.0, 2.0, size=(total, features)).astype(np.float32)
    noise = 0.05 * rng.standard_normal(size=(total,)).astype(np.float32)
    y = np.zeros((total,), dtype=np.float32)
    if features >= 1:
        y += np.sin(3.0 * X[:, 0])
    if features >= 2:
        y += 0.5 * np.cos(2.0 * X[:, 1])
    if features >= 4:
        y += 0.2 * X[:, 2] * X[:, 3]
    if features >= 5:
        y += -0.1 * (X[:, 4] ** 2)
    if features >= 6:
        y += 0.05 * np.sin(X[:, 5] * X[:, 0])
    y = (y + noise).reshape(-1, 1)

    X_train = X[:n_train]
    X_test = X[n_train:]
    y_train = y[:n_train]
    y_test = y[n_train:]
    return DatasetBundle(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        meta={
            "task": "sine",
            "features": features,
            "train_size": n_train,
            "test_size": n_test,
        },
    )


def _build_tabular_mixed(
    *,
    seed: int,
    n_train: int,
    n_test: int,
    features: int,
) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    total = n_train + n_test
    X = rng.uniform(-2.0, 2.0, size=(total, features)).astype(np.float32)
    noise = 0.05 * rng.standard_normal(size=(total,)).astype(np.float32)
    y = np.zeros((total,), dtype=np.float32)
    if features >= 1:
        y += 0.35 * np.sin(2.0 * X[:, 0])
    if features >= 2:
        y += 0.25 * np.cos(1.5 * X[:, 1])
    if features >= 4:
        y += 0.25 * X[:, 2] * X[:, 3]
    if features >= 5:
        y += -0.10 * (X[:, 4] ** 2)
    if features >= 6:
        y += 0.20 * np.tanh(X[:, 5])
    if features >= 7:
        y += 0.15 * np.maximum(0.0, X[:, 6])
    if features >= 8:
        y += 0.10 * np.where(X[:, 7] > 0.0, X[:, 7] ** 2, -0.5 * X[:, 7])
    if features >= 10:
        y += 0.05 * (X[:, 8] ** 3) - 0.03 * X[:, 9]
    y = (y + noise).reshape(-1, 1)

    X_train = X[:n_train]
    X_test = X[n_train:]
    y_train = y[:n_train]
    y_test = y[n_train:]
    return DatasetBundle(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        meta={
            "task": "mixed",
            "features": features,
            "train_size": n_train,
            "test_size": n_test,
        },
    )


def _build_tabular_teacher_mlp(
    *,
    seed: int,
    n_train: int,
    n_test: int,
    features: int,
    activation: str,
) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    total = n_train + n_test
    X = rng.normal(0.0, 1.0, size=(total, features)).astype(np.float32)

    hidden1 = max(32, min(256, int(features * 4)))
    hidden2 = max(16, min(256, int(features * 2)))

    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    act_key = activation.lower()
    if act_key == "relu":
        act_fn = _relu
    elif act_key == "tanh":
        act_fn = np.tanh
    else:
        raise ValueError("activation must be one of: 'relu', 'tanh'")

    def _scaled(shape: tuple[int, ...], fan_in: int) -> np.ndarray:
        return rng.normal(0.0, 1.0 / np.sqrt(max(1, fan_in)), size=shape).astype(np.float32)

    W1 = _scaled((features, hidden1), fan_in=features)
    b1 = _scaled((hidden1,), fan_in=features)
    W2 = _scaled((hidden1, hidden2), fan_in=hidden1)
    b2 = _scaled((hidden2,), fan_in=hidden1)
    W3 = _scaled((hidden2, 1), fan_in=hidden2)
    b3 = _scaled((1,), fan_in=hidden2)

    h1 = act_fn(X @ W1 + b1)
    h2 = act_fn(h1 @ W2 + b2)
    y = (h2 @ W3 + b3).reshape(-1).astype(np.float32)

    y_std = float(np.std(y))
    if y_std > 1e-6:
        y = y / y_std
    y = y + 0.05 * rng.standard_normal(size=(total,)).astype(np.float32)
    y = y.reshape(-1, 1)

    X_train = X[:n_train]
    X_test = X[n_train:]
    y_train = y[:n_train]
    y_test = y[n_train:]
    return DatasetBundle(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        meta={
            "task": f"teacher_mlp_{act_key}",
            "teacher_hidden1": hidden1,
            "teacher_hidden2": hidden2,
            "features": features,
            "train_size": n_train,
            "test_size": n_test,
        },
    )


def _build_dataset(
    *,
    task: str,
    seed: int,
    n_train: int,
    n_test: int,
    features: int,
) -> DatasetBundle:
    key = str(task).lower().strip()
    if key in {"sine", "tabular_sine"}:
        return _build_tabular_sine(seed=seed, n_train=n_train, n_test=n_test, features=features)
    if key in {"mixed", "tabular_mixed"}:
        return _build_tabular_mixed(seed=seed, n_train=n_train, n_test=n_test, features=features)
    if key in {"teacher_relu", "teacher_mlp_relu"}:
        return _build_tabular_teacher_mlp(
            seed=seed, n_train=n_train, n_test=n_test, features=features, activation="relu"
        )
    if key in {"teacher_tanh", "teacher_mlp_tanh"}:
        return _build_tabular_teacher_mlp(
            seed=seed, n_train=n_train, n_test=n_test, features=features, activation="tanh"
        )
    raise ValueError("task must be one of: sine, mixed, teacher_relu, teacher_tanh")


def _train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    val_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if val_fraction <= 0.0:
        return X, y, X[:0], y[:0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])
    n_val = max(1, int(round(X.shape[0] * val_fraction)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]
