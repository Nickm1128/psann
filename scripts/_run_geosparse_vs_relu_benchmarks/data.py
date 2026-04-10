# ruff: noqa: F403,F405
from __future__ import annotations

from .shared import *

DATASET_NAMES = [
    "syn_sparse_linear",
    "syn_friedman1",
    "syn_piecewise_sine",
    "real_california_housing",
    "real_diabetes",
    "real_linnerud",
]
DEFAULT_GEO_ACTIVATIONS = "psann,relu_sigmoid_psann"


@dataclass
class DatasetSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    x_scaler: StandardScaler
    y_scaler: Optional[StandardScaler]


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_csv(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def resolve_geo_shape(input_dim: int, shape: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
    if shape is not None:
        return int(shape[0]), int(shape[1])
    return 1, int(input_dim)


def _maybe_cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _compute_epochs(n_train: int, batch_size: int, target_steps: int) -> Tuple[int, int]:
    steps_per_epoch = int(math.ceil(n_train / batch_size))
    epochs = int(math.ceil(target_steps / steps_per_epoch))
    return max(1, epochs), steps_per_epoch


def _subsample(
    X: np.ndarray, y: np.ndarray, n_samples: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    if n_samples is None or len(X) <= n_samples:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=int(n_samples), replace=False)
    return X[idx], y[idx]


def split_and_scale(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    test_size: float,
    val_size: float,
    scale_y: bool,
) -> DatasetSplit:
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=test_size + val_size, random_state=seed
    )
    val_frac = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=1.0 - val_frac, random_state=seed
    )

    x_scaler = StandardScaler().fit(X_train)
    X_train_s = x_scaler.transform(X_train).astype(np.float32, copy=False)
    X_val_s = x_scaler.transform(X_val).astype(np.float32, copy=False)
    X_test_s = x_scaler.transform(X_test).astype(np.float32, copy=False)

    y_scaler = None
    if scale_y:
        y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
        y_train = y_scaler.transform(y_train.reshape(-1, 1)).ravel()
        y_val = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
        y_test = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    y_train = y_train.astype(np.float32, copy=False)
    y_val = y_val.astype(np.float32, copy=False)
    y_test = y_test.astype(np.float32, copy=False)

    return DatasetSplit(
        X_train=X_train_s,
        y_train=y_train,
        X_val=X_val_s,
        y_val=y_val,
        X_test=X_test_s,
        y_test=y_test,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
    )


def load_dataset(name: str, *, seed: int) -> Tuple[np.ndarray, np.ndarray, str]:
    if name == "syn_sparse_linear":
        X, y = make_regression(
            n_samples=5000,
            n_features=200,
            n_informative=10,
            noise=5.0,
            bias=10.0,
            random_state=seed,
        )
    elif name == "syn_friedman1":
        X, y = make_friedman1(
            n_samples=5000,
            n_features=20,
            noise=0.5,
            random_state=seed,
        )
    elif name == "syn_piecewise_sine":
        rng = np.random.default_rng(seed)
        n_samples = 5000
        n_features = 10
        X = rng.normal(size=(n_samples, n_features))
        t = X[:, 0]
        y = np.where(
            t < 0,
            0.5 * np.sin(3.0 * t) + 0.2 * t,
            1.0 * np.sin(6.0 * t + 0.5) - 0.1 * t + 0.5,
        )
        y = y + 0.1 * rng.normal(size=n_samples)
    elif name == "real_california_housing":
        data = fetch_california_housing()
        X, y = data.data, data.target
    elif name == "real_diabetes":
        data = load_diabetes()
        X, y = data.data, data.target
    elif name == "real_linnerud":
        data = load_linnerud()
        X = data.data
        y = data.target[:, 0]
    else:
        raise ValueError(f"Unknown dataset: {name}")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    return X, y, "regression"
