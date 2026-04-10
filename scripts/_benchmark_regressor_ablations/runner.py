# ruff: noqa: F403,F405
from __future__ import annotations

from .metrics import (
    _classification_metrics,
    _count_params,
    _history_stats,
    _regression_metrics,
    _split_train_val,
)
from .shared import *


def _run_single(
    model: ModelSpec,
    dataset: DatasetBundle,
    *,
    seed: int,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    val_fraction: float,
    scale_y: bool,
    save_model_path: Optional[Path] = None,
    save_preds_path: Optional[Path] = None,
) -> Dict[str, Any]:
    seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    params = dict(model.params)
    params.update(
        {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "device": device,
            "random_state": int(seed),
            "target_scaler": "standard" if scale_y else None,
        }
    )
    estimator = model.estimator(**params)
    start = time.perf_counter()
    X_train, y_train, X_val, y_val, y_train_labels, y_val_labels = _split_train_val(
        dataset, val_fraction=float(val_fraction), seed=int(seed)
    )
    validation_data = (X_val, y_val) if X_val.size else None
    estimator.fit(X_train, y_train, verbose=0, validation_data=validation_data)
    if save_model_path is not None:
        save_model_path.parent.mkdir(parents=True, exist_ok=True)
        estimator.save(str(save_model_path))
    elapsed = time.perf_counter() - start

    pred_train = estimator.predict(X_train)
    pred_val = estimator.predict(X_val) if X_val.size else None
    pred_test = estimator.predict(dataset.X_test)
    result: Dict[str, Any] = {
        "train_time_s": float(elapsed),
        "n_params": _count_params(estimator),
        "train_size": int(X_train.shape[0]),
        "val_size": int(X_val.shape[0]),
        "test_size": int(dataset.X_test.shape[0]),
        "scale_y": bool(scale_y),
    }
    history = getattr(estimator, "history_", []) or []
    result.update(_history_stats(history))
    if dataset.task == "classification":
        if y_train_labels is None or dataset.y_test_labels is None:
            raise ValueError("classification dataset missing label arrays")
        num_classes = int(dataset.meta.get("classes", pred_train.shape[-1]))
        metrics_train = _classification_metrics(y_train_labels, pred_train, num_classes=num_classes)
        metrics_train.update(_regression_metrics(y_train, pred_train))
        result["metrics_train"] = metrics_train
        if pred_val is not None and y_val_labels is not None:
            metrics_val = _classification_metrics(y_val_labels, pred_val, num_classes=num_classes)
            metrics_val.update(_regression_metrics(y_val, pred_val))
            result["metrics_val"] = metrics_val
        metrics_test = _classification_metrics(
            dataset.y_test_labels, pred_test, num_classes=num_classes
        )
        metrics_test.update(_regression_metrics(dataset.y_test, pred_test))
        result["metrics_test"] = metrics_test
    else:
        result["metrics_train"] = _regression_metrics(y_train, pred_train)
        if pred_val is not None and y_val.size:
            result["metrics_val"] = _regression_metrics(y_val, pred_val)
        result["metrics_test"] = _regression_metrics(dataset.y_test, pred_test)
    if save_preds_path is not None:
        save_preds_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "y_train": y_train,
            "y_val": y_val,
            "y_test": dataset.y_test,
            "pred_train": pred_train,
            "pred_val": pred_val,
            "pred_test": pred_test,
        }
        if y_train_labels is not None:
            payload["y_train_labels"] = y_train_labels
        if y_val_labels is not None:
            payload["y_val_labels"] = y_val_labels
        if dataset.y_test_labels is not None:
            payload["y_test_labels"] = dataset.y_test_labels
        np.savez_compressed(save_preds_path, **payload)
    return result
