# ruff: noqa: F403,F405
from __future__ import annotations

from .shared import *


def _default_output_dir() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return Path("reports") / "ablations" / f"{stamp}_regressor_ablations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        type=str,
        default="tabular_sine,tabular_shifted,classification_clusters,context_rotating_moons,ts_periodic,ts_regime_switch,ts_drift,ts_shock",
        help="Comma-separated dataset names to run.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="res_base,res_relu_sigmoid_psann,res_drop_path,res_no_norm,wrn_base,wrn_no_phase,wrn_no_film,wrn_spec_gate_rfft,wrn_spec_gate_feats,sgr_base,sgr_no_gate,sgr_fourier_feats,sgr_no_phase",
        help="Comma-separated model keys to run.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1",
        help="Comma-separated seeds for dataset/model runs.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to use: cpu|cuda|auto.")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs per run.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Fraction of training data held out for validation (0 disables).",
    )
    parser.add_argument(
        "--scale-y",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Standardize target values using train split (metrics stay in original scale).",
    )
    parser.add_argument("--out", type=str, default=None, help="Output directory.")
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Persist fitted estimators under outputs/models/.",
    )
    parser.add_argument(
        "--save-preds",
        action="store_true",
        help="Persist predictions under outputs/preds/.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs already present in results.jsonl.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available datasets and exit.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available model keys and exit.",
    )
    return parser.parse_args()


def _load_existing_run_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    run_ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        run_id = payload.get("run_id")
        if isinstance(run_id, str):
            run_ids.add(run_id)
    return run_ids


def _write_summary(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _aggregate_seed_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        dataset = str(row.get("dataset"))
        model = str(row.get("model"))
        groups.setdefault((dataset, model), []).append(row)

    def _mean_std(values: List[float]) -> Tuple[float, float]:
        if not values:
            return float("nan"), float("nan")
        mean = float(sum(values) / len(values))
        var = float(sum((v - mean) ** 2 for v in values) / len(values))
        return mean, math.sqrt(var)

    allowed_prefixes = (
        "metrics_",
        "train_time_s",
        "train_time_s_total",
        "n_params",
        "loss_curve_volatility",
        "loss_curve_mean_abs_diff",
        "val_curve_volatility",
        "grad_norm_max",
        "epoch_time_s_mean",
        "step_time_s_mean",
    )
    summary_rows: List[Dict[str, Any]] = []
    for (dataset, model), entries in sorted(groups.items()):
        row: Dict[str, Any] = {"dataset": dataset, "model": model, "n_runs": len(entries)}
        keys = set()
        for entry in entries:
            keys.update(entry.keys())
        for key in sorted(keys):
            if key in {"dataset", "model", "seed", "status"}:
                continue
            if not key.startswith(allowed_prefixes):
                continue
            vals = []
            for entry in entries:
                val = entry.get(key)
                if isinstance(val, (int, float)):
                    vals.append(float(val))
            if not vals:
                continue
            mean, std = _mean_std(vals)
            row[f"{key}_mean"] = mean
            row[f"{key}_std"] = std
        summary_rows.append(row)
    return summary_rows


def _flatten_result(entry: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in entry.items():
        if isinstance(value, dict):
            for sub_k, sub_v in value.items():
                flat[f"{key}_{sub_k}"] = sub_v
        else:
            flat[key] = value
    return flat


def _load_results(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    entries: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            entries.append(payload)
    return entries
