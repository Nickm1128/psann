"""Run the full PSANN experiment suite and optionally commit results.

Example:
  python scripts/run_full_suite.py --device cuda --git-commit
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MIXED_ACTIVATION_CONFIG: Dict[str, Any] = {
    "activation_types": ["psann", "relu"],
    "activation_ratios": [0.25, 0.75],
    "ratio_sum_tol": 1e-3,
    "mix_layout": "random",
}


def _parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_csv_ints(value: str) -> List[int]:
    return [int(item) for item in _parse_csv(value)]


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _run_cmd(
    cmd: List[str],
    *,
    cwd: Path,
    log_path: Path,
    dry_run: bool,
) -> None:
    rendered = " ".join(shlex.quote(part) for part in cmd)
    if dry_run:
        print(f"[dry-run] {rendered}")
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"$ {rendered}\n")
        handle.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            handle.write(line)
        rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"Command failed with exit code {rc}: {rendered}")


def _git_info(repo_root: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {"available": False}
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            text=True,
        ).strip()
    except Exception:
        return info
    info.update(
        {
            "available": True,
            "head": head,
            "dirty": bool(status),
            "status": status.splitlines() if status else [],
        }
    )
    return info


def _git_commit(repo_root: Path, paths: Iterable[Path], message: str, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] git add {' '.join(str(p) for p in paths)}")
        print(f"[dry-run] git commit -m {message!r}")
        return
    rel_paths = [str(p.relative_to(repo_root)) for p in paths]
    subprocess.check_call(["git", "add", *rel_paths], cwd=str(repo_root))
    subprocess.check_call(["git", "commit", "-m", message], cwd=str(repo_root))


def _dataset_paths(repo_root: Path) -> Dict[str, Path]:
    return {
        "jena": repo_root / "datasets" / "Jena Climate 2009-2016" / "jena_climate_2009_2016.csv",
        "beijing": repo_root
        / "datasets"
        / "Beijing Air Quality"
        / "PRSA_Data_Guanyuan_20130301-20170228.csv",
        "eaf": repo_root / "datasets" / "Industrial Data from the Electric Arc Furnace" / "eaf_temp.csv",
    }


def _check_light_probe_data(
    tasks: List[str],
    *,
    repo_root: Path,
    allow_missing: bool,
) -> Tuple[List[str], List[str]]:
    paths = _dataset_paths(repo_root)
    missing: List[str] = []
    filtered: List[str] = []
    for task in tasks:
        key = task.strip().lower()
        if key not in paths:
            filtered.append(task)
            continue
        if paths[key].exists():
            filtered.append(task)
        else:
            missing.append(key)
    if missing and not allow_missing:
        missing_paths = [str(paths[k]) for k in missing]
        raise SystemExit(
            "Missing dataset files for light probes:\n"
            + "\n".join(f"- {p}" for p in missing_paths)
            + "\nUse --allow-missing-data to skip missing tasks."
        )
    return filtered, missing


def _collect_env_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_capability"] = torch.cuda.get_device_capability(0)
    except Exception as exc:  # pragma: no cover - best-effort
        info["torch_error"] = str(exc)
    return info


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the full PSANN experiment suite.")
    ap.add_argument("--device", type=str, default="auto", help="cpu|cuda|auto")
    ap.add_argument(
        "--out-root",
        type=str,
        default=None,
        help="Root directory for suite outputs (default: reports/full_suite/<timestamp>).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print commands without running.")
    ap.add_argument(
        "--allow-missing-data",
        action="store_true",
        help="Skip light-probe tasks if required datasets are missing.",
    )
    ap.add_argument("--skip-light-probes", action="store_true")
    ap.add_argument("--skip-ablations", action="store_true")
    ap.add_argument("--skip-geo-bench", action="store_true")
    ap.add_argument("--skip-geo-sweep", action="store_true")
    ap.add_argument("--skip-geo-micro", action="store_true")
    ap.add_argument(
        "--git-commit",
        dest="git_commit",
        action="store_true",
        default=True,
        help="Commit suite results to git (default).",
    )
    ap.add_argument(
        "--no-git-commit",
        dest="git_commit",
        action="store_false",
        help="Do not commit suite results to git.",
    )
    ap.add_argument(
        "--git-message",
        type=str,
        default=None,
        help="Commit message (default: Add full suite results <timestamp>).",
    )
    ap.add_argument(
        "--light-probe-tasks",
        type=str,
        default="jena,beijing,eaf",
        help="Comma list of light-probe tasks (jena,beijing,eaf).",
    )
    ap.add_argument(
        "--light-probe-seeds",
        type=str,
        default="0,1,2,3,4",
        help="Comma list of seeds for light probes.",
    )
    ap.add_argument("--light-probe-epochs", type=int, default=20)
    ap.add_argument("--light-probe-pr-snapshots", action="store_true")
    ap.add_argument(
        "--light-probe-match-params",
        dest="light_probe_match_params",
        action="store_true",
        default=True,
    )
    ap.add_argument(
        "--light-probe-no-match-params",
        dest="light_probe_match_params",
        action="store_false",
    )
    ap.add_argument("--light-probe-skip-deps", dest="light_probe_skip_deps", action="store_true", default=True)
    ap.add_argument("--light-probe-no-skip-deps", dest="light_probe_skip_deps", action="store_false")
    ap.add_argument(
        "--ablations-datasets",
        type=str,
        default="tabular_sine,tabular_shifted,classification_clusters,context_rotating_moons,"
        "ts_periodic,ts_regime_switch,ts_drift,ts_shock",
    )
    ap.add_argument("--ablations-models", type=str, default=None)
    ap.add_argument("--ablations-seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--ablations-epochs", type=int, default=25)
    ap.add_argument("--ablations-batch-size", type=int, default=128)
    ap.add_argument("--ablations-lr", type=float, default=1e-3)
    ap.add_argument("--ablations-val-fraction", type=float, default=0.1)
    ap.add_argument("--ablations-no-resume", dest="ablations_resume", action="store_false")
    ap.add_argument("--geo-shape", type=str, default="12x12")
    ap.add_argument("--geo-depth", type=int, default=6)
    ap.add_argument("--geo-k", type=int, default=8)
    ap.add_argument("--geo-epochs", type=int, default=25)
    ap.add_argument("--geo-batch-size", type=int, default=128)
    ap.add_argument("--geo-lr", type=float, default=1e-3)
    ap.add_argument("--geo-train-size", type=int, default=4096)
    ap.add_argument("--geo-test-size", type=int, default=1024)
    ap.add_argument("--geo-val-fraction", type=float, default=0.1)
    ap.add_argument("--geo-match-tolerance", type=float, default=0.01)
    ap.add_argument("--geo-dense-max-width", type=int, default=4096)
    ap.add_argument(
        "--geo-activation-config",
        type=str,
        default=None,
        help="JSON string for mixed activation config (default: 25/75 psann/relu).",
    )
    ap.add_argument(
        "--geo-sweep-shapes",
        type=str,
        default="4x4,8x8",
    )
    ap.add_argument("--geo-sweep-depths", type=str, default="4,8")
    ap.add_argument("--geo-sweep-ks", type=str, default="4,8,16")
    ap.add_argument("--geo-sweep-activations", type=str, default="relu,psann,mixed")
    ap.add_argument("--geo-sweep-seeds", type=str, default="0,1,2")
    ap.add_argument("--geo-sweep-epochs", type=int, default=10)
    ap.add_argument("--geo-sweep-batch-size", type=int, default=128)
    ap.add_argument("--geo-sweep-train-size", type=int, default=4096)
    ap.add_argument("--geo-sweep-test-size", type=int, default=1024)
    ap.add_argument("--geo-sweep-val-fraction", type=float, default=0.1)
    ap.add_argument("--geo-sweep-no-resume", dest="geo_sweep_resume", action="store_false")
    ap.add_argument("--geo-micro-shape", type=str, default="12x12")
    ap.add_argument("--geo-micro-k", type=int, default=8)
    ap.add_argument("--geo-micro-steps", type=int, default=50)
    ap.add_argument("--geo-micro-warmup", type=int, default=10)
    ap.set_defaults(ablations_resume=True, geo_sweep_resume=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root) if args.out_root else REPO_ROOT / "reports" / "full_suite" / stamp
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "suite_output.txt"

    activation_cfg = (
        json.loads(args.geo_activation_config)
        if args.geo_activation_config
        else DEFAULT_MIXED_ACTIVATION_CONFIG
    )

    suite_manifest: Dict[str, Any] = {
        "timestamp": stamp,
        "out_root": str(out_root),
        "device": args.device,
        "env": _collect_env_info(),
        "git": _git_info(REPO_ROOT),
        "commands": [],
        "skipped": [],
        "missing_data": [],
    }

    if not args.skip_light_probes:
        tasks = _parse_csv(args.light_probe_tasks)
        tasks, missing = _check_light_probe_data(
            tasks, repo_root=REPO_ROOT, allow_missing=args.allow_missing_data
        )
        suite_manifest["missing_data"].extend(missing)
        if tasks:
            cmd = [
                sys.executable,
                "scripts/run_light_probes.py",
                "--epochs",
                str(args.light_probe_epochs),
                "--device",
                args.device,
                "--results-dir",
                str(out_root / "light_probes"),
                "--tasks",
                *tasks,
                "--seeds",
                *[str(s) for s in _parse_csv_ints(args.light_probe_seeds)],
            ]
            if args.light_probe_pr_snapshots:
                cmd.append("--pr-snapshots")
            if args.light_probe_match_params:
                cmd.append("--match-params")
            else:
                cmd.append("--no-match-params")
            if args.light_probe_skip_deps:
                cmd.append("--skip-deps")
            suite_manifest["commands"].append(cmd)
            _run_cmd(cmd, cwd=REPO_ROOT, log_path=log_path, dry_run=args.dry_run)
        else:
            suite_manifest["skipped"].append("light_probes")
    else:
        suite_manifest["skipped"].append("light_probes")

    if not args.skip_ablations:
        datasets = args.ablations_datasets
        cmd = [
            sys.executable,
            "scripts/benchmark_regressor_ablations.py",
            "--datasets",
            datasets,
            "--seeds",
            args.ablations_seeds,
            "--device",
            args.device,
            "--epochs",
            str(args.ablations_epochs),
            "--batch-size",
            str(args.ablations_batch_size),
            "--lr",
            str(args.ablations_lr),
            "--val-fraction",
            str(args.ablations_val_fraction),
            "--out",
            str(out_root / "ablations"),
        ]
        if args.ablations_models:
            cmd.extend(["--models", args.ablations_models])
        if args.ablations_resume:
            cmd.append("--resume")
        suite_manifest["commands"].append(cmd)
        _run_cmd(cmd, cwd=REPO_ROOT, log_path=log_path, dry_run=args.dry_run)
    else:
        suite_manifest["skipped"].append("ablations")

    if not args.skip_geo_bench:
        cmd = [
            sys.executable,
            "scripts/benchmark_geo_sparse_vs_dense.py",
            "--task",
            "mixed",
            "--sparse-activation",
            "mixed",
            "--activation-config",
            json.dumps(activation_cfg),
            "--shape",
            args.geo_shape,
            "--depth",
            str(args.geo_depth),
            "--k",
            str(args.geo_k),
            "--device",
            args.device,
            "--epochs",
            str(args.geo_epochs),
            "--batch-size",
            str(args.geo_batch_size),
            "--lr",
            str(args.geo_lr),
            "--train-size",
            str(args.geo_train_size),
            "--test-size",
            str(args.geo_test_size),
            "--val-fraction",
            str(args.geo_val_fraction),
            "--match-tolerance",
            str(args.geo_match_tolerance),
            "--dense-max-width",
            str(args.geo_dense_max_width),
            "--out",
            str(out_root / "geo_sparse"),
        ]
        suite_manifest["commands"].append(cmd)
        _run_cmd(cmd, cwd=REPO_ROOT, log_path=log_path, dry_run=args.dry_run)
    else:
        suite_manifest["skipped"].append("geo_bench")

    if not args.skip_geo_sweep:
        cmd = [
            sys.executable,
            "scripts/geo_sparse_sweep.py",
            "--task",
            "mixed",
            "--shapes",
            args.geo_sweep_shapes,
            "--depths",
            args.geo_sweep_depths,
            "--ks",
            args.geo_sweep_ks,
            "--activations",
            args.geo_sweep_activations,
            "--seeds",
            args.geo_sweep_seeds,
            "--device",
            args.device,
            "--epochs",
            str(args.geo_sweep_epochs),
            "--batch-size",
            str(args.geo_sweep_batch_size),
            "--train-size",
            str(args.geo_sweep_train_size),
            "--test-size",
            str(args.geo_sweep_test_size),
            "--val-fraction",
            str(args.geo_sweep_val_fraction),
            "--activation-config",
            json.dumps(activation_cfg),
            "--out",
            str(out_root / "geo_sparse_sweep"),
        ]
        if args.geo_sweep_resume:
            cmd.append("--resume")
        suite_manifest["commands"].append(cmd)
        _run_cmd(cmd, cwd=REPO_ROOT, log_path=log_path, dry_run=args.dry_run)
    else:
        suite_manifest["skipped"].append("geo_sweep")

    if not args.skip_geo_micro:
        cmd = [
            sys.executable,
            "scripts/benchmark_geo_sparse_micro.py",
            "--shape",
            args.geo_micro_shape,
            "--k",
            str(args.geo_micro_k),
            "--steps",
            str(args.geo_micro_steps),
            "--warmup",
            str(args.geo_micro_warmup),
            "--device",
            args.device,
            "--out",
            str(out_root / "geo_sparse_micro"),
        ]
        suite_manifest["commands"].append(cmd)
        _run_cmd(cmd, cwd=REPO_ROOT, log_path=log_path, dry_run=args.dry_run)
    else:
        suite_manifest["skipped"].append("geo_micro")

    _write_json(out_root / "suite_manifest.json", suite_manifest)

    if args.git_commit:
        message = args.git_message or f"Add full suite results {stamp}"
        _git_commit(REPO_ROOT, [out_root], message, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
