from __future__ import annotations

import json
from pathlib import Path

import pytest

from psann.scripts import hisso_log_run


def test_hisso_logging_cli_remains_included_in_the_wheel_package_definition():
    """The public module CLI must survive the clean-wheel boundary."""

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    wheel_config = pyproject.read_text(encoding="utf-8").split(
        "[tool.hatch.build.targets.sdist]", 1
    )[0]
    assert 'packages = ["src/psann"]' in wheel_config
    assert '"scripts",' not in wheel_config


def _write_config(path: Path) -> None:
    config = {
        "estimator": {
            "target": "psann.PSANNRegressor",
            "params": {
                "hidden_layers": 1,
                "hidden_units": 12,
                "epochs": 1,
                "batch_size": 24,
                "lr": 1e-3,
                "random_state": 11,
            },
        },
        "episodic": {
            "enabled": True,
            "strategy": {
                "kind": "hisso",
                "schedule": {
                    "episode_length": 12,
                    "batch_episodes": 3,
                    "updates_per_epoch": 2,
                },
                "primary_transform": "softmax",
                "transition_penalty": 0.05,
                "mixed_precision": False,
            },
        },
        "data": {
            "loader": "psann.scripts.hisso_log_run.toy_hisso_dataset",
            "kwargs": {
                "steps": 96,
                "features": 3,
                "seed": 7,
                "train_fraction": 0.6,
                "val_fraction": 0.2,
            },
        },
        "training": {
            "verbose": 0,
        },
        "evaluation": {
            "portfolio_prices_key": "prices_test",
            "trans_cost": 0.0,
        },
    }
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def test_hisso_logging_cli_emits_metrics(tmp_path):
    config_path = tmp_path / "config.json"
    _write_config(config_path)

    output_dir = tmp_path / "artifacts"
    exit_code = hisso_log_run.main(
        [
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--run-name",
            "smoke",
            "--device",
            "cpu",
            "--seed",
            "123",
            "--keep-checkpoints",
        ]
    )

    assert exit_code == 0

    run_dir = output_dir / "smoke"
    metrics_path = run_dir / "metrics.json"
    resolved_path = run_dir / "config_resolved.yaml"
    events_path = run_dir / "events.csv"
    checkpoint_dir = run_dir / "checkpoints"

    assert metrics_path.exists()
    assert resolved_path.exists()
    assert events_path.exists()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["status"] == "success"
    assert "duration_seconds" in metrics
    assert metrics["train_loss"] is not None
    hisso_metrics = metrics["hisso"]
    assert hisso_metrics is not None
    assert hisso_metrics["best_epoch"] in (1, None)
    assert hisso_metrics["throughput_eps_per_sec"] is not None
    profile = hisso_metrics.get("profile", {})
    assert profile.get("episode_batch_size") == 3
    assert profile.get("updates_per_epoch") == 2
    assert "portfolio_metrics" in metrics

    history_len = metrics.get("history_length", 0)
    assert history_len >= 1

    events_text = events_path.read_text(encoding="utf-8")
    assert "dataset.shapes" in events_text
    assert "run.completed" in events_text

    best_ckpt = checkpoint_dir / "best.pt"
    latest_ckpt = checkpoint_dir / "latest.pt"
    assert best_ckpt.exists()
    assert latest_ckpt.exists()

    resolved_yaml = resolved_path.read_text(encoding="utf-8")
    assert "episodic:" in resolved_yaml
    assert "strategy:" in resolved_yaml
    assert "kind: hisso" in resolved_yaml
    assert "schedule:" in resolved_yaml
    assert "output_dir" in resolved_yaml


def test_hisso_logging_cli_preserves_tagged_strategy_fields_without_flat_round_trip(tmp_path):
    config_path = tmp_path / "canonical.json"
    _write_config(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["episodic"]["strategy"] = {
        "kind": "hisso",
        "schedule": {
            "episode_length": 9,
            "batch_episodes": 2,
            "updates_per_epoch": 3,
            "random_state": 17,
        },
        "reward": "portfolio",
        "primary_transform": "tanh",
        "transition_penalty": 0.0,
        "input_noise_std": 0.125,
        "gradient_clip": 0.25,
        "mixed_precision": False,
        "amp_dtype": "bfloat16",
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    output_dir = tmp_path / "artifacts"
    assert (
        hisso_log_run.main(
            [
                "--config",
                str(config_path),
                "--output-dir",
                str(output_dir),
                "--run-name",
                "canonical",
                "--device",
                "cpu",
                "--seed",
                "123",
            ]
        )
        == 0
    )
    resolved = (output_dir / "canonical" / "config_resolved.yaml").read_text(encoding="utf-8")
    assert "random_state: 17" in resolved
    assert "gradient_clip: 0.25" in resolved
    assert "input_noise_std: 0.125" in resolved
    assert "reward: finance" in resolved
    assert "batch_episodes: 2" in resolved
    assert "updates_per_epoch: 3" in resolved


def test_hisso_logging_cli_rejects_target_selection_inside_canonical_warm_start():
    with pytest.raises(ValueError, match="episodic.strategy.warm_start.y_key"):
        hisso_log_run._normalise_canonical_cli_strategy(
            {
                "kind": "hisso",
                "warm_start": {"epochs": 1, "y_key": "y_train"},
            }
        )


def test_hisso_logging_cli_accepts_hisso_preset_and_all_maintained_templates():
    assert isinstance(
        hisso_log_run._normalise_canonical_cli_strategy("hisso"), hisso_log_run.HISSOConfig
    )
    configs_dir = Path(__file__).resolve().parents[1] / "configs" / "hisso"
    config_paths = sorted(configs_dir.glob("*.yaml"))
    assert config_paths
    for path in config_paths:
        config = hisso_log_run._load_config(path)
        assert isinstance(
            hisso_log_run._normalise_canonical_cli_strategy(config["episodic"]["strategy"]),
            hisso_log_run.HISSOConfig,
        )


def test_hisso_logging_cli_respects_output_dir(tmp_path):
    config_path = tmp_path / "config.json"
    _write_config(config_path)

    out_root = tmp_path / "outdir"
    run_name = "dircheck"

    exit_code = hisso_log_run.main(
        [
            "--config",
            str(config_path),
            "--output-dir",
            str(out_root),
            "--run-name",
            run_name,
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 0

    run_dir = out_root / run_name
    assert run_dir.exists() and run_dir.is_dir()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "events.csv").exists()
