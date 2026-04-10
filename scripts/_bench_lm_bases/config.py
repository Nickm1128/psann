# ruff: noqa: F403,F405
from __future__ import annotations

from .shared import *


def _now_utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _system_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version.split(" ")[0],
        "platform": sys.platform,
        "torch": torch.__version__,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append(
                {
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                }
            )
        info["gpus"] = gpus
    return info


def _default_config() -> Dict[str, Any]:
    return {
        "bench": {
            "run_name": "base_shootout",
            "out_dir": "reports/benchmarks",
            "bases": [],
            "seeds": [1337],
            "reuse_tokenizer": True,
            "with_lm_eval": False,
            "save_run_logs": False,
            "lm_eval_tasks": ["lambada_openai", "hellaswag"],
            "lm_eval_limit": 256,
            "lm_eval_num_fewshot": 0,
        },
        "data": {
            "dataset": "iohadrubin/wikitext-103-raw-v1",
            "name": None,
            "data_files": None,
            "revision": None,
            "train_split": "train",
            "val_split": "validation",
            "text_key": "text",
            "streaming": True,
            "shuffle": True,
            "shuffle_buffer": 10000,
            "ascii_only": False,
            "languages": [],
            "lang_threshold": 0.8,
            "max_length": 512,
        },
        "tokenizer": {
            "backend": "tokenizers",
            "vocab_size": 16384,
            "min_frequency": 2,
            "sample_limit": 20000,
            "save_dir": "runs/tokenizers/base_compare_quick",
            "model_path": None,
            "special_tokens_map_path": None,
        },
        "train": {
            "d_model": 256,
            "n_layers": 4,
            "n_heads": 4,
            "d_mlp": 1024,
            "dropout": 0.0,
            "positional_encoding": "rope",
            "attn_impl": "auto",
            "batch_tokens": 32768,
            "grad_accum_steps": 1,
            "lr": 2e-4,
            "weight_decay": 0.01,
            "warmup_steps": 200,
            "amp": "bf16",
            "grad_checkpoint": False,
            "ddp": "off",
            "max_steps": 300,
            "log_interval_steps": 50,
            "save_interval_steps": 500,
            "torch_compile": False,
            "torch_compile_mode": "default",
            "torch_compile_fullgraph": False,
            "torch_compile_dynamic": False,
            "sine_params": {
                "amp_init": 1.0,
                "freq_init": 1.0,
                "damp_init": 0.01,
                "trainable": True,
            },
        },
        "eval": {
            "batch_tokens": 32768,
            "max_tokens": 200000,
            "max_batches": 0,
        },
        "distill": {
            "enabled": False,
            "teacher_base": "transformer",
            "teacher_ckpt": None,
            "teacher_max_steps": 0,
            "alpha": 0.5,
            "temperature": 2.0,
        },
        "sweep": {},
    }


class _TeeStream:
    def __init__(self, *streams: Any):
        self._streams = [s for s in streams if s is not None]

    def write(self, data: str) -> None:
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass
        if "\n" in data:
            self.flush()

    def flush(self) -> None:
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self) -> bool:  # pragma: no cover - passthrough helper
        for s in self._streams:
            fn = getattr(s, "isatty", None)
            if callable(fn):
                try:
                    if bool(fn()):
                        return True
                except Exception:
                    pass
        return False


@contextmanager
def _tee_run_logs(log_path: Path, *, enabled: bool) -> Iterable[None]:
    if not enabled:
        yield
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        orig_out = sys.stdout
        orig_err = sys.stderr
        sys.stdout = _TeeStream(orig_out, fh)  # type: ignore[assignment]
        sys.stderr = _TeeStream(orig_err, fh)  # type: ignore[assignment]
        try:
            yield
        finally:
            sys.stdout = orig_out  # type: ignore[assignment]
            sys.stderr = orig_err  # type: ignore[assignment]


def _coerce_betas(value: Any, default: tuple[float, float] = (0.9, 0.95)) -> tuple[float, float]:
    if value is None:
        return default
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if len(parts) != 2:
            return default
        try:
            return (float(parts[0]), float(parts[1]))
        except Exception:
            return default
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return (float(value[0]), float(value[1]))
        except Exception:
            return default
    return default


def _coerce_ddp(value: Any, default: str = "off") -> str:
    # YAML 1.1 parses "on"/"off" as booleans; accept that and map to the expected strings.
    if value is None:
        return default
    if isinstance(value, bool):
        return "on" if value else "off"
    return str(value)


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)  # type: ignore[index]
        else:
            base[key] = value
    return base


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    write_json(path, payload)
