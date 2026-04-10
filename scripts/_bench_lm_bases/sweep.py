# ruff: noqa: F403,F405
from __future__ import annotations

from .shared import *


def _parse_bases(cli_value: Optional[str], cfg: Dict[str, Any]) -> List[str]:
    if cli_value:
        bases = [b.strip() for b in cli_value.split(",") if b.strip()]
        return bases
    bench_bases = cfg.get("bench", {}).get("bases") or []
    if bench_bases:
        return [str(b).strip() for b in bench_bases if str(b).strip()]
    discovered = list_bases()
    return discovered if discovered else ["respsann", "waveresnet"]


def _flatten_sweep_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a nested sweep spec into dotted-key -> values mappings."""

    out: Dict[str, Any] = {}

    def _walk(prefix: str, node: Any) -> None:
        if not isinstance(node, dict):
            if prefix:
                out[prefix] = node
            return
        for k, v in node.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                _walk(key, v)
            else:
                out[key] = v

    _walk("", spec)
    return out


def _set_by_dotted_path(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = [p for p in str(dotted).split(".") if p]
    if not parts:
        return
    cur: Dict[str, Any] = cfg
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _format_sweep_slug(overrides: Dict[str, Any]) -> str:
    if not overrides:
        return "default"
    abbrev = {
        "lr": "lr",
        "warmup_steps": "wu",
        "batch_tokens": "btok",
        "grad_accum_steps": "ga",
        "weight_decay": "wd",
        "amp": "amp",
        "grad_checkpoint": "gckpt",
        "d_model": "dm",
        "n_layers": "L",
        "n_heads": "H",
        "d_mlp": "mlp",
        "max_length": "T",
    }
    parts: List[str] = []
    for key in sorted(overrides.keys()):
        leaf = str(key).split(".")[-1]
        tag = abbrev.get(leaf, leaf)
        val = overrides[key]
        if isinstance(val, bool):
            sval = "t" if val else "f"
        elif isinstance(val, float):
            sval = f"{val:g}"
        else:
            sval = str(val)
        sval = sval.replace(" ", "")
        sval = re.sub(r"[^A-Za-z0-9._-]+", "", sval)
        parts.append(f"{tag}{sval}")
    slug = "_".join(parts)
    return (slug[:120] or "default") if slug else "default"


def _expand_sweep_configs(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return a list of sweep runs: [{id, slug, overrides, cfg}, ...]."""

    sweep = cfg.get("sweep") or {}
    if not sweep:
        return [
            {
                "id": 0,
                "slug": "default",
                "overrides": {},
                "cfg": cfg,
            }
        ]
    if not isinstance(sweep, dict):
        raise SystemExit("Config key 'sweep' must be a mapping.")

    flat = _flatten_sweep_spec(sweep)
    dims: List[tuple[str, List[Any]]] = []
    for key, values in sorted(flat.items(), key=lambda kv: str(kv[0])):
        if values is None:
            continue
        if isinstance(values, (list, tuple)):
            opts = list(values)
        else:
            opts = [values]
        if not opts:
            continue
        dims.append((str(key), opts))

    if not dims:
        return [
            {
                "id": 0,
                "slug": "default",
                "overrides": {},
                "cfg": cfg,
            }
        ]

    keys = [k for k, _ in dims]
    value_lists = [vals for _, vals in dims]
    expanded: List[Dict[str, Any]] = []
    for idx, combo in enumerate(product(*value_lists)):
        run_cfg = copy.deepcopy(cfg)
        run_cfg.pop("sweep", None)
        overrides: Dict[str, Any] = {}
        for key, val in zip(keys, combo):
            overrides[key] = val
            _set_by_dotted_path(run_cfg, key, val)
        expanded.append(
            {
                "id": idx,
                "slug": _format_sweep_slug(overrides),
                "overrides": overrides,
                "cfg": run_cfg,
            }
        )
    return expanded
