# ruff: noqa: F403,F405
from __future__ import annotations

from .data import resolve_geo_shape
from .shared import *


def count_geosparse_params(
    input_dim: int,
    output_dim: int,
    *,
    depth: int,
    k: int,
    shape: Optional[Tuple[int, int]],
    activation_type: str,
    activation_config_json: Optional[str] = None,
) -> int:
    activation_config = (
        json.loads(str(activation_config_json)) if activation_config_json is not None else None
    )
    model = GeoSparseNet(
        int(input_dim),
        int(output_dim),
        shape=resolve_geo_shape(input_dim, shape),
        depth=int(depth),
        k=int(k),
        activation_type=activation_type,
        activation_config=activation_config,
        norm="rms",
        drop_path_max=0.0,
        residual_alpha_init=0.0,
        bias=True,
        compute_mode="gather",
        seed=1337,
    )
    return count_params(model, trainable_only=True)


@lru_cache(maxsize=None)
def count_dense_params(
    input_dim: int,
    output_dim: int,
    *,
    depth: int,
    hidden_units: int,
    activation_type: str,
) -> int:
    model = PSANNNet(
        int(input_dim),
        int(output_dim),
        hidden_layers=int(depth),
        hidden_units=int(hidden_units),
        hidden_width=int(hidden_units),
        activation_type=activation_type,
    )
    return count_params(model, trainable_only=True)


def _best_dense_width_for_target(
    *,
    input_dim: int,
    output_dim: int,
    dense_depth: int,
    target_params: int,
    max_width: int = 16384,
) -> Dict[str, int]:
    if max_width < 1:
        raise ValueError("max_width must be >= 1")

    def params_for(width: int) -> int:
        return count_dense_params(
            input_dim,
            output_dim,
            depth=dense_depth,
            hidden_units=int(width),
            activation_type="relu",
        )

    lo = 1
    hi = 1
    hi_params = params_for(hi)
    while hi_params < target_params and hi < max_width:
        hi = min(max_width, hi * 2)
        hi_params = params_for(hi)

    if hi_params < target_params:
        # Can't reach target within width budget; return best-at-max_width.
        mismatch = abs(hi_params - target_params)
        return {"dense_width": hi, "dense_params": hi_params, "mismatch": mismatch}

    # Binary search for smallest width whose params >= target.
    lo = 1
    while lo < hi:
        mid = (lo + hi) // 2
        mid_params = params_for(mid)
        if mid_params < target_params:
            lo = mid + 1
        else:
            hi = mid

    candidates = [lo]
    if lo > 1:
        candidates.append(lo - 1)
    # Guard against tiny non-monotonicity / initialization quirks.
    if lo + 1 <= max_width:
        candidates.append(lo + 1)

    best_width = None
    best_params = None
    best_mismatch = None
    for w in sorted(set(candidates)):
        p = params_for(w)
        mismatch = abs(p - target_params)
        if best_mismatch is None or mismatch < best_mismatch:
            best_width = w
            best_params = p
            best_mismatch = mismatch
    return {
        "dense_width": int(best_width),
        "dense_params": int(best_params),
        "mismatch": int(best_mismatch),
    }


def match_dense_width_to_geosparse(
    *,
    input_dim: int,
    output_dim: int,
    geo_depth: int,
    geo_k: int,
    geo_shape: Optional[Tuple[int, int]],
    geo_activation_type: str,
    geo_activation_config_json: Optional[str],
    dense_depth: int,
    tol: float,
) -> Dict[str, Any]:
    target = count_geosparse_params(
        input_dim,
        output_dim,
        depth=geo_depth,
        k=geo_k,
        shape=geo_shape,
        activation_type=geo_activation_type,
        activation_config_json=geo_activation_config_json,
    )
    preferred = _best_dense_width_for_target(
        input_dim=input_dim,
        output_dim=output_dim,
        dense_depth=dense_depth,
        target_params=target,
    )
    rel_mismatch = float(preferred["mismatch"]) / max(1, target)
    if rel_mismatch <= tol:
        return {
            "target_params": int(target),
            "dense_depth": int(dense_depth),
            "dense_width": int(preferred["dense_width"]),
            "dense_params": int(preferred["dense_params"]),
            "rel_mismatch": rel_mismatch,
        }

    # Fallback: for small targets, parameter granularity can make exact matching
    # impossible at higher depths. Try shallower dense models to honor tol.
    best = {
        "target_params": int(target),
        "dense_depth": int(dense_depth),
        "dense_width": int(preferred["dense_width"]),
        "dense_params": int(preferred["dense_params"]),
        "rel_mismatch": rel_mismatch,
        "_mismatch_abs": int(preferred["mismatch"]),
    }
    for depth in range(1, max(1, int(dense_depth))):
        cand = _best_dense_width_for_target(
            input_dim=input_dim,
            output_dim=output_dim,
            dense_depth=depth,
            target_params=target,
        )
        cand_rel = float(cand["mismatch"]) / max(1, target)
        if cand_rel < best["rel_mismatch"]:
            best = {
                "target_params": int(target),
                "dense_depth": int(depth),
                "dense_width": int(cand["dense_width"]),
                "dense_params": int(cand["dense_params"]),
                "rel_mismatch": cand_rel,
                "_mismatch_abs": int(cand["mismatch"]),
            }
        if cand_rel <= tol:
            best = {
                "target_params": int(target),
                "dense_depth": int(depth),
                "dense_width": int(cand["dense_width"]),
                "dense_params": int(cand["dense_params"]),
                "rel_mismatch": cand_rel,
                "_mismatch_abs": int(cand["mismatch"]),
            }
            break

    if best["rel_mismatch"] > tol:
        raise RuntimeError(
            f"Could not match dense params within tol={tol:.3f}: "
            f"target={target} best_dense={best['dense_params']} "
            f"(depth={best['dense_depth']} width={best['dense_width']}) "
            f"rel_mismatch={best['rel_mismatch']:.3f}"
        )
    if best["dense_depth"] != dense_depth:
        print(
            f"[match] dense_depth fallback {dense_depth} -> {best['dense_depth']} "
            f"to satisfy tol={tol:.3f} (abs_mismatch={best['_mismatch_abs']})"
        )
    best.pop("_mismatch_abs", None)
    return best


def build_geosparse_estimator(
    *,
    input_dim: int,
    shape: Optional[Tuple[int, int]],
    geo_depth: int,
    geo_k: int,
    activation_type: str,
    activation_config: Optional[Dict[str, Any]],
    amp: bool,
    amp_dtype: str,
    compile: bool,
    compile_backend: str,
    compile_mode: str,
    compile_fullgraph: bool,
    compile_dynamic: bool,
    device: str,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
) -> GeoSparseRegressor:
    return GeoSparseRegressor(
        hidden_layers=geo_depth,
        activation_type=activation_type,
        activation=activation_config,
        shape=resolve_geo_shape(input_dim, shape),
        k=geo_k,
        compute_mode="gather",
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        optimizer="adam",
        weight_decay=0.0,
        amp=amp,
        amp_dtype=amp_dtype,
        compile=compile,
        compile_backend=compile_backend,
        compile_mode=compile_mode,
        compile_fullgraph=compile_fullgraph,
        compile_dynamic=compile_dynamic,
        device=device,
        random_state=seed,
    )


def build_dense_estimator(
    *,
    dense_depth: int,
    dense_width: int,
    amp: bool,
    amp_dtype: str,
    compile: bool,
    compile_backend: str,
    compile_mode: str,
    compile_fullgraph: bool,
    compile_dynamic: bool,
    device: str,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
) -> PSANNRegressor:
    return PSANNRegressor(
        hidden_layers=dense_depth,
        hidden_units=dense_width,
        activation_type="relu",
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        optimizer="adam",
        weight_decay=0.0,
        amp=amp,
        amp_dtype=amp_dtype,
        compile=compile,
        compile_backend=compile_backend,
        compile_mode=compile_mode,
        compile_fullgraph=compile_fullgraph,
        compile_dynamic=compile_dynamic,
        device=device,
        random_state=seed,
    )
