# ruff: noqa: F403,F405
from __future__ import annotations

from .env import DATA_ROOT, ensure_jena_dataset
from .shared import *


def build_windows(
    df: pd.DataFrame, feature_cols: List[str], target_col: str, context: int, horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    values = df[feature_cols].values.astype(np.float32)
    target = df[target_col].values.astype(np.float32)
    idxs = []
    for start in range(context, len(df) - horizon):
        end = start
        idxs.append((start - context, start, end, end + horizon))
    Xw = np.stack([values[s:e] for (s, e, _, __) in idxs], axis=0)
    Yw = np.stack([target[s2:e2] for (_, __, s2, e2) in idxs], axis=0)
    return Xw, Yw


def split_train_val_test(
    X: np.ndarray, y: np.ndarray, val_frac: float = 0.15, test_frac: float = 0.15
):
    n = X.shape[0]
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test
    return (
        X[:n_train],
        y[:n_train],
        X[n_train : n_train + n_val],
        y[n_train : n_train + n_val],
        X[n_train + n_val :],
        y[n_train + n_val :],
    )


def load_jena_light(context: int = 72, horizon: int = 36, subset_days: Optional[int] = 120):
    def _norm(s: str) -> str:
        trans = {"–": "-", "—": "-", "‑": "-", "−": "-"}
        return "".join(trans.get(ch, ch) for ch in s).lower()

    base = DATA_ROOT / "Jena Climate 2009-2016"
    csv = base / "jena_climate_2009_2016.csv"
    if not csv.exists():
        try:
            csv = ensure_jena_dataset()
        except FileNotFoundError:
            csv = None
    if csv is None or not Path(csv).exists():
        candidates = [d for d in DATA_ROOT.iterdir() if d.is_dir() and "jena" in _norm(d.name)]
        found = None
        for d in candidates:
            hits = list(d.rglob("jena_climate_2009_2016.csv"))
            if hits:
                found = hits[0]
                break
            hits = list(d.rglob("*jena*climate*2016*.csv"))
            if hits:
                found = hits[0]
                break
        if found is None:
            raise FileNotFoundError(f"Could not find Jena climate CSV under {DATA_ROOT}")
        csv = found
    df = pd.read_csv(csv)
    target_col = next(
        (c for c in df.columns if c.strip().lower().startswith("t ") or "degc" in c.lower()), None
    )
    if target_col is None:
        raise RuntimeError("Could not find temperature column (e.g., T (degC))")
    num_df = df.select_dtypes(include=[np.number]).copy()
    if subset_days is not None:
        num_df = num_df.tail(subset_days * 144)
    num_df = (num_df - num_df.mean()) / (num_df.std().replace(0, 1.0))
    feature_cols = list(num_df.columns)
    Xw, Yw = build_windows(num_df, feature_cols, target_col, context, horizon)
    return split_train_val_test(Xw, Yw)


def load_beijing_light(
    station_name: str = "Guanyuan",
    context: int = 24,
    horizon: int = 6,
    subset_days: Optional[int] = 120,
):
    base = DATA_ROOT / "Beijing Air Quality"
    station_file = None
    for p in base.glob("PRSA_Data_*_20130301-20170228.csv"):
        if station_name.lower() in p.name.lower():
            station_file = p
            break
    if station_file is None:
        raise FileNotFoundError(f"Could not find station file containing {station_name}")
    df = pd.read_csv(station_file)
    target_col = (
        "PM2.5" if "PM2.5" in df.columns else df.select_dtypes(include=[np.number]).columns[0]
    )
    num_df = df.select_dtypes(include=[np.number]).copy().ffill().bfill().fillna(0.0)
    if subset_days is not None:
        num_df = num_df.tail(subset_days * 24)
    num_df = (num_df - num_df.mean()) / (num_df.std().replace(0, 1.0))
    feature_cols = list(num_df.columns)
    Xw, Yw = build_windows(num_df, feature_cols, target_col, context, horizon)
    return split_train_val_test(Xw, Yw)


def load_eaf_temp_lite(
    context: int = 16, horizon: int = 1, heats_limit: int = 5, min_rows: int = 120
):
    path = DATA_ROOT / "Industrial Data from the Electric Arc Furnace" / "eaf_temp.csv"
    df = pd.read_csv(path)
    if not {"HEATID", "DATETIME", "TEMP"}.issubset(df.columns):
        raise RuntimeError("Missing expected columns in eaf_temp.csv")
    df["DATETIME"] = pd.to_datetime(df["DATETIME"], errors="coerce")
    df = df.dropna(subset=["DATETIME"]).sort_values(["HEATID", "DATETIME"])
    heats_df = df.groupby("HEATID").size().reset_index(name="n").sort_values("n", ascending=False)
    selected = heats_df.query("n >= @min_rows").head(heats_limit)
    if selected.empty:
        fallback = heats_df.head(heats_limit)
        if fallback.empty:
            raise RuntimeError("No heats found in eaf_temp.csv")
        print(
            f"[warn] No EAF heats with >= {min_rows} rows; using top {len(fallback)} heats with >= {int(fallback['n'].min())} rows instead."
        )
        selected = fallback
    parts = []
    for hid in selected["HEATID"]:
        seg = df[df["HEATID"] == hid].copy()
        num_cols = ["TEMP"] + (["VALO2_PPM"] if "VALO2_PPM" in seg.columns else [])
        seg_num = seg[num_cols]
        seg_num = (seg_num - seg_num.mean()) / (seg_num.std().replace(0, 1.0))
        Xw, Yw = build_windows(
            seg_num, feature_cols=num_cols, target_col="TEMP", context=context, horizon=horizon
        )
        if Xw.size == 0:
            continue
        parts.append((Xw, Yw))
    if not parts:
        raise RuntimeError("No EAF heats with sufficient rows found for lite run")
    X = np.concatenate([p[0] for p in parts], axis=0)
    Y = np.concatenate([p[1] for p in parts], axis=0)
    return split_train_val_test(X, Y)
