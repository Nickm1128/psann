from pathlib import Path

import nbformat as nbf


def md(text: str):
    from textwrap import dedent

    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    from textwrap import dedent

    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


def main() -> None:
    nb = nbf.v4.new_notebook()
    cells_data = []
    cells_data.extend(
        [
            (
                "md",
                """
                # ResPSANN Compute-Parity Experiments (Colab Runner)

                This notebook orchestrates the experiments described in `plan.txt` using the datasets summarised in `data_descriptions.txt`. Execute it inside Google Colab (GPU runtime recommended).
                """,
            ),
            (
                "md",
                """
                ## Run Checklist
                - Prefer Google Colab with a GPU runtime (recommended) before running any experiments.
                - Let the setup cell install the latest published `psann` package via `pip`; no repository clone is required.
                - Upload or mount the dataset directory so that `DATA_ROOT` points to it (defaults to `<working dir>/datasets`).
                - Adjust `GLOBAL_CONFIG` and the experiment toggles before launching training to stay within the Colab budget.
                - Keep the heavy training cells disabled until you are ready to execute them in Colab.
                """,
            ),
            (
                "code",
                """
                import os
                import sys
                from pathlib import Path

                COLAB = "google.colab" in sys.modules

                DEFAULT_PROJECT_ROOT = Path("/content") if COLAB else Path.cwd()
                PROJECT_ROOT = Path(os.getenv("PSANN_PROJECT_ROOT", DEFAULT_PROJECT_ROOT)).resolve()

                DATA_ROOT = Path(os.getenv("PSANN_DATA_ROOT", PROJECT_ROOT / "datasets")).resolve()
                RESULTS_ROOT = Path(os.getenv("PSANN_RESULTS_ROOT", PROJECT_ROOT / "colab_results")).resolve()
                FIGURE_ROOT = RESULTS_ROOT / "figures"

                RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
                FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

                if not DATA_ROOT.exists():
                    print(f"[WARN] DATA_ROOT {DATA_ROOT} does not exist yet. Upload datasets or update PSANN_DATA_ROOT.")

                print(f"Colab runtime         : {COLAB}")
                print(f"Project root          : {PROJECT_ROOT}")
                print(f"Dataset root          : {DATA_ROOT}")
                print(f"Results directory     : {RESULTS_ROOT}")
                """,
            ),
            (
                "code",
                """
                import subprocess
                import sys

                def install_dependencies():
                    base_packages = [
                        "psann",
                        "pandas>=2.0",
                        "numpy>=1.24",
                        "scikit-learn>=1.3",
                        "torch>=2.1",
                        "torchvision>=0.16",
                        "torchaudio>=2.1",
                        "lightgbm>=4.0",
                        "xgboost>=1.7",
                        "catboost>=1.2",
                        "shap>=0.44",
                        "matplotlib>=3.7",
                        "seaborn>=0.13",
                        "plotly>=5.18",
                        "imbalanced-learn>=0.12",
                        "tqdm>=4.66",
                        "einops>=0.7",
                        "rich>=13.7",
                    ]
                    print("Installing psann and supporting packages...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + base_packages)

                if COLAB:
                    install_dependencies()
                else:
                    print("Skipping dependency installation because we are not inside Colab.")
                """,
            ),
        ]
    )
    cells_data.extend(
        [
            (
                "code",
                """
                def load_jena_climate(data_root: Path) -> pd.DataFrame:
                    path = data_root / "Jena Climate 2009-2016" / "jena_climate_2009_2016.csv"
                    if not path.exists():
                        raise FileNotFoundError(f"Jena climate CSV not found at {path}")
                    df = pd.read_csv(path)
                    df["datetime"] = pd.to_datetime(df["Date Time"], dayfirst=True)
                    df = df.drop(columns=["Date Time"])
                    numeric_cols = [col for col in df.columns if col != "datetime"]
                    df[numeric_cols] = df[numeric_cols].astype(np.float32)
                    df = df.sort_values("datetime").reset_index(drop=True)
                    return df


                def prepare_jena_bundle(
                    df: pd.DataFrame,
                    target: str = "T (degC)",
                    context_steps: int = 72,
                    horizon_steps: int = 36,
                    resample_factor: int = 1,
                ) -> DatasetBundle:
                    df = df.copy()
                    if resample_factor > 1:
                        df = df.iloc[::resample_factor].reset_index(drop=True)
                    df = add_calendar_features(df, "datetime")
                    feature_cols = [c for c in df.columns if c not in ("datetime", target)]
                    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

                    values = df[feature_cols].to_numpy(dtype=np.float32)
                    target_values = df[target].to_numpy(dtype=np.float32)
                    timestamps = df["datetime"].to_numpy()

                    windows = []
                    targets = []
                    ts_list = []
                    for idx in range(context_steps, len(df) - horizon_steps):
                        window = values[idx - context_steps : idx]
                        target_value = target_values[idx + horizon_steps]
                        windows.append(window)
                        targets.append(target_value)
                        ts_list.append(timestamps[idx])
                    X = np.stack(windows)
                    y = np.asarray(targets, dtype=np.float32)[:, None]
                    ts = np.asarray(ts_list)

                    df_windows = pd.DataFrame({"datetime": ts})
                    train_df, val_df, test_df = train_val_test_split_by_time(
                        df_windows, "datetime", "2015-01-01", "2016-01-01"
                    )
                    train_idx = train_df.index.to_numpy()
                    val_idx = val_df.index.to_numpy()
                    test_idx = test_df.index.to_numpy()

                    target_slug = (
                        target.lower()
                        .replace(" ", "")
                        .replace("(", "")
                        .replace(")", "")
                        .replace("/", "")
                    )
                    bundle_name = f"Jena_{target_slug}_{context_steps}ctx_{horizon_steps}h"

                    bundle = DatasetBundle(
                        name=bundle_name,
                        task_type="regression",
                        input_kind="sequence",
                        feature_names=feature_cols,
                        target_names=[target],
                        train={"X": X[train_idx], "y": y[train_idx]},
                        val={"X": X[val_idx], "y": y[val_idx]},
                        test={"X": X[test_idx], "y": y[test_idx]},
                        metadata={
                            "context_steps": context_steps,
                            "horizon_steps": horizon_steps,
                            "resample_factor": resample_factor,
                        },
                    )
                    return bundle
                """,
            ),
            (
                "code",
                """
                def load_har_engineered(data_root: Path):
                    base = data_root / "Human Activity Recognition" / "UCI HAR Dataset"
                    X_train = pd.read_csv(base / "train" / "X_train.txt", delim_whitespace=True, header=None)
                    y_train = pd.read_csv(base / "train" / "y_train.txt", header=None, squeeze=True)
                    subject_train = pd.read_csv(base / "train" / "subject_train.txt", header=None, squeeze=True)

                    X_test = pd.read_csv(base / "test" / "X_test.txt", delim_whitespace=True, header=None)
                    y_test = pd.read_csv(base / "test" / "y_test.txt", header=None, squeeze=True)
                    subject_test = pd.read_csv(base / "test" / "subject_test.txt", header=None, squeeze=True)

                    y_train = y_train.values.astype(int) - 1
                    y_test = y_test.values.astype(int) - 1

                    features = (base / "features.txt").read_text().strip().splitlines()
                    feature_names = [line.split()[1] for line in features]

                    X_train.columns = feature_names
                    X_test.columns = feature_names

                    train_df = X_train.copy()
                    test_df = X_test.copy()
                    train_df["label"] = y_train
                    train_df["subject"] = subject_train.values
                    test_df["label"] = y_test
                    test_df["subject"] = subject_test.values

                    return train_df, test_df, feature_names


                def prepare_har_engineered_bundle(
                    train_df: pd.DataFrame,
                    test_df: pd.DataFrame,
                    feature_names: List[str],
                    val_fraction: float = 0.15,
                ) -> DatasetBundle:
                    from sklearn.model_selection import StratifiedShuffleSplit

                    X = train_df[feature_names].to_numpy(dtype=np.float32)
                    y = train_df["label"].to_numpy(dtype=np.int64)
                    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=GLOBAL_CONFIG["seed"])
                    train_idx, val_idx = next(splitter.split(X, y))

                    X_train = X[train_idx]
                    y_train = y[train_idx][:, None]
                    X_val = X[val_idx]
                    y_val = y[val_idx][:, None]

                    X_test = test_df[feature_names].to_numpy(dtype=np.float32)
                    y_test = test_df["label"].to_numpy(dtype=np.int64)[:, None]

                    bundle = DatasetBundle(
                        name="HAR_engineered",
                        task_type="classification",
                        input_kind="tabular",
                        feature_names=feature_names,
                        target_names=["activity"],
                        train={"X": X_train, "y": y_train},
                        val={"X": X_val, "y": y_val},
                        test={"X": X_test, "y": y_test},
                        metadata={
                            "n_classes": 6,
                            "label_mapping": {
                                0: "WALKING",
                                1: "WALKING_UPSTAIRS",
                                2: "WALKING_DOWNSTAIRS",
                                3: "SITTING",
                                4: "STANDING",
                                5: "LAYING",
                            },
                        },
                    )
                    return bundle


                def load_har_raw_sequences(data_root: Path):
                    base = data_root / "Human Activity Recognition" / "UCI HAR Dataset"
                    axes = [
                        "body_acc_x",
                        "body_acc_y",
                        "body_acc_z",
                        "body_gyro_x",
                        "body_gyro_y",
                        "body_gyro_z",
                        "total_acc_x",
                        "total_acc_y",
                        "total_acc_z",
                    ]

                    def load_split(split: str):
                        signals = []
                        for axis in axes:
                            path = base / split / "Inertial Signals" / f"{axis}_{split}.txt"
                            arr = np.loadtxt(path)
                            signals.append(arr[:, :, None])
                        X = np.concatenate(signals, axis=2).astype(np.float32)
                        y = np.loadtxt(base / split / f"y_{split}.txt").astype(int) - 1
                        return X, y

                    X_train, y_train = load_split("train")
                    X_test, y_test = load_split("test")
                    return X_train, y_train, X_test, y_test, axes


                def prepare_har_raw_bundle(
                    X_train: np.ndarray,
                    y_train: np.ndarray,
                    X_test: np.ndarray,
                    y_test: np.ndarray,
                    val_fraction: float = 0.15,
                ) -> DatasetBundle:
                    from sklearn.model_selection import StratifiedShuffleSplit

                    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=GLOBAL_CONFIG["seed"])
                    train_idx, val_idx = next(splitter.split(X_train, y_train))
                    bundle = DatasetBundle(
                        name="HAR_raw_sequence",
                        task_type="classification",
                        input_kind="sequence",
                        feature_names=[f"axis_{i}" for i in range(X_train.shape[2])],
                        target_names=["activity"],
                        train={"X": X_train[train_idx], "y": y_train[train_idx][:, None]},
                        val={"X": X_train[val_idx], "y": y_train[val_idx][:, None]},
                        test={"X": X_test, "y": y_test[:, None]},
                        metadata={
                            "sequence_length": X_train.shape[1],
                            "n_channels": X_train.shape[2],
                            "n_classes": 6,
                        },
                    )
                    return bundle
                """,
            ),
            (
                "code",
                """
                def load_rossmann_frames(data_root: Path):
                    base = data_root / "Kaggle Rossmann Store Sales" / "rossmann-store-sales"
                    train_path = base / "train.csv"
                    test_path = base / "test.csv"
                    store_path = base / "store.csv"
                    train = pd.read_csv(train_path, parse_dates=["Date"])
                    test = pd.read_csv(test_path, parse_dates=["Date"])
                    store = pd.read_csv(store_path)
                    return train, test, store


                def is_promo2_active(row: pd.Series) -> int:
                    if not row.get("Promo2", 0):
                        return 0
                    month = row["Date"].month
                    if isinstance(row["PromoInterval"], str) and row["PromoInterval"]:
                        months = {
                            m.strip(): i
                            for i, m in enumerate(
                                ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                                start=1,
                            )
                        }
                        promo_months = [months.get(m, 0) for m in row["PromoInterval"].split(",")]
                        return int(month in promo_months)
                    return 0


                def preprocess_rossmann(train: pd.DataFrame, store: pd.DataFrame):
                    df = train.merge(store, on="Store", how="left")
                    df = df[df["Open"] != 0].copy()

                    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(df["CompetitionDistance"].median())
                    df["CompetitionOpenSinceYear"] = df["CompetitionOpenSinceYear"].fillna(df["CompetitionOpenSinceYear"].median())
                    df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(df["CompetitionOpenSinceMonth"].median())
                    df["Promo2SinceWeek"] = df["Promo2SinceWeek"].fillna(0)
                    df["Promo2SinceYear"] = df["Promo2SinceYear"].fillna(0)
                    df["PromoInterval"] = df["PromoInterval"].fillna("")

                    df["Date"] = pd.to_datetime(df["Date"])
                    df["Year"] = df["Date"].dt.year
                    df["Month"] = df["Date"].dt.month
                    df["Day"] = df["Date"].dt.day
                    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
                    df["DayOfWeek"] = df["Date"].dt.dayofweek

                    df["IsPromo2Month"] = df.apply(is_promo2_active, axis=1)

                    state_holiday_map = {"0": "None", "a": "PublicHoliday", "b": "EasterHoliday", "c": "Christmas"}
                    df["StateHoliday"] = df["StateHoliday"].replace(state_holiday_map)

                    categorical_cols = ["StoreType", "Assortment", "StateHoliday", "PromoInterval"]
                    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

                    df["CustomersLag7"] = df.groupby("Store")["Customers"].shift(7)
                    df["SalesLag7"] = df.groupby("Store")["Sales"].shift(7)
                    df["SalesMA14"] = df.groupby("Store")["Sales"].transform(lambda s: s.rolling(14, min_periods=1).mean())
                    df["PromoMovingAvg"] = df.groupby("Store")["Promo"].transform(lambda s: s.rolling(30, min_periods=1).mean())

                    df = df.dropna().reset_index(drop=True)

                    feature_cols = [c for c in df.columns if c not in ("Sales", "Date")]
                    target_col = "Sales"
                    return df, feature_cols, target_col


                def prepare_rossmann_bundle(df: pd.DataFrame, feature_cols: List[str], target_col: str) -> DatasetBundle:
                    df = df.copy()
                    train_df, val_df, test_df = train_val_test_split_by_time(df, "Date", "2015-08-01", "2015-09-01")

                    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
                    y_train = train_df[target_col].to_numpy(dtype=np.float32)[:, None]
                    X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
                    y_val = val_df[target_col].to_numpy(dtype=np.float32)[:, None]
                    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
                    y_test = test_df[target_col].to_numpy(dtype=np.float32)[:, None]

                    bundle = DatasetBundle(
                        name="Rossmann_sales",
                        task_type="regression",
                        input_kind="tabular",
                        feature_names=feature_cols,
                        target_names=[target_col],
                        train={"X": X_train, "y": y_train},
                        val={"X": X_val, "y": y_val},
                        test={"X": X_test, "y": y_test},
                        metadata={
                            "train_range": [str(train_df["Date"].min()), str(train_df["Date"].max())],
                            "val_range": [str(val_df["Date"].min()), str(val_df["Date"].max())],
                            "test_range": [str(test_df["Date"].min()), str(test_df["Date"].max())],
                        },
                    )
                    return bundle
                """,
            ),
        ]
    )
    cells_data.extend(
        [
            (
                "code",
                """
                def load_beijing_stations(data_root: Path) -> Dict[str, pd.DataFrame]:
                    base = data_root / "Beijing Air Quality"
                    if not base.exists():
                        raise FileNotFoundError(f"Beijing Air Quality directory not found at {base}")
                    stations: Dict[str, pd.DataFrame] = {}
                    for csv_path in base.glob("PRSA_Data_*.csv"):
                        station_name = csv_path.stem.replace("PRSA_Data_", "")
                        print(f"Loading Beijing station {station_name}...")
                        df = pd.read_csv(csv_path)
                        df["datetime"] = pd.to_datetime(
                            df[["year", "month", "day", "hour"]].rename(columns=str)
                        )
                        df = df.sort_values("datetime").reset_index(drop=True)
                        if "No" in df.columns:
                            df = df.drop(columns=["No"])
                        stations[station_name] = df
                    return stations


                def preprocess_beijing_station(df: pd.DataFrame, target_col: str = "PM2.5") -> Tuple[pd.DataFrame, pd.DataFrame]:
                    df = df.copy()
                    pollutant_cols = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
                    meteorology_cols = ["PRES", "DEWP", "TEMP", "RAIN", "WSPM"]
                    for col in pollutant_cols + meteorology_cols:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors="coerce")
                    mask = df[pollutant_cols + meteorology_cols].isna()
                    df[pollutant_cols + meteorology_cols] = df[pollutant_cols + meteorology_cols].interpolate(limit=6, limit_direction="both")
                    df[pollutant_cols + meteorology_cols] = df[pollutant_cols + meteorology_cols].fillna(method="ffill").fillna(method="bfill")

                    calendar = pd.DataFrame(
                        {
                            "hour": df["datetime"].dt.hour,
                            "dow": df["datetime"].dt.dayofweek,
                            "month": df["datetime"].dt.month,
                        }
                    )
                    calendar["hour_sin"] = np.sin(2 * np.pi * calendar["hour"] / 24.0)
                    calendar["hour_cos"] = np.cos(2 * np.pi * calendar["hour"] / 24.0)
                    calendar["dow_sin"] = np.sin(2 * np.pi * calendar["dow"] / 7.0)
                    calendar["dow_cos"] = np.cos(2 * np.pi * calendar["dow"] / 7.0)
                    calendar["month_sin"] = np.sin(2 * np.pi * calendar["month"] / 12.0)
                    calendar["month_cos"] = np.cos(2 * np.pi * calendar["month"] / 12.0)

                    feature_frame = pd.concat(
                        [df[["datetime", target_col]], df[pollutant_cols + meteorology_cols], calendar],
                        axis=1,
                    )
                    mask_frame = mask.astype(np.float32)
                    mask_frame.columns = [f"{col}_mask" for col in mask_frame.columns]
                    feature_frame = pd.concat([feature_frame, mask_frame], axis=1)
                    return feature_frame, mask_frame


                def build_temporal_windows(
                    frame: pd.DataFrame,
                    target_col: str,
                    feature_cols: List[str],
                    context: int,
                    horizon: int,
                    drop_na: bool = True,
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
                    values = frame[feature_cols].to_numpy(dtype=np.float32)
                    targets = frame[target_col].to_numpy(dtype=np.float32)
                    timestamps = frame["datetime"].to_numpy()
                    windows: List[np.ndarray] = []
                    y_list: List[float] = []
                    ts_list: List[np.datetime64] = []
                    for idx in range(context, len(frame) - horizon):
                        window = values[idx - context : idx]
                        target = targets[idx + horizon]
                        if drop_na and (np.isnan(window).any() or np.isnan(target)):
                            continue
                        windows.append(window)
                        y_list.append(target)
                        ts_list.append(timestamps[idx])
                    if not windows:
                        return (
                            np.empty((0, context, len(feature_cols)), dtype=np.float32),
                            np.empty((0,), dtype=np.float32),
                            np.empty((0,), dtype="datetime64[ns]"),
                        )
                    X = np.stack(windows)
                    y = np.asarray(y_list, dtype=np.float32)
                    ts = np.asarray(ts_list)
                    return X, y, ts


                def assemble_beijing_cross_station_bundle(
                    stations: Dict[str, pd.DataFrame],
                    train_stations: List[str],
                    val_station: str,
                    test_station: str,
                    target: str = "PM2.5",
                    context: int = 24,
                    horizon: int = 6,
                ) -> DatasetBundle:
                    feature_frames: Dict[str, pd.DataFrame] = {}
                    feature_cols: Optional[List[str]] = None
                    for name, df in stations.items():
                        features, _ = preprocess_beijing_station(df, target_col=target)
                        feature_frames[name] = features
                        if feature_cols is None:
                            feature_cols = [col for col in features.columns if col not in ("datetime", target)]
                    assert feature_cols is not None

                    def collect(names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
                        arrays = []
                        targets = []
                        for station_name in names:
                            frame = feature_frames[station_name]
                            X, y, _ = build_temporal_windows(frame, target, feature_cols, context, horizon)
                            arrays.append(X)
                            targets.append(y)
                        if arrays:
                            X_all = np.concatenate(arrays, axis=0)
                            y_all = np.concatenate(targets, axis=0)[:, None]
                        else:
                            X_all = np.empty((0, context, len(feature_cols)), dtype=np.float32)
                            y_all = np.empty((0, 1), dtype=np.float32)
                        return X_all, y_all

                    X_train, y_train = collect(train_stations)
                    X_val, y_val = collect([val_station])
                    X_test, y_test = collect([test_station])

                    bundle = DatasetBundle(
                        name=f"Beijing_PM25_{context}h_ctx_{horizon}h_horizon",
                        task_type="regression",
                        input_kind="sequence",
                        feature_names=feature_cols,
                        target_names=[target],
                        train={"X": X_train, "y": y_train},
                        val={"X": X_val, "y": y_val},
                        test={"X": X_test, "y": y_test},
                        metadata={
                            "context_hours": context,
                            "horizon_hours": horizon,
                            "train_stations": train_stations,
                            "val_station": val_station,
                            "test_station": test_station,
                        },
                    )
                    return bundle
                """,
            ),
        ]
    )
    cells_data.extend(
        [
            (
                "code",
                """
                import itertools
                import json
                import math
                import random
                import time
                from dataclasses import dataclass, field
                from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Literal

                import numpy as np
                import pandas as pd
                import torch
                import torch.nn as nn
                from torch.utils.data import DataLoader, TensorDataset
                from tqdm.auto import tqdm

                SEED = int(os.getenv("PSANN_GLOBAL_SEED", "2025"))
                random.seed(SEED)
                np.random.seed(SEED)
                torch.manual_seed(SEED)

                DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(f"Using device: {DEVICE}")
                if DEVICE.type == "cuda":
                    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

                GLOBAL_CONFIG: Dict[str, Any] = {
                    "seed": SEED,
                    "device": DEVICE,
                    "default_epochs": 40,
                    "default_lr": 1e-3,
                    "default_weight_decay": 0.0,
                    "default_batch_size": 256,
                    "max_time_minutes": 15.0,
                    "num_workers": 2 if DEVICE.type == "cuda" else 0,
                    "label_smoothing": 0.05,
                    "results_root": RESULTS_ROOT,
                    "figure_root": FIGURE_ROOT,
                }
                """,
            ),
            (
                "code",
                """
                @dataclass
                class TrainConfig:
                    epochs: int
                    batch_size: int
                    learning_rate: float
                    weight_decay: float = 0.0
                    max_minutes: Optional[float] = None
                    early_stopping: bool = True
                    patience: int = 10
                    gradient_clip: Optional[float] = None
                    scheduler: Optional[str] = None
                    scheduler_params: Optional[Dict[str, Any]] = None
                    warmup_steps: int = 0
                    max_batches_per_epoch: Optional[int] = None


                @dataclass
                class ModelSpec:
                    name: str
                    builder: Callable[[Tuple[int, ...], int, Dict[str, Any]], nn.Module]
                    train_config: TrainConfig
                    task_type: Literal["regression", "classification", "multitask"]
                    input_kind: Literal["tabular", "sequence"]
                    group: str = "baseline"
                    extra: Dict[str, Any] = field(default_factory=dict)
                    param_target: Optional[int] = None
                    notes: str = ""


                @dataclass
                class DatasetBundle:
                    name: str
                    task_type: Literal["regression", "classification", "multitask"]
                    input_kind: Literal["tabular", "sequence"]
                    feature_names: List[str]
                    target_names: List[str]
                    train: Dict[str, np.ndarray]
                    val: Dict[str, np.ndarray]
                    test: Dict[str, np.ndarray]
                    metadata: Dict[str, Any] = field(default_factory=dict)

                    def summary(self) -> Dict[str, Any]:
                        info = {
                            "name": self.name,
                            "task_type": self.task_type,
                            "input_kind": self.input_kind,
                            "n_train": len(self.train["X"]),
                            "n_val": len(self.val["X"]),
                            "n_test": len(self.test["X"]),
                            "input_shape": tuple(self.train["X"].shape[1:]),
                            "target_shape": tuple(self.train["y"].shape[1:]) if self.train["y"].ndim > 1 else (),
                        }
                        info.update({f"meta_{k}": v for k, v in self.metadata.items() if isinstance(v, (int, float, str))})
                        return info


                @dataclass
                class ExperimentResult:
                    dataset: str
                    task: str
                    model: str
                    group: str
                    split: str
                    metrics: Dict[str, float]
                    params: int
                    train_wall_seconds: float
                    notes: str = ""


                class ResultLogger:
                    def __init__(self) -> None:
                        self._rows: List[ExperimentResult] = []

                    def append(self, row: ExperimentResult) -> None:
                        self._rows.append(row)

                    def to_frame(self) -> pd.DataFrame:
                        records = []
                        for row in self._rows:
                            rec = {
                                "dataset": row.dataset,
                                "task": row.task,
                                "model": row.model,
                                "group": row.group,
                                "split": row.split,
                                "params": row.params,
                                "train_wall_seconds": row.train_wall_seconds,
                                "notes": row.notes,
                            }
                            rec.update(row.metrics)
                            records.append(rec)
                        return pd.DataFrame(records)


                RESULT_LOGGER = ResultLogger()
                """,
            ),
            (
                "code",
                """
                from sklearn.metrics import accuracy_score, f1_score, log_loss


                def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
                    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


                def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
                    return float(np.mean(np.abs(y_true - y_pred)))


                def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
                    denom = (np.abs(y_true) + np.abs(y_pred) + 1e-8) / 2.0
                    return float(np.mean(np.abs(y_true - y_pred) / denom))


                def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
                    ss_res = np.sum((y_true - y_pred) ** 2)
                    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                    return float(1 - ss_res / ss_tot) if ss_tot != 0 else float("nan")


                def mase(y_true: np.ndarray, y_pred: np.ndarray, seasonal_period: int = 1) -> float:
                    if len(y_true) <= seasonal_period:
                        return float("nan")
                    naive = np.mean(np.abs(np.diff(y_true, n=seasonal_period)))
                    return float(np.mean(np.abs(y_true - y_pred)) / (naive + 1e-8))


                def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
                    confidences = probs.max(axis=1)
                    predictions = probs.argmax(axis=1)
                    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
                    ece = 0.0
                    for i in range(n_bins):
                        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
                        if not np.any(mask):
                            continue
                        bin_acc = np.mean(predictions[mask] == y_true[mask])
                        bin_conf = np.mean(confidences[mask])
                        ece += np.abs(bin_acc - bin_conf) * np.mean(mask)
                    return float(ece)


                def classification_metrics(y_true: np.ndarray, logits: np.ndarray, average: str = "macro") -> Dict[str, float]:
                    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
                    preds = probs.argmax(axis=1)
                    metrics = {
                        "accuracy": float(accuracy_score(y_true, preds)),
                        "f1_macro": float(f1_score(y_true, preds, average=average)),
                        "nll": float(log_loss(y_true, probs, labels=list(range(probs.shape[1])))),
                    }
                    metrics["ece"] = expected_calibration_error(probs, y_true, n_bins=15)
                    return metrics


                def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, seasonal_period: int = 1) -> Dict[str, float]:
                    return {
                        "rmse": rmse(y_true, y_pred),
                        "mae": mae(y_true, y_pred),
                        "smape": smape(y_true, y_pred),
                        "r2": r2_score_np(y_true, y_pred),
                        "mase": mase(y_true, y_pred, seasonal_period=seasonal_period),
                    }
                """,
            ),
            (
                "code",
                """
                def build_dataloader(
                    X: np.ndarray,
                    y: np.ndarray,
                    batch_size: int,
                    shuffle: bool,
                    task_type: Literal["regression", "classification", "multitask"] = "regression",
                    drop_last: bool = False,
                ) -> DataLoader:
                    X_tensor = torch.from_numpy(X).float()
                    if task_type == "classification":
                        y_tensor = torch.from_numpy(y.squeeze()).long()
                    else:
                        y_tensor = torch.from_numpy(y.astype(np.float32))
                    dataset = TensorDataset(X_tensor, y_tensor)
                    loader = DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        drop_last=drop_last,
                        num_workers=GLOBAL_CONFIG["num_workers"],
                        pin_memory=(DEVICE.type == "cuda"),
                    )
                    return loader


                class Timer:
                    def __enter__(self):
                        self.start = time.perf_counter()
                        return self

                    def __exit__(self, exc_type, exc_value, traceback):
                        self.end = time.perf_counter()

                    @property
                    def elapsed(self) -> float:
                        return getattr(self, "end", time.perf_counter()) - getattr(self, "start", time.perf_counter())
                """,
            ),
        ]
    )
    cells_data.extend(
        [
            (
                "code",
                """
                def coerce_decimal(series: pd.Series) -> pd.Series:
                    if pd.api.types.is_numeric_dtype(series):
                        return series
                    as_str = series.astype(str).str.replace(" ", "")
                    as_str = as_str.replace({"nan": np.nan, "None": np.nan})
                    as_str = as_str.str.replace(",", ".", regex=False)
                    return pd.to_numeric(as_str, errors="coerce")


                def coerce_datetime(series: pd.Series) -> pd.Series:
                    as_str = series.astype(str).str.strip()
                    as_str = as_str.replace({"nan": np.nan, "NaT": np.nan})
                    as_str = as_str.str.replace(",", ".", n=1, regex=False)
                    return pd.to_datetime(as_str, errors="coerce")


                def ensure_float(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
                    for col in columns:
                        if col in df.columns:
                            df[col] = coerce_decimal(df[col])
                    return df


                def ensure_datetime(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
                    for col in columns:
                        if col in df.columns:
                            df[col] = coerce_datetime(df[col])
                    return df


                def add_calendar_features(frame: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
                    ts = pd.to_datetime(frame[timestamp_col])
                    frame[f"{timestamp_col}_year"] = ts.dt.year
                    frame[f"{timestamp_col}_month"] = ts.dt.month
                    frame[f"{timestamp_col}_day"] = ts.dt.day
                    frame[f"{timestamp_col}_hour"] = ts.dt.hour
                    frame[f"{timestamp_col}_dow"] = ts.dt.dayofweek
                    frame[f"{timestamp_col}_week"] = ts.dt.isocalendar().week.astype(int)
                    frame[f"{timestamp_col}_dayofyear"] = ts.dt.dayofyear
                    frame[f"{timestamp_col}_sin_hour"] = np.sin(2 * np.pi * frame[f"{timestamp_col}_hour"] / 24.0)
                    frame[f"{timestamp_col}_cos_hour"] = np.cos(2 * np.pi * frame[f"{timestamp_col}_hour"] / 24.0)
                    frame[f"{timestamp_col}_sin_dayofyear"] = np.sin(2 * np.pi * frame[f"{timestamp_col}_dayofyear"] / 365.25)
                    frame[f"{timestamp_col}_cos_dayofyear"] = np.cos(2 * np.pi * frame[f"{timestamp_col}_dayofyear"] / 365.25)
                    return frame


                def train_val_test_split_by_time(df: pd.DataFrame, time_col: str, train_end: str, val_end: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
                    ts = pd.to_datetime(df[time_col])
                    train_mask = ts < pd.to_datetime(train_end)
                    val_mask = (ts >= pd.to_datetime(train_end)) & (ts < pd.to_datetime(val_end))
                    test_mask = ts >= pd.to_datetime(val_end)
                    return df[train_mask].copy(), df[val_mask].copy(), df[test_mask].copy()
                """,
            ),
            (
                "code",
                """
                EAF_TABLES = [
                    "eaf_temp",
                    "eaf_gaslance_mat",
                    "inj_mat",
                    "eaf_transformer",
                    "eaf_added_materials",
                    "basket_charged",
                    "lf_added_materials",
                    "lf_initial_chemical_measurements",
                    "eaf_final_chemical_measurements",
                    "ladle_tapping",
                ]


                def parse_duration_minutes(value: Any) -> Optional[float]:
                    if pd.isna(value):
                        return np.nan
                    s = str(value).strip()
                    if not s:
                        return np.nan
                    s = s.replace(" ", "")
                    if ":" not in s:
                        return coerce_decimal(pd.Series([s])).iloc[0]
                    parts = s.split(":")
                    try:
                        hours = float(parts[0])
                        minutes = float(parts[1])
                        return hours * 60.0 + minutes
                    except Exception:
                        return np.nan
                """,
            ),
            (
                "code",
                """
                def load_eaf_tables(data_root: Path) -> Dict[str, pd.DataFrame]:
                    base = data_root / "Industrial Data from the Electric Arc Furnace"
                    if not base.exists():
                        raise FileNotFoundError(f"EAF directory not found at {base}")
                    tables: Dict[str, pd.DataFrame] = {}
                    for name in EAF_TABLES:
                        path = base / f"{name}.csv"
                        if not path.exists():
                            print(f"[WARN] Missing table {path}")
                            continue
                        print(f"Loading {name}...")
                        if name in {"eaf_gaslance_mat", "inj_mat"}:
                            df = pd.read_csv(path, dtype=str)
                            df = ensure_datetime(df, ["REVTIME"])
                            numeric_cols = [c for c in df.columns if c not in ("REVTIME", "HEATID")]
                            df = ensure_float(df, numeric_cols)
                        elif name == "eaf_temp":
                            df = pd.read_csv(path)
                            df = ensure_datetime(df, ["DATETIME"])
                            numeric_cols = [c for c in df.columns if c not in ("HEATID", "DATETIME")]
                            df = ensure_float(df, numeric_cols)
                        elif name == "eaf_transformer":
                            df = pd.read_csv(path, dtype=str)
                            df = ensure_datetime(df, ["STARTTIME"])
                            df["DURATION_MIN"] = df["DURATION"].astype(str).str.replace(" ", "")
                            df["DURATION_MIN"] = df["DURATION_MIN"].apply(parse_duration_minutes)
                            df = ensure_float(df, ["DURATION_MIN", "MW"])
                        else:
                            df = pd.read_csv(path, dtype=str)
                            datetime_cols = [c for c in df.columns if "DATE" in c.upper() or "TIME" in c.upper()]
                            if datetime_cols:
                                df = ensure_datetime(df, datetime_cols)
                            numeric_cols = [c for c in df.columns if c not in datetime_cols and c not in ("HEATID", "RECID", "POSITIONROW")]
                            df = ensure_float(df, numeric_cols)
                        tables[name] = df
                    return tables


                def compute_heatwise_aggregates(df: pd.DataFrame, heat_col: str, aggregations: Dict[str, List[str]]) -> pd.DataFrame:
                    grouped = df.groupby(heat_col).agg(aggregations)
                    grouped.columns = [f"{col}_{agg}" for col, agg in grouped.columns]
                    grouped = grouped.reset_index()
                    return grouped


                def merge_asof_multikey(base_df: pd.DataFrame, lookup_df: pd.DataFrame, on: str, by: str, suffix: str, tolerance: pd.Timedelta) -> pd.DataFrame:
                    if lookup_df is None or lookup_df.empty:
                        return base_df
                    merge_cols = [c for c in lookup_df.columns if c not in (on, by)]
                    ordered_lookup = lookup_df.sort_values([by, on])
                    merged = pd.merge_asof(
                        base_df.sort_values([by, on]),
                        ordered_lookup[[by, on] + merge_cols],
                        left_on=on,
                        right_on=on,
                        by=by,
                        direction="backward",
                        tolerance=tolerance,
                        allow_exact_matches=True,
                    )
                    merged = merged.sort_index()
                    rename_map = {col: f"{col}_{suffix}" for col in merge_cols}
                    merged = merged.rename(columns=rename_map)
                    return merged
                """,
            ),
            (
                "code",
                """
                def prepare_eaf_temp_and_o2_bundles(
                    tables: Dict[str, pd.DataFrame],
                    history_lags: List[int] = (1, 2, 3, 6),
                    horizon: int = 1,
                ) -> Tuple[DatasetBundle, DatasetBundle]:
                    temp = tables["eaf_temp"].copy()
                    temp["DATETIME"] = pd.to_datetime(temp["DATETIME"])
                    temp = temp.sort_values(["HEATID", "DATETIME"]).reset_index(drop=True)
                    temp = temp.drop_duplicates(subset=["HEATID", "DATETIME"], keep="last")

                    for lag in history_lags:
                        temp[f"TEMP_lag_{lag}"] = temp.groupby("HEATID")["TEMP"].shift(lag)
                        temp[f"VALO2_lag_{lag}"] = temp.groupby("HEATID")["VALO2_PPM"].shift(lag)

                    temp["TEMP_target"] = temp.groupby("HEATID")["TEMP"].shift(-horizon)
                    temp["VALO2_target"] = temp.groupby("HEATID")["VALO2_PPM"].shift(-horizon)

                    temp["HEAT_START"] = temp.groupby("HEATID")["DATETIME"].transform("min")
                    temp["minutes_from_heat_start"] = (temp["DATETIME"] - temp["HEAT_START"]).dt.total_seconds() / 60.0
                    temp["sample_index"] = temp.groupby("HEATID").cumcount()
                    temp["minutes_between_samples"] = temp.groupby("HEATID")["DATETIME"].diff().dt.total_seconds().fillna(0.0) / 60.0

                    gas = tables.get("eaf_gaslance_mat")
                    if gas is not None and not gas.empty:
                        gas = gas.sort_values(["HEATID", "REVTIME"])
                        for col in ["O2_AMOUNT", "GAS_AMOUNT", "O2_FLOW", "GAS_FLOW"]:
                            if col in gas.columns:
                                gas[f"{col}_cum"] = gas.groupby("HEATID")[col].cumsum()
                        temp = merge_asof_multikey(
                            temp,
                            gas,
                            on="DATETIME",
                            by="HEATID",
                            suffix="gas",
                            tolerance=pd.Timedelta(minutes=30),
                        )

                    inj = tables.get("inj_mat")
                    if inj is not None and not inj.empty:
                        inj = inj.sort_values(["HEATID", "REVTIME"])
                        for col in ["INJ_AMOUNT_CARBON", "INJ_FLOW_CARBON"]:
                            if col in inj.columns:
                                inj[f"{col}_cum"] = inj.groupby("HEATID")[col].cumsum()
                        temp = merge_asof_multikey(
                            temp,
                            inj,
                            on="DATETIME",
                            by="HEATID",
                            suffix="inj",
                            tolerance=pd.Timedelta(minutes=30),
                        )

                    transformer = tables.get("eaf_transformer")
                    if transformer is not None and not transformer.empty:
                        transformer = transformer.sort_values(["HEATID", "STARTTIME"])
                        temp = merge_asof_multikey(
                            temp,
                            transformer,
                            on="DATETIME",
                            by="HEATID",
                            suffix="xfmr",
                            tolerance=pd.Timedelta(hours=2),
                        )

                    temp = add_calendar_features(temp, "DATETIME")
                    feature_cols = [
                        col
                        for col in temp.columns
                        if col
                        not in {
                            "TEMP",
                            "VALO2_PPM",
                            "TEMP_target",
                            "VALO2_target",
                            "HEATID",
                            "HEAT_START",
                            "DATETIME",
                        }
                        and not col.endswith("_xfmr")
                    ]
                    feature_cols = [c for c in feature_cols if temp[c].dtype != "O"]

                    temp = temp.dropna(subset=feature_cols + ["TEMP_target", "VALO2_target"]).reset_index(drop=True)

                    temp["year"] = temp["DATETIME"].dt.year
                    heat_year = temp.groupby("HEATID")["year"].max().reset_index().rename(columns={"year": "heat_year"})
                    temp = temp.merge(heat_year, on="HEATID", how="left")

                    train_mask = temp["heat_year"] <= 2016
                    val_mask = temp["heat_year"] == 2017
                    test_mask = temp["heat_year"] >= 2018

                    def build_split(mask: pd.Series) -> Dict[str, np.ndarray]:
                        X = temp.loc[mask, feature_cols].to_numpy(dtype=np.float32)
                        y_temp = temp.loc[mask, "TEMP_target"].to_numpy(dtype=np.float32)[:, None]
                        y_o2 = temp.loc[mask, "VALO2_target"].to_numpy(dtype=np.float32)[:, None]
                        return {"X": X, "y_temp": y_temp, "y_o2": y_o2}

                    train_split = build_split(train_mask)
                    val_split = build_split(val_mask)
                    test_split = build_split(test_mask)

                    temp_bundle = DatasetBundle(
                        name="EAF_TEMP_forecast",
                        task_type="regression",
                        input_kind="tabular",
                        feature_names=feature_cols,
                        target_names=["TEMP_target"],
                        train={"X": train_split["X"], "y": train_split["y_temp"]},
                        val={"X": val_split["X"], "y": val_split["y_temp"]},
                        test={"X": test_split["X"], "y": test_split["y_temp"]},
                        metadata={
                            "horizon_steps": horizon,
                            "history_lags": list(history_lags),
                            "feature_source": "temp + gas + injection + calendar",
                        },
                    )

                    o2_bundle = DatasetBundle(
                        name="EAF_VALO2_forecast",
                        task_type="regression",
                        input_kind="tabular",
                        feature_names=feature_cols,
                        target_names=["VALO2_target"],
                        train={"X": train_split["X"], "y": train_split["y_o2"]},
                        val={"X": val_split["X"], "y": val_split["y_o2"]},
                        test={"X": test_split["X"], "y": test_split["y_o2"]},
                        metadata={
                            "horizon_steps": horizon,
                            "history_lags": list(history_lags),
                            "feature_source": "temp + gas + injection + calendar",
                        },
                    )

                    return temp_bundle, o2_bundle
                """,
            ),
            (
                "code",
                """
                def prepare_eaf_chemistry_bundle(tables: Dict[str, pd.DataFrame]) -> DatasetBundle:
                    chem = tables["eaf_final_chemical_measurements"].copy()
                    chem = ensure_datetime(chem, ["DATETIME"])
                    chem = chem.sort_values(["HEATID", "DATETIME"])
                    chem = chem.drop_duplicates(subset=["HEATID"], keep="last")

                    target_cols = [c for c in chem.columns if c not in ("HEATID", "POSITIONROW", "DATETIME")]
                    chem = ensure_float(chem, target_cols)

                    temp = tables["eaf_temp"].copy()
                    temp = ensure_datetime(temp, ["DATETIME"])
                    temp = temp.sort_values(["HEATID", "DATETIME"])
                    temp["sample_index"] = temp.groupby("HEATID").cumcount()
                    temp = add_calendar_features(temp, "DATETIME")
                    temp_aggs = compute_heatwise_aggregates(
                        temp,
                        "HEATID",
                        {
                            "TEMP": ["mean", "max", "min", "last"],
                            "VALO2_PPM": ["mean", "max", "last"],
                            "DATETIME_month": ["last"],
                            "DATETIME_hour": ["mean"],
                            "sample_index": ["max"],
                        },
                    )

                    gas = tables.get("eaf_gaslance_mat")
                    if gas is not None:
                        gas_aggs = compute_heatwise_aggregates(
                            gas,
                            "HEATID",
                            {
                                "O2_AMOUNT": ["max"],
                                "GAS_AMOUNT": ["max"],
                                "O2_FLOW": ["mean", "max"],
                                "GAS_FLOW": ["mean", "max"],
                            },
                        )
                    else:
                        gas_aggs = pd.DataFrame(columns=["HEATID"])

                    inj = tables.get("inj_mat")
                    if inj is not None:
                        inj_aggs = compute_heatwise_aggregates(
                            inj,
                            "HEATID",
                            {
                                "INJ_AMOUNT_CARBON": ["max"],
                                "INJ_FLOW_CARBON": ["mean", "max"],
                            },
                        )
                    else:
                        inj_aggs = pd.DataFrame(columns=["HEATID"])

                    transformer = tables.get("eaf_transformer")
                    if transformer is not None:
                        transformer_aggs = compute_heatwise_aggregates(
                            transformer,
                            "HEATID",
                            {
                                "MW": ["mean", "max"],
                                "DURATION_MIN": ["sum"],
                            },
                        )
                    else:
                        transformer_aggs = pd.DataFrame(columns=["HEATID"])

                    features = chem[["HEATID", "DATETIME"]].merge(temp_aggs, on="HEATID", how="left")
                    features = features.merge(gas_aggs, on="HEATID", how="left")
                    features = features.merge(inj_aggs, on="HEATID", how="left")
                    features = features.merge(transformer_aggs, on="HEATID", how="left")

                    features = ensure_float(features, [c for c in features.columns if c not in ("HEATID", "DATETIME")])
                    features = add_calendar_features(features, "DATETIME")
                    feature_cols = [c for c in features.columns if c not in ("HEATID", "DATETIME")]
                    features = features.dropna(subset=feature_cols).reset_index(drop=True)

                    merged = features.merge(chem[["HEATID"] + target_cols], on="HEATID", how="inner")
                    merged = merged.dropna().reset_index(drop=True)

                    merged["year"] = pd.to_datetime(merged["DATETIME"]).dt.year
                    train_mask = merged["year"] <= 2016
                    val_mask = merged["year"] == 2017
                    test_mask = merged["year"] >= 2018

                    X_train = merged.loc[train_mask, feature_cols].to_numpy(dtype=np.float32)
                    y_train = merged.loc[train_mask, target_cols].to_numpy(dtype=np.float32)
                    X_val = merged.loc[val_mask, feature_cols].to_numpy(dtype=np.float32)
                    y_val = merged.loc[val_mask, target_cols].to_numpy(dtype=np.float32)
                    X_test = merged.loc[test_mask, feature_cols].to_numpy(dtype=np.float32)
                    y_test = merged.loc[test_mask, target_cols].to_numpy(dtype=np.float32)

                    bundle = DatasetBundle(
                        name="EAF_chemistry",
                        task_type="regression",
                        input_kind="tabular",
                        feature_names=feature_cols,
                        target_names=target_cols,
                        train={"X": X_train, "y": y_train},
                        val={"X": X_val, "y": y_val},
                        test={"X": X_test, "y": y_test},
                        metadata={
                            "target_dim": len(target_cols),
                            "note": "heat-level aggregates for final composition",
                        },
                    )
                    return bundle
                """,
            ),
        ]
    )
    cells_data.extend(
        [
            (
                "code",
                """
                from psann.nn import ResidualPSANNNet


                class IdentitySpine(nn.Module):
                    def forward(self, x: torch.Tensor) -> torch.Tensor:
                        if x.ndim == 3:
                            return x.reshape(x.size(0), -1)
                        return x


                class TemporalConvSpine(nn.Module):
                    def __init__(
                        self,
                        input_channels: int,
                        hidden_channels: int,
                        kernel_size: int = 3,
                        stride: int = 2,
                        depth: int = 2,
                        activation: Callable[[], nn.Module] = nn.GELU,
                    ):
                        super().__init__()
                        layers: List[nn.Module] = []
                        channels = input_channels
                        for _ in range(depth):
                            layers.append(
                                nn.Conv1d(
                                    channels,
                                    hidden_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=kernel_size // 2,
                                )
                            )
                            layers.append(nn.BatchNorm1d(hidden_channels))
                            layers.append(activation())
                            channels = hidden_channels
                        self.net = nn.Sequential(*layers)
                        self.pool = nn.AdaptiveAvgPool1d(1)

                    def forward(self, x: torch.Tensor) -> torch.Tensor:
                        z = x.transpose(1, 2)
                        z = self.net(z)
                        z = self.pool(z).squeeze(-1)
                        return z


                class TemporalAttentionSpine(nn.Module):
                    def __init__(self, input_dim: int, num_heads: int = 1, ff_factor: int = 2, dropout: float = 0.1):
                        super().__init__()
                        self.norm = nn.LayerNorm(input_dim)
                        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
                        self.ff = nn.Sequential(
                            nn.LayerNorm(input_dim),
                            nn.Linear(input_dim, ff_factor * input_dim),
                            nn.GELU(),
                            nn.Linear(ff_factor * input_dim, input_dim),
                        )

                    def forward(self, x: torch.Tensor) -> torch.Tensor:
                        z = self.norm(x)
                        attn_out, _ = self.attn(z, z, z)
                        z = z + attn_out
                        z = z + self.ff(z)
                        return z.mean(dim=1)


                class FlattenSpine(nn.Module):
                    def forward(self, x: torch.Tensor) -> torch.Tensor:
                        if x.ndim == 3:
                            return x.reshape(x.size(0), -1)
                        return x


                class SequencePSANNModel(nn.Module):
                    def __init__(
                        self,
                        input_shape: Tuple[int, ...],
                        output_dim: int,
                        *,
                        hidden_layers: int,
                        hidden_units: int,
                        spine_type: str = "flatten",
                        spine_params: Optional[Dict[str, Any]] = None,
                        activation_type: str = "psann",
                    ):
                        super().__init__()
                        spine_params = spine_params or {}
                        time_steps, channels = input_shape
                        if spine_type == "conv":
                            self.spine = TemporalConvSpine(
                                channels,
                                spine_params.get("channels", hidden_units),
                                kernel_size=spine_params.get("kernel_size", 5),
                                stride=spine_params.get("stride", 2),
                                depth=spine_params.get("depth", 2),
                            )
                            psann_input_dim = spine_params.get("channels", hidden_units)
                        elif spine_type == "attention":
                            self.spine = TemporalAttentionSpine(
                                input_dim=channels,
                                num_heads=spine_params.get("num_heads", 1),
                                ff_factor=spine_params.get("ff_factor", 2),
                                dropout=spine_params.get("dropout", 0.1),
                            )
                            psann_input_dim = channels
                        elif spine_type == "flatten":
                            self.spine = FlattenSpine()
                            psann_input_dim = time_steps * channels
                        else:
                            self.spine = IdentitySpine()
                            psann_input_dim = time_steps * channels
                        self.core = ResidualPSANNNet(
                            psann_input_dim,
                            output_dim,
                            hidden_layers=hidden_layers,
                            hidden_units=hidden_units,
                            activation_type=activation_type,
                        )

                    def forward(self, x: torch.Tensor) -> torch.Tensor:
                        if x.ndim == 3:
                            z = self.spine(x)
                        else:
                            z = x
                        return self.core(z)


                class TabularPSANNModel(nn.Module):
                    def __init__(
                        self,
                        input_dim: int,
                        output_dim: int,
                        *,
                        hidden_layers: int,
                        hidden_units: int,
                        activation_type: str = "psann",
                    ):
                        super().__init__()
                        self.core = ResidualPSANNNet(
                            input_dim,
                            output_dim,
                            hidden_layers=hidden_layers,
                            hidden_units=hidden_units,
                            activation_type=activation_type,
                        )

                    def forward(self, x: torch.Tensor) -> torch.Tensor:
                        if x.ndim > 2:
                            x = x.reshape(x.size(0), -1)
                        return self.core(x)


                class MLPModel(nn.Module):
                    def __init__(self, input_dim: int, output_dim: int, hidden_layers: int = 3, hidden_units: int = 256, dropout: float = 0.1):
                        super().__init__()
                        layers: List[nn.Module] = []
                        in_dim = input_dim
                        for _ in range(hidden_layers):
                            layers.append(nn.Linear(in_dim, hidden_units))
                            layers.append(nn.ReLU())
                            layers.append(nn.Dropout(dropout))
                            in_dim = hidden_units
                        layers.append(nn.Linear(in_dim, output_dim))
                        self.net = nn.Sequential(*layers)

                    def forward(self, x: torch.Tensor) -> torch.Tensor:
                        if x.ndim > 2:
                            x = x.reshape(x.size(0), -1)
                        return self.net(x)


                class LSTMHead(nn.Module):
                    def __init__(self, input_dim: int, hidden_units: int, num_layers: int, output_dim: int, bidirectional: bool = False, dropout: float = 0.1):
                        super().__init__()
                        self.lstm = nn.LSTM(
                            input_dim,
                            hidden_units,
                            num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0.0,
                            batch_first=True,
                            bidirectional=bidirectional,
                        )
                        out_dim = hidden_units * (2 if bidirectional else 1)
                        self.head = nn.Sequential(
                            nn.LayerNorm(out_dim),
                            nn.Linear(out_dim, output_dim),
                        )

                    def forward(self, x: torch.Tensor) -> torch.Tensor:
                        if x.ndim == 2:
                            x = x.unsqueeze(1)
                        _, (h_n, _) = self.lstm(x)
                        z = h_n[-1]
                        return self.head(z)


                class TinyTCNBlock(nn.Module):
                    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
                        super().__init__()
                        self.conv = nn.Sequential(
                            nn.Conv1d(channels, channels, kernel_size, padding="same", dilation=dilation),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Conv1d(channels, channels, kernel_size, padding="same", dilation=dilation),
                            nn.GELU(),
                            nn.Dropout(dropout),
                        )

                    def forward(self, x: torch.Tensor) -> torch.Tensor:
                        return x + self.conv(x)


                class TinyTCN(nn.Module):
                    def __init__(self, input_channels: int, output_dim: int, hidden_channels: int = 128, layers: int = 3, kernel_size: int = 3, dropout: float = 0.1):
                        super().__init__()
                        self.pre = nn.Conv1d(input_channels, hidden_channels, kernel_size=1)
                        blocks = []
                        for i in range(layers):
                            blocks.append(TinyTCNBlock(hidden_channels, kernel_size, dilation=2 ** i, dropout=dropout))
                        self.blocks = nn.Sequential(*blocks)
                        self.head = nn.Sequential(
                            nn.AdaptiveAvgPool1d(1),
                            nn.Flatten(),
                            nn.Linear(hidden_channels, output_dim),
                        )

                    def forward(self, x: torch.Tensor) -> torch.Tensor:
                        z = x.transpose(1, 2)
                        z = self.pre(z)
                        z = self.blocks(z)
                        z = self.head(z)
                        return z


                def build_psann_tabular(input_shape: Tuple[int, ...], output_dim: int, extra: Dict[str, Any]) -> nn.Module:
                    hidden_layers = extra.get("hidden_layers", 8)
                    hidden_units = extra.get("hidden_units", 256)
                    activation_type = extra.get("activation_type", "psann")
                    return TabularPSANNModel(
                        input_dim=int(np.prod(input_shape)),
                        output_dim=output_dim,
                        hidden_layers=hidden_layers,
                        hidden_units=hidden_units,
                        activation_type=activation_type,
                    )


                def build_psann_sequence(input_shape: Tuple[int, ...], output_dim: int, extra: Dict[str, Any]) -> nn.Module:
                    hidden_layers = extra.get("hidden_layers", 8)
                    hidden_units = extra.get("hidden_units", 256)
                    spine_type = extra.get("spine_type", "flatten")
                    spine_params = extra.get("spine_params", {})
                    activation_type = extra.get("activation_type", "psann")
                    return SequencePSANNModel(
                        input_shape,
                        output_dim,
                        hidden_layers=hidden_layers,
                        hidden_units=hidden_units,
                        spine_type=spine_type,
                        spine_params=spine_params,
                        activation_type=activation_type,
                    )


                def build_mlp_model(input_shape: Tuple[int, ...], output_dim: int, extra: Dict[str, Any]) -> nn.Module:
                    hidden_layers = extra.get("hidden_layers", 3)
                    hidden_units = extra.get("hidden_units", 256)
                    dropout = extra.get("dropout", 0.1)
                    return MLPModel(
                        input_dim=int(np.prod(input_shape)),
                        output_dim=output_dim,
                        hidden_layers=hidden_layers,
                        hidden_units=hidden_units,
                        dropout=dropout,
                    )


                def build_lstm_model(input_shape: Tuple[int, ...], output_dim: int, extra: Dict[str, Any]) -> nn.Module:
                    sequence_length, channels = input_shape
                    hidden_units = extra.get("hidden_units", 128)
                    num_layers = extra.get("num_layers", 1)
                    bidirectional = extra.get("bidirectional", False)
                    return LSTMHead(
                        input_dim=channels,
                        hidden_units=hidden_units,
                        num_layers=num_layers,
                        output_dim=output_dim,
                        bidirectional=bidirectional,
                        dropout=extra.get("dropout", 0.1),
                    )


                def build_tcn_model(input_shape: Tuple[int, ...], output_dim: int, extra: Dict[str, Any]) -> nn.Module:
                    sequence_length, channels = input_shape
                    hidden_channels = extra.get("hidden_channels", 128)
                    layers = extra.get("layers", 3)
                    kernel_size = extra.get("kernel_size", 3)
                    dropout = extra.get("dropout", 0.1)
                    return TinyTCN(
                        input_channels=channels,
                        output_dim=output_dim,
                        hidden_channels=hidden_channels,
                        layers=layers,
                        kernel_size=kernel_size,
                        dropout=dropout,
                    )
                """,
            ),
        ]
    )
    cells_data.extend(
        [
            (
                "code",
                """
                def count_trainable_parameters(model: nn.Module) -> int:
                    return sum(p.numel() for p in model.parameters() if p.requires_grad)


                def evaluate_model(model: nn.Module, loader: DataLoader, spec: ModelSpec) -> Tuple[np.ndarray, np.ndarray]:
                    model.eval()
                    preds = []
                    truths = []
                    with torch.no_grad():
                        for X_batch, y_batch in loader:
                            X_batch = X_batch.to(DEVICE)
                            y_batch = y_batch.to(DEVICE)
                            outputs = model(X_batch)
                            preds.append(outputs.detach().cpu().numpy())
                            truths.append(y_batch.detach().cpu().numpy())
                    y_pred = np.concatenate(preds, axis=0)
                    y_true = np.concatenate(truths, axis=0)
                    return y_true, y_pred


                def train_model_on_bundle(bundle: DatasetBundle, spec: ModelSpec, task_name: str) -> Dict[str, Any]:
                    input_shape = bundle.train["X"].shape[1:]
                    if spec.task_type == "classification":
                        output_dim = int(bundle.metadata.get("n_classes", np.unique(bundle.train["y"]).size))
                    else:
                        output_dim = bundle.train["y"].shape[1] if bundle.train["y"].ndim > 1 else 1

                    model = spec.builder(input_shape, output_dim, spec.extra)
                    model.to(DEVICE)
                    params = count_trainable_parameters(model)
                    optimizer_cls = torch.optim.AdamW if spec.train_config.weight_decay > 0 else torch.optim.Adam
                    optimizer = optimizer_cls(
                        model.parameters(),
                        lr=spec.train_config.learning_rate,
                        weight_decay=spec.train_config.weight_decay,
                    )
                    scheduler = None
                    if spec.train_config.scheduler == "cosine":
                        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=spec.train_config.epochs)

                    train_loader = build_dataloader(
                        bundle.train["X"],
                        bundle.train["y"],
                        spec.train_config.batch_size,
                        shuffle=True,
                        task_type=spec.task_type,
                    )
                    val_loader = build_dataloader(
                        bundle.val["X"],
                        bundle.val["y"],
                        spec.train_config.batch_size,
                        shuffle=False,
                        task_type=spec.task_type,
                    )
                    test_loader = build_dataloader(
                        bundle.test["X"],
                        bundle.test["y"],
                        spec.train_config.batch_size,
                        shuffle=False,
                        task_type=spec.task_type,
                    )

                    best_state = None
                    best_val_metric = -float("inf")
                    patience_counter = spec.train_config.patience
                    history = []
                    criterion_reg = nn.MSELoss()

                    with Timer() as timer:
                        for epoch in range(spec.train_config.epochs):
                            model.train()
                            running_loss = 0.0
                            batches = 0
                            for step, (X_batch, y_batch) in enumerate(train_loader, start=1):
                                X_batch = X_batch.to(DEVICE)
                                y_batch = y_batch.to(DEVICE)
                                optimizer.zero_grad()
                                outputs = model(X_batch)
                                if spec.task_type == "classification":
                                    loss = nn.functional.cross_entropy(
                                        outputs,
                                        y_batch,
                                        label_smoothing=GLOBAL_CONFIG["label_smoothing"],
                                    )
                                else:
                                    target = y_batch
                                    if target.ndim == 1:
                                        target = target.unsqueeze(-1)
                                    loss = criterion_reg(outputs, target)
                                loss.backward()
                                if spec.train_config.gradient_clip is not None:
                                    nn.utils.clip_grad_norm_(model.parameters(), spec.train_config.gradient_clip)
                                optimizer.step()
                                running_loss += loss.item()
                                batches += 1
                                if spec.train_config.max_batches_per_epoch and batches >= spec.train_config.max_batches_per_epoch:
                                    break
                            if scheduler is not None:
                                scheduler.step()
                            avg_loss = running_loss / max(1, batches)
                            val_true, val_pred = evaluate_model(model, val_loader, spec)
                            if spec.task_type == "classification":
                                metrics = classification_metrics(val_true, val_pred)
                                score = metrics["accuracy"]
                            else:
                                metrics = regression_metrics(val_true.squeeze(), val_pred.squeeze())
                                score = -metrics["rmse"]
                            history.append({"epoch": epoch + 1, "train_loss": avg_loss, "val_score": score})
                            if score > best_val_metric:
                                best_val_metric = score
                                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                                patience_counter = spec.train_config.patience
                            else:
                                patience_counter -= 1
                            if spec.train_config.early_stopping and patience_counter <= 0:
                                break
                            if spec.train_config.max_minutes is not None and timer.elapsed / 60.0 > spec.train_config.max_minutes:
                                print(f"[INFO] Time budget reached for {spec.name}; stopping at epoch {epoch + 1}.")
                                break

                    if best_state is not None:
                        model.load_state_dict(best_state)

                    train_true, train_pred = evaluate_model(model, train_loader, spec)
                    val_true, val_pred = evaluate_model(model, val_loader, spec)
                    test_true, test_pred = evaluate_model(model, test_loader, spec)

                    if spec.task_type == "classification":
                        train_metrics = classification_metrics(train_true, train_pred)
                        val_metrics = classification_metrics(val_true, val_pred)
                        test_metrics = classification_metrics(test_true, test_pred)
                    else:
                        train_metrics = regression_metrics(train_true.squeeze(), train_pred.squeeze())
                        val_metrics = regression_metrics(val_true.squeeze(), val_pred.squeeze())
                        test_metrics = regression_metrics(test_true.squeeze(), test_pred.squeeze())

                    RESULT_LOGGER.append(
                        ExperimentResult(
                            dataset=bundle.name,
                            task=task_name,
                            model=spec.name,
                            group=spec.group,
                            split="train",
                            params=params,
                            train_wall_seconds=timer.elapsed,
                            metrics=train_metrics,
                            notes=spec.notes,
                        )
                    )
                    RESULT_LOGGER.append(
                        ExperimentResult(
                            dataset=bundle.name,
                            task=task_name,
                            model=spec.name,
                            group=spec.group,
                            split="val",
                            params=params,
                            train_wall_seconds=timer.elapsed,
                            metrics=val_metrics,
                            notes=spec.notes,
                        )
                    )
                    RESULT_LOGGER.append(
                        ExperimentResult(
                            dataset=bundle.name,
                            task=task_name,
                            model=spec.name,
                            group=spec.group,
                            split="test",
                            params=params,
                            train_wall_seconds=timer.elapsed,
                            metrics=test_metrics,
                            notes=spec.notes,
                        )
                    )

                    model_cpu = model.to("cpu")

                    return {
                        "model": model_cpu,
                        "train_metrics": train_metrics,
                        "val_metrics": val_metrics,
                        "test_metrics": test_metrics,
                        "train_true": train_true,
                        "train_pred": train_pred,
                        "val_true": val_true,
                        "val_pred": val_pred,
                        "test_true": test_true,
                        "test_pred": test_pred,
                        "history": history,
                        "params": params,
                        "train_time": timer.elapsed,
                    }
                """,
            ),
        ]
    )
    cells_data.extend(
        [
            (
                "code",
                """
                def permutation_importance(
                    model: nn.Module,
                    bundle: DatasetBundle,
                    spec: ModelSpec,
                    feature_groups: Dict[str, List[int]],
                    split: str = "test",
                    n_repeats: int = 5,
                ) -> pd.DataFrame:
                    data = getattr(bundle, split)
                    baseline_loader = build_dataloader(
                        data["X"],
                        data["y"],
                        spec.train_config.batch_size,
                        shuffle=False,
                        task_type=spec.task_type,
                    )
                    y_true, y_pred = evaluate_model(model, baseline_loader, spec)
                    if spec.task_type == "classification":
                        baseline_metric = classification_metrics(y_true, y_pred)["accuracy"]
                    else:
                        baseline_metric = regression_metrics(y_true.squeeze(), y_pred.squeeze())["rmse"]

                    rows = []
                    for group_name, columns in feature_groups.items():
                        deltas = []
                        cols = np.atleast_1d(columns)
                        for _ in range(n_repeats):
                            X_perm = data["X"].copy()
                            if bundle.input_kind == "tabular":
                                for col in cols:
                                    np.random.shuffle(X_perm[:, col])
                            else:
                                for col in cols:
                                    np.random.shuffle(X_perm[:, :, col])
                            loader = build_dataloader(
                                X_perm,
                                data["y"],
                                spec.train_config.batch_size,
                                shuffle=False,
                                task_type=spec.task_type,
                            )
                            y_true_perm, y_pred_perm = evaluate_model(model, loader, spec)
                            if spec.task_type == "classification":
                                metric_value = classification_metrics(y_true_perm, y_pred_perm)["accuracy"]
                                delta = baseline_metric - metric_value
                            else:
                                metric_value = regression_metrics(y_true_perm.squeeze(), y_pred_perm.squeeze())["rmse"]
                                delta = metric_value - baseline_metric
                            deltas.append(delta)
                        rows.append(
                            {
                                "group": group_name,
                                "mean_delta": float(np.mean(deltas)),
                                "std_delta": float(np.std(deltas)),
                                "baseline": baseline_metric,
                            }
                        )
                    return pd.DataFrame(rows)


                def compute_shap_importance(
                    model: nn.Module,
                    bundle: DatasetBundle,
                    spec: ModelSpec,
                    split: str = "val",
                    sample_size: int = 512,
                ) -> Dict[str, Any]:
                    import shap

                    data = getattr(bundle, split)
                    X = data["X"]
                    if len(X) == 0:
                        raise ValueError(f"No samples available in {split} split for SHAP computation.")
                    sample_size = min(sample_size, len(X))
                    idx = np.random.choice(len(X), size=sample_size, replace=False)
                    X_sample = X[idx]

                    model_cpu = model.to("cpu").eval()

                    def predict_fn(batch: np.ndarray) -> np.ndarray:
                        with torch.no_grad():
                            inputs = torch.from_numpy(batch).float()
                            outputs = model_cpu(inputs)
                            if spec.task_type == "classification":
                                return torch.softmax(outputs, dim=-1).numpy()
                            return outputs.numpy()

                    if bundle.input_kind == "tabular":
                        background = X_sample[: min(128, sample_size)]
                        explainer = shap.KernelExplainer(predict_fn, background)
                        shap_values = explainer.shap_values(X_sample)
                    else:
                        background = torch.from_numpy(X_sample[: min(64, sample_size)]).float()
                        explainer = shap.DeepExplainer(model_cpu, background)
                        shap_values = explainer.shap_values(torch.from_numpy(X_sample).float())

                    model.to(DEVICE)
                    return {"explainer": explainer, "shap_values": shap_values, "sample_indices": idx}


                def compute_jacobian_singular_values(model: nn.Module, inputs: torch.Tensor, max_samples: int = 128) -> np.ndarray:
                    model.eval()
                    inputs = inputs[:max_samples].to(DEVICE).requires_grad_(True)
                    outputs = model(inputs)
                    if outputs.ndim == 1:
                        outputs = outputs.unsqueeze(-1)
                    jacobian_rows = []
                    for i in range(outputs.shape[1]):
                        grad_outputs = torch.zeros_like(outputs)
                        grad_outputs[:, i] = 1.0
                        grads = torch.autograd.grad(outputs, inputs, grad_outputs=grad_outputs, retain_graph=True, create_graph=False)[0]
                        jacobian_rows.append(grads.reshape(grads.size(0), -1).detach().cpu().numpy())
                    jacobian = np.concatenate(jacobian_rows, axis=1)
                    sigma = np.linalg.svd(jacobian, compute_uv=False)
                    return sigma


                def participation_ratio(singular_values: np.ndarray) -> float:
                    if singular_values.size == 0:
                        return float("nan")
                    numerator = (singular_values ** 2).sum() ** 2
                    denominator = (singular_values ** 4).sum() + 1e-8
                    return float(numerator / denominator)


                def frequency_response_probe(model: nn.Module, input_dim: int, frequencies: Iterable[float], amplitude: float = 1.0) -> pd.DataFrame:
                    model.eval()
                    rows = []
                    times = torch.linspace(0, 2 * math.pi, steps=512).unsqueeze(0)
                    for freq in frequencies:
                        signal = amplitude * torch.sin(freq * times)
                        if input_dim > 1:
                            signal = signal.repeat(1, input_dim)
                        signal = signal.to(DEVICE).float()
                        with torch.no_grad():
                            output = model(signal)
                        energy = output.pow(2).mean().sqrt().item()
                        rows.append({"frequency": freq, "output_rms": energy})
                    return pd.DataFrame(rows)


                def evaluate_robustness(
                    model: nn.Module,
                    bundle: DatasetBundle,
                    spec: ModelSpec,
                    corruption_fn: Callable[[np.ndarray, float], np.ndarray],
                    split: str = "test",
                    levels: Iterable[float] = (0.0, 0.1, 0.2, 0.3),
                ) -> pd.DataFrame:
                    rows = []
                    base_data = getattr(bundle, split)
                    for level in levels:
                        X_corrupted = corruption_fn(base_data["X"], level)
                        loader = build_dataloader(
                            X_corrupted,
                            base_data["y"],
                            spec.train_config.batch_size,
                            shuffle=False,
                            task_type=spec.task_type,
                        )
                        y_true, y_pred = evaluate_model(model, loader, spec)
                        if spec.task_type == "classification":
                            metrics = classification_metrics(y_true, y_pred)
                        else:
                            metrics = regression_metrics(y_true.squeeze(), y_pred.squeeze())
                        row = {"level": level}
                        row.update(metrics)
                        rows.append(row)
                    return pd.DataFrame(rows)
                """,
            ),
        ]
    )
    cells_data.extend(
        [
            (
                "code",
                """
                DATA_BUNDLES: Dict[str, DatasetBundle] = {}

                print("Loading datasets...")

                eaf_tables = load_eaf_tables(DATA_ROOT)
                eaf_temp_bundle, eaf_o2_bundle = prepare_eaf_temp_and_o2_bundles(eaf_tables)
                eaf_chem_bundle = prepare_eaf_chemistry_bundle(eaf_tables)
                DATA_BUNDLES[eaf_temp_bundle.name] = eaf_temp_bundle
                DATA_BUNDLES[eaf_o2_bundle.name] = eaf_o2_bundle
                DATA_BUNDLES[eaf_chem_bundle.name] = eaf_chem_bundle

                beijing_stations = load_beijing_stations(DATA_ROOT)
                train_stations = [s for s in beijing_stations.keys() if s not in {"Wanshouxigong", "Huairou"}]
                beijing_bundle = assemble_beijing_cross_station_bundle(
                    beijing_stations,
                    train_stations=train_stations,
                    val_station="Wanshouxigong",
                    test_station="Huairou",
                    target="PM2.5",
                    context=24,
                    horizon=6,
                )
                DATA_BUNDLES[beijing_bundle.name] = beijing_bundle

                jena_df = load_jena_climate(DATA_ROOT)
                jena_bundle = prepare_jena_bundle(jena_df, target="T (degC)", context_steps=72, horizon_steps=36)
                DATA_BUNDLES[jena_bundle.name] = jena_bundle

                har_train_df, har_test_df, har_feature_names = load_har_engineered(DATA_ROOT)
                har_engineered_bundle = prepare_har_engineered_bundle(har_train_df, har_test_df, har_feature_names)
                DATA_BUNDLES[har_engineered_bundle.name] = har_engineered_bundle

                X_har_train_raw, y_har_train_raw, X_har_test_raw, y_har_test_raw, har_axes = load_har_raw_sequences(DATA_ROOT)
                har_raw_bundle = prepare_har_raw_bundle(X_har_train_raw, y_har_train_raw, X_har_test_raw, y_har_test_raw)
                DATA_BUNDLES[har_raw_bundle.name] = har_raw_bundle

                ross_train, ross_test, ross_store = load_rossmann_frames(DATA_ROOT)
                ross_prepared, ross_features, ross_target = preprocess_rossmann(ross_train, ross_store)
                ross_bundle = prepare_rossmann_bundle(ross_prepared, ross_features, ross_target)
                DATA_BUNDLES[ross_bundle.name] = ross_bundle

                print("Available dataset bundles:")
                for name, bundle in DATA_BUNDLES.items():
                    print(f" - {name}: {bundle.summary()}")
                """,
            ),
        ]
    )
    cells_data.extend(
        [
            (
                "code",
                """
                EXPERIMENT_REGISTRY: Dict[str, List[ModelSpec]] = {}

                common_regression_train = TrainConfig(
                    epochs=60,
                    batch_size=512,
                    learning_rate=1e-3,
                    weight_decay=1e-4,
                    patience=10,
                    max_minutes=GLOBAL_CONFIG["max_time_minutes"],
                    gradient_clip=1.0,
                )

                common_sequence_train = TrainConfig(
                    epochs=50,
                    batch_size=256,
                    learning_rate=1e-3,
                    weight_decay=1e-4,
                    patience=8,
                    max_minutes=GLOBAL_CONFIG["max_time_minutes"],
                    gradient_clip=1.0,
                )

                common_classification_train = TrainConfig(
                    epochs=50,
                    batch_size=256,
                    learning_rate=5e-4,
                    weight_decay=5e-5,
                    patience=8,
                    max_minutes=GLOBAL_CONFIG["max_time_minutes"],
                    gradient_clip=1.0,
                )


                def register_specs(bundle: DatasetBundle):
                    specs: List[ModelSpec] = []
                    if bundle.input_kind == "tabular":
                        train_cfg = common_regression_train if bundle.task_type == "regression" else common_classification_train
                        specs.append(
                            ModelSpec(
                                name="ResPSANN_tabular",
                                builder=build_psann_tabular,
                                train_config=train_cfg,
                                task_type=bundle.task_type,
                                input_kind="tabular",
                                group="psann",
                                extra={"hidden_layers": 8, "hidden_units": 256},
                                notes="Residual PSANN core",
                            )
                        )
                        specs.append(
                            ModelSpec(
                                name="MLP_baseline",
                                builder=build_mlp_model,
                                train_config=train_cfg,
                                task_type=bundle.task_type,
                                input_kind="tabular",
                                group="baseline",
                                extra={"hidden_layers": 4, "hidden_units": 256, "dropout": 0.1},
                                notes="ReLU MLP with similar parameter budget",
                            )
                        )
                    else:
                        train_cfg = common_sequence_train if bundle.task_type == "regression" else common_classification_train
                        specs.append(
                            ModelSpec(
                                name="ResPSANN_conv_spine",
                                builder=build_psann_sequence,
                                train_config=train_cfg,
                                task_type=bundle.task_type,
                                input_kind="sequence",
                                group="psann",
                                extra={
                                    "hidden_layers": 6,
                                    "hidden_units": 192,
                                    "spine_type": "conv",
                                    "spine_params": {"channels": 192, "depth": 2, "kernel_size": 5, "stride": 2},
                                },
                                notes="ResPSANN with strided Conv1d spine",
                            )
                        )
                        specs.append(
                            ModelSpec(
                                name="ResPSANN_attention_spine",
                                builder=build_psann_sequence,
                                train_config=train_cfg,
                                task_type=bundle.task_type,
                                input_kind="sequence",
                                group="psann",
                                extra={
                                    "hidden_layers": 6,
                                    "hidden_units": 192,
                                    "spine_type": "attention",
                                    "spine_params": {"num_heads": 1},
                                },
                                notes="ResPSANN with single-head attention spine",
                            )
                        )
                        specs.append(
                            ModelSpec(
                                name="LSTM_baseline",
                                builder=build_lstm_model,
                                train_config=train_cfg,
                                task_type=bundle.task_type,
                                input_kind="sequence",
                                group="baseline",
                                extra={"hidden_units": 192, "num_layers": 1, "dropout": 0.1},
                                notes="Single-layer LSTM baseline",
                            )
                        )
                        specs.append(
                            ModelSpec(
                                name="TCN_baseline",
                                builder=build_tcn_model,
                                train_config=train_cfg,
                                task_type=bundle.task_type,
                                input_kind="sequence",
                                group="baseline",
                                extra={"hidden_channels": 192, "layers": 3, "kernel_size": 3, "dropout": 0.1},
                                notes="Tiny TCN baseline",
                            )
                        )
                    EXPERIMENT_REGISTRY[bundle.name] = specs


                for bundle in DATA_BUNDLES.values():
                    register_specs(bundle)

                print("Registered model specs:")
                for dataset_name, specs in EXPERIMENT_REGISTRY.items():
                    print(f"- {dataset_name}: {[spec.name for spec in specs]}")
                """,
            ),
        ]
    )
    cells_data.extend(
        [
            (
                "code",
                """
                RUN_EXPERIMENTS = {
                    "EAF_TEMP_forecast": False,
                    "EAF_VALO2_forecast": False,
                    "EAF_chemistry": False,
                    "Beijing_PM25_24h_ctx_6h_horizon": False,
                    "Jena_tdegc_72ctx_36h": False,
                    "HAR_engineered": False,
                    "HAR_raw_sequence": False,
                    "Rossmann_sales": False,
                }
                """,
            ),
            (
                "code",
                """
                EXPERIMENT_ARTIFACTS: Dict[str, Dict[str, Any]] = {}

                for dataset_name, run_flag in RUN_EXPERIMENTS.items():
                    if not run_flag:
                        continue
                    if dataset_name not in DATA_BUNDLES:
                        print(f"[WARN] Dataset {dataset_name} not loaded; skipping.")
                        continue
                    bundle = DATA_BUNDLES[dataset_name]
                    specs = EXPERIMENT_REGISTRY.get(dataset_name, [])
                    if not specs:
                        print(f"[WARN] No model specs registered for {dataset_name}; skipping.")
                        continue
                    print("=" * 80)
                    print(f"Dataset: {dataset_name} ({bundle.task_type}, {bundle.input_kind})")
                    for spec in specs:
                        print(f"  -> Training {spec.name}")
                        result = train_model_on_bundle(bundle, spec, task_name=dataset_name)
                        EXPERIMENT_ARTIFACTS.setdefault(dataset_name, {})[spec.name] = result
                        artifact_path = RESULTS_ROOT / f\"{dataset_name}_{spec.name}_predictions.npz\"
                        np.savez_compressed(
                            artifact_path,
                            train_true=result["train_true"],
                            train_pred=result["train_pred"],
                            val_true=result["val_true"],
                            val_pred=result["val_pred"],
                            test_true=result["test_true"],
                            test_pred=result["test_pred"],
                        )
                        print(f"    Validation metrics: {result['val_metrics']}")
                        print(f"    Test metrics       : {result['test_metrics']}")
                        print(f"    Saved predictions to {artifact_path}")
                """,
            ),
            (
                "code",
                """
                results_df = RESULT_LOGGER.to_frame()
                results_path = RESULTS_ROOT / "experiment_metrics.csv"
                if not results_df.empty:
                    results_df.to_csv(results_path, index=False)
                    display(results_df)
                    print(f"Metrics saved to {results_path}")
                else:
                    print("No experiments were run yet. Toggle RUN_EXPERIMENTS before executing the training cell.")
                """,
            ),
        ]
    )
    cells_data.extend(
        [
            (
                "code",
                """
                TARGET_DATASET = "EAF_TEMP_forecast"
                TARGET_MODEL = "ResPSANN_tabular"

                if TARGET_DATASET in EXPERIMENT_ARTIFACTS and TARGET_MODEL in EXPERIMENT_ARTIFACTS[TARGET_DATASET]:
                    bundle = DATA_BUNDLES[TARGET_DATASET]
                    spec = next(spec for spec in EXPERIMENT_REGISTRY[TARGET_DATASET] if spec.name == TARGET_MODEL)
                    trained_model = EXPERIMENT_ARTIFACTS[TARGET_DATASET][TARGET_MODEL]["model"]

                    prefix_groups = {
                        "temp_lags": [i for i, name in enumerate(bundle.feature_names) if name.startswith("TEMP_lag")],
                        "valo2_lags": [i for i, name in enumerate(bundle.feature_names) if name.startswith("VALO2_lag")],
                        "gas_flow": [i for i, name in enumerate(bundle.feature_names) if "gas" in name.lower()],
                        "inj": [i for i, name in enumerate(bundle.feature_names) if "inj" in name.lower()],
                        "calendar": [i for i, name in enumerate(bundle.feature_names) if "DATETIME" in name],
                    }

                    perm_df = permutation_importance(
                        trained_model,
                        bundle,
                        spec,
                        feature_groups=prefix_groups,
                        split="test",
                        n_repeats=5,
                    )
                    display(perm_df.sort_values("mean_delta", ascending=False))
                else:
                    print("Train the target model first; EXPERIMENT_ARTIFACTS does not contain it yet.")
                """,
            ),
            (
                "code",
                """
                TARGET_DATASET = "Jena_tdegc_72ctx_36h"
                TARGET_MODEL = "ResPSANN_conv_spine"

                if TARGET_DATASET in EXPERIMENT_ARTIFACTS and TARGET_MODEL in EXPERIMENT_ARTIFACTS[TARGET_DATASET]:
                    bundle = DATA_BUNDLES[TARGET_DATASET]
                    spec = next(spec for spec in EXPERIMENT_REGISTRY[TARGET_DATASET] if spec.name == TARGET_MODEL)
                    trained_model = EXPERIMENT_ARTIFACTS[TARGET_DATASET][TARGET_MODEL]["model"].to(DEVICE)
                    sample_loader = build_dataloader(
                        bundle.val["X"],
                        bundle.val["y"],
                        batch_size=32,
                        shuffle=False,
                        task_type=spec.task_type,
                    )
                    sample_batch = next(iter(sample_loader))[0][:64]
                    singular_values = compute_jacobian_singular_values(trained_model, sample_batch, max_samples=64)
                    pr = participation_ratio(singular_values)
                    print(f"Participation ratio: {pr:.4f}")
                    trained_model.to("cpu")
                else:
                    print("Train the target model first to access EXPERIMENT_ARTIFACTS.")
                """,
            ),
            (
                "code",
                """
                TARGET_DATASET = "Beijing_PM25_24h_ctx_6h_horizon"
                TARGET_MODEL = "ResPSANN_conv_spine"

                if TARGET_DATASET in EXPERIMENT_ARTIFACTS and TARGET_MODEL in EXPERIMENT_ARTIFACTS[TARGET_DATASET]:
                    bundle = DATA_BUNDLES[TARGET_DATASET]
                    spec = next(spec for spec in EXPERIMENT_REGISTRY[TARGET_DATASET] if spec.name == TARGET_MODEL)
                    trained_model = EXPERIMENT_ARTIFACTS[TARGET_DATASET][TARGET_MODEL]["model"]

                    def missingness_fn(X: np.ndarray, level: float) -> np.ndarray:
                        rng = np.random.default_rng(GLOBAL_CONFIG["seed"])
                        mask = rng.random(size=X.shape) < level
                        X_corrupted = X.copy()
                        X_corrupted[mask] = 0.0
                        return X_corrupted

                    robustness_df = evaluate_robustness(
                        trained_model,
                        bundle,
                        spec,
                        corruption_fn=missingness_fn,
                        split="test",
                        levels=[0.0, 0.1, 0.2, 0.3, 0.4],
                    )
                    display(robustness_df)
                else:
                    print("Train the target model before running robustness experiments.")
                """,
            ),
            (
                "code",
                """
                TARGET_DATASET = "HAR_raw_sequence"
                if TARGET_DATASET in EXPERIMENT_ARTIFACTS:
                    results = EXPERIMENT_ARTIFACTS[TARGET_DATASET]
                    if "ResPSANN_conv_spine" in results and "ResPSANN_attention_spine" in results:
                        conv_acc = results["ResPSANN_conv_spine"]["test_metrics"]["accuracy"]
                        attn_acc = results["ResPSANN_attention_spine"]["test_metrics"]["accuracy"]
                        print(f"Conv spine accuracy: {conv_acc:.4f}")
                        print(f"Attention spine accuracy: {attn_acc:.4f}")
                    else:
                        print("Run both PSANN spine variants on HAR_raw_sequence first.")
                else:
                    print("Train HAR_raw_sequence models before evaluating H5.")
                """,
            ),
        ]
    )
    cells_data.extend(
        [
            (
                "md",
                """
                ## Notebook Complete
                All core experiment scaffolding is now in place. Toggle the runs you need, execute the training cell per hypothesis, and archive outputs from `colab_results/` before ending your Colab session.
                """,
            ),
        ]
    )
    nb["cells"] = []
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
        },
    }
    for cell_type, content in cells_data:
        if cell_type == "md":
            nb["cells"].append(md(content))
        else:
            nb["cells"].append(code(content))
    output_path = Path("respsann_compute_parity_colab.ipynb")
    nbf.write(nb, output_path.open("w", encoding="utf-8"))
    print(f"Notebook scaffold written to {output_path}")


if __name__ == "__main__":
    main()
