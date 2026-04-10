# ruff: noqa: F401
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.datasets import (
    fetch_california_housing,
    load_diabetes,
    load_linnerud,
    make_friedman1,
    make_regression,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from scripts._cli_utils import ensure_src_dir, write_json
except ImportError:  # pragma: no cover - supports `python scripts/foo.py`
    from _cli_utils import ensure_src_dir, write_json  # type: ignore

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = ensure_src_dir(_REPO_ROOT)

from psann import GeoSparseRegressor, PSANNRegressor, count_params  # noqa: E402
from psann.nn import PSANNNet  # noqa: E402
from psann.nn_geo_sparse import GeoSparseNet  # noqa: E402
