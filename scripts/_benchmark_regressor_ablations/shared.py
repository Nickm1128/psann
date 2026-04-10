# ruff: noqa: F401
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from scripts._cli_utils import ensure_src_dir, parse_comma_list, slugify
except ImportError:  # pragma: no cover - supports `python scripts/foo.py`
    from _cli_utils import ensure_src_dir, parse_comma_list, slugify  # type: ignore

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = ensure_src_dir(_REPO_ROOT)

from gpu_env_report import gather_env_info
from psann import ResPSANNRegressor, SGRPSANNRegressor, WaveResNetRegressor
from psann.params import count_params as psann_count_params
from psann.utils import (
    make_context_rotating_moons,
    make_drift_series,
    make_regime_switch_ts,
    make_shock_series,
    seed_all,
)
