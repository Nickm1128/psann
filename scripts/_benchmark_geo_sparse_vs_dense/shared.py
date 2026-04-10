# ruff: noqa: F401
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from scripts._cli_utils import ensure_src_dir, write_json
except ImportError:  # pragma: no cover - supports `python scripts/foo.py`
    from _cli_utils import ensure_src_dir, write_json  # type: ignore

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = ensure_src_dir(_REPO_ROOT)

from gpu_env_report import gather_env_info
from psann.layers.sine_residual import RMSNorm
from psann.nn_geo_sparse import GeoSparseNet
from psann.nn_geo_sparse import _build_activation as _build_geo_activation
from psann.params import count_params, dense_mlp_params, geo_sparse_net_params, match_dense_width
from psann.utils import choose_device, seed_all
