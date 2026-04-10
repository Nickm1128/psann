# ruff: noqa: F401
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import subprocess
import sys
import time
import types
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    from scripts._cli_utils import ensure_src_dir, write_json
except ImportError:  # pragma: no cover - supports `python scripts/foo.py`
    from _cli_utils import ensure_src_dir, write_json  # type: ignore

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = ensure_src_dir(_REPO_ROOT)

from gpu_env_report import gather_env_info
from psann.conv import PSANNConv1dNet
from psann.params import count_params, match_dense_width
from psann.utils import choose_device
from psann.utils import seed_all as psann_seed_all
