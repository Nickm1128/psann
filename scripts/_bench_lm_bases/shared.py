# ruff: noqa: F401
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from scripts._cli_utils import ensure_src_dir, utc_timestamp_tag, write_json
except ImportError:  # pragma: no cover - supports `python scripts/foo.py`
    from _cli_utils import ensure_src_dir, utc_timestamp_tag, write_json  # type: ignore

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ensure_src_dir(ROOT)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import yaml
except Exception as exc:  # pragma: no cover - import guard
    raise SystemExit("Missing dependency 'pyyaml'. Install via `pip install pyyaml`.") from exc

import torch
from psannlm.lm.api import psannLM
from psannlm.lm.config import TrainConfig
from psannlm.lm.data.dataset import HFTextStreamingLMDataset, PackingConfig, build_text_filter
from psannlm.lm.data.tokenizer import Tokenizer, TokenizerConfig
from psannlm.lm.models.registry import get_base, list_bases
from psannlm.lm.models.sine import SineConfig
from psannlm.lm.train.trainer import Trainer
