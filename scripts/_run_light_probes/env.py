# ruff: noqa: F403,F405
from __future__ import annotations

from .shared import *


def _maybe_install(module: str, package: str | None = None) -> None:
    """Install a dependency if it cannot be imported."""
    try:
        importlib.import_module(module)
    except ImportError:
        target = package or module
        print(f"[deps] Installing {target}...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", target])


_DEPENDENCIES = [
    ("psann", "psann"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("torchaudio", "torchaudio"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("sklearn", "scikit-learn"),
]


def ensure_dependencies() -> None:
    for module_name, package_name in _DEPENDENCIES:
        _maybe_install(module_name, package_name)


def _ensure_torch_dynamo_stub() -> None:
    """Provide minimal torch._dynamo API pieces when missing."""

    def _make_disable():
        def _disable(fn=None, recursive=True):
            if fn is None:

                def decorator(f):
                    return f

                return decorator
            return fn

        return _disable

    def _graph_break(*_args, **_kwargs):
        return None

    try:
        dynamo = importlib.import_module("torch._dynamo")
    except Exception:  # pragma: no cover - defensive stub
        dynamo = None

    if dynamo is None:
        stub = types.ModuleType("torch._dynamo")
        stub.disable = _make_disable()
        stub.graph_break = _graph_break
        sys.modules["torch._dynamo"] = stub
        torch._dynamo = stub  # type: ignore[attr-defined]
        return

    if not getattr(dynamo, "disable", None):
        dynamo.disable = _make_disable()  # type: ignore[attr-defined]
    if not getattr(dynamo, "graph_break", None):
        dynamo.graph_break = _graph_break  # type: ignore[attr-defined]
    torch._dynamo = dynamo  # type: ignore[attr-defined]


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.getenv("PSANN_DATA_ROOT", PROJECT_ROOT / "datasets")).resolve()
DEFAULT_RESULTS_ROOT = (PROJECT_ROOT / "colab_results_light").resolve()
RESULTS_ROOT = DEFAULT_RESULTS_ROOT


def configure_results_root(path: Optional[str]) -> Path:
    """Resolve and create the results directory, updating the module global."""
    global RESULTS_ROOT
    target = Path(path).expanduser().resolve() if path else DEFAULT_RESULTS_ROOT
    target.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT = target
    return target


def get_results_root() -> Path:
    return RESULTS_ROOT


REQUIRED_DATASETS = [
    DATA_ROOT / "Beijing Air Quality",
    DATA_ROOT / "Industrial Data from the Electric Arc Furnace",
]

JENA_ZIP_URL = (
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
)


def _datasets_ready() -> bool:
    for path in REQUIRED_DATASETS:
        if not path.exists():
            return False
        if path.is_dir() and not any(path.rglob("*")):
            return False
    return True


def ensure_jena_dataset() -> Path:
    base = DATA_ROOT / "Jena Climate 2009-2016"
    csv = base / "jena_climate_2009_2016.csv"
    if csv.exists():
        return csv
    base.mkdir(parents=True, exist_ok=True)
    tmp_zip = base / "jena_climate_2009_2016.csv.zip"
    import urllib.request

    try:
        print(f"[data] Downloading Jena Climate dataset to {base} ...")
        urllib.request.urlretrieve(JENA_ZIP_URL, tmp_zip)
        with zipfile.ZipFile(tmp_zip, "r") as zf:
            zf.extractall(base)
    except Exception as exc:  # pragma: no cover - network failure surfaces to user
        print(f"[warn] Failed to download Jena dataset: {exc}")
    finally:
        if tmp_zip.exists():
            tmp_zip.unlink()
    if csv.exists():
        return csv
    matches = list(base.glob("**/jena_climate_2009_2016.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not locate Jena climate CSV under {base}")


def seed_all(seed: int) -> None:
    psann_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device(arg: str) -> torch.device:
    return choose_device(arg)


def _fix_backslash_artifacts(root: Path) -> None:
    for leftover in root.iterdir():
        name = leftover.name
        if "\\\\" in name and name.lower().startswith("datasets"):
            rel = Path(*Path(name.replace("\\", "/")).parts)
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                leftover.replace(dest)
                print(f"[fix] Moved stray {leftover} -> {dest}")
            except Exception as exc:
                print(f"[warn] Could not move {leftover}: {exc}")


def maybe_extract_datasets_zip() -> None:
    _fix_backslash_artifacts(PROJECT_ROOT)
    if _datasets_ready():
        return
    zip_path = PROJECT_ROOT / "datasets.zip"
    if not zip_path.exists():
        print(f"[warn] Required datasets missing under {DATA_ROOT} and datasets.zip not found.")
        return
    print(f"[info] Extracting {zip_path} to {PROJECT_ROOT}/datasets (robust normalisation)...")
    with zipfile.ZipFile(zip_path, "r") as z:
        for zi in z.infolist():
            name = zi.filename
            norm = name.replace("\\", "/").lstrip("./")
            parts = [p for p in norm.split("/") if p]
            if not parts:
                continue
            if parts[0].lower() != "datasets":
                parts = ["datasets"] + parts
            dest = PROJECT_ROOT.joinpath(*parts)
            if zi.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                with z.open(zi) as src, open(dest, "wb") as f:
                    f.write(src.read())
    _fix_backslash_artifacts(PROJECT_ROOT)
