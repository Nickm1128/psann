import argparse
import os
import sys
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run GPU pytest suite with outputs")
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Directory to store test artifacts (default: outputs/gpu_tests/<timestamp>)",
    )
    parser.add_argument(
        "--markers",
        type=str,
        default="gpu",
        help="Pytest -m marker expression to select tests (default: gpu)",
    )
    parser.add_argument(
        "--testpaths",
        type=str,
        default="tests/gpu",
        help="Paths to pass to pytest (default: tests/gpu)",
    )
    args, extra = parser.parse_known_args()

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    outdir = Path(args.outdir) if args.outdir else Path("outputs") / "gpu_tests" / ts
    outdir.mkdir(parents=True, exist_ok=True)

    # Let tests know where to write artifacts
    os.environ["PSANN_OUTPUT_DIR"] = str(outdir)

    # Defer import to after env var set
    import pytest  # type: ignore

    junit_path = outdir / "junit.xml"
    log_path = outdir / "pytest.log"

    cmd = [
        "-q",
        "-m",
        args.markers,
        "--maxfail=1",
        "--disable-warnings",
        "--durations=10",
        "--junitxml",
        str(junit_path),
        "--log-file",
        str(log_path),
        *args.testpaths.split(),
        *extra,
    ]

    print("Running pytest with:", " ".join(cmd))
    code = pytest.main(cmd)
    print(f"JUnit report: {junit_path}")
    print(f"Pytest log:   {log_path}")
    sys.exit(code)


if __name__ == "__main__":
    main()
