"""Generate a simple loss curve PNG from a metrics.csv file.

Usage:
  python scripts/plot_loss_from_csv.py --csv reports/benchmarks/<ts>/metrics.csv \
      --out reports/benchmarks/<ts>/loss_curve.png

The CSV is expected to have at least the columns: step, loss.
If matplotlib is not available or the CSV lacks data, the script exits quietly.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to metrics.csv")
    ap.add_argument(
        "--out", required=True, help="Output PNG path for the loss curve"
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)
    if not csv_path.exists():
        return

    steps: list[int] = []
    losses: list[float] = []
    try:
        with csv_path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    steps.append(int(row.get("step", "") or 0))
                    losses.append(float(row.get("loss", "nan")))
                except Exception:
                    continue
    except Exception:
        return

    if not steps or not losses:
        return

    try:
        import matplotlib.pyplot as plt  # type: ignore

        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6, 4))
        plt.plot(steps, losses, marker="o", linewidth=1)
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title("Training loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=144)
        plt.close()
    except Exception:
        # matplotlib may not be installed; fail quietly per scripts convention
        return


if __name__ == "__main__":
    main()

