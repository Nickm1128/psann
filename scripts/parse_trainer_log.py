"""Parse Trainer stdout to extract loss/perplexity and write CSV (+ optional plot).

Looks for lines like:
  rank=0 epoch=1 step=50 loss=1.2345 ppl=3.456 lr=1e-4 grad_norm=... toks/step~...

Usage:
  python scripts/parse_trainer_log.py --log reports/benchmarks/<ts>/tiny_benchmark.log --out reports/benchmarks/<ts>
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List


LINE_RE = re.compile(
    r"epoch=(?P<epoch>\d+)\s+step=(?P<step>\d+)\s+loss=(?P<loss>[\d\.eE+-]+)\s+ppl=(?P<ppl>[\d\.eE+-]+)\s+lr=(?P<lr>[\d\.eE+-]+)\s+grad_norm=(?P<gn>[\d\.eE+-]+)\s+toks/step~(?P<toks>\d+)"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--plot", action="store_true", help="Also write loss_curve.png if matplotlib is available")
    args = ap.parse_args()

    logp = Path(args.log)
    outdir = Path(args.out)
    lines: List[str] = []
    try:
        lines = logp.read_text(encoding="utf-8").splitlines()
    except Exception:
        # If log missing, produce empty metrics.csv and exit
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "metrics.csv").write_text("epoch,step,loss,ppl,lr,grad_norm,toks\n", encoding="utf-8")
        return

    rows: List[str] = ["epoch,step,loss,ppl,lr,grad_norm,toks"]
    xs: List[int] = []
    ys: List[float] = []
    for ln in lines:
        m = LINE_RE.search(ln)
        if not m:
            continue
        epoch = int(m.group("epoch"))
        step = int(m.group("step"))
        loss = float(m.group("loss"))
        ppl = float(m.group("ppl"))
        lr = float(m.group("lr"))
        gn = float(m.group("gn"))
        toks = int(m.group("toks"))
        rows.append(f"{epoch},{step},{loss:.6f},{ppl:.6f},{lr:.6g},{gn:.6f},{toks}")
        xs.append(step)
        ys.append(loss)

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "metrics.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    if args.plot and xs and ys:
        try:
            import matplotlib.pyplot as plt  # type: ignore

            plt.figure(figsize=(6, 4))
            plt.plot(xs, ys, marker="o", linewidth=1)
            plt.xlabel("step")
            plt.ylabel("loss")
            plt.title("Training loss")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(outdir / "loss_curve.png", dpi=144)
            plt.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

