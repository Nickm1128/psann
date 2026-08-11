# Tested Constraints

Constraint files capture reproducible validation snapshots without making every
transitive package a permanent runtime pin.

For the Phase 1 Python 3.11 CPU/workstation profile:

```bash
python -m pip install -c constraints/workplace-py311.txt -e .[dev,sklearn]
```

Install the correct PyTorch wheel for the target CPU/CUDA environment before applying
the constraint file when the default package index is not appropriate. Accelerator
images should record the PyTorch build suffix, CUDA runtime, and driver separately.

The package metadata continues to declare supported dependency floors. Update a
constraint snapshot only after the fast suite, scoped coverage, lint, hygiene, and
built-wheel smoke checks pass together.

`workplace-floor.txt` is the blocking core floor consumed by CI. It proves the
declared NumPy 1.26, PyTorch 2.4, SciPy 1.11, and scikit-learn 1.4 boundary from an
installed wheel. `workplace-py311.txt` is the reviewed current workstation snapshot,
also consumed by CI so dependency drift cannot leave the maintained file untested.

`explain-floor-py311.txt` pins the lowest supported SHAP release in an isolated
NumPy-2-capable job. The current SHAP job resolves the newest version admitted by the
package extra. Explanation constraints intentionally remain separate from the
NumPy-1.26 core floor.

`deployment-py311.txt` is the locked CPU reference-service snapshot used by
`deploy/Dockerfile`. Torch is pinned and installed from the official CPU wheel index
before the remaining locked dependencies. Update that file only after the container
can load a mounted `.psann` artifact and pass health, readiness, metadata, and
prediction smoke checks.
