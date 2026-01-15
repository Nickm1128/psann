# How to Add a Model Base / Benchmark / Dataset

This guide documents the expected steps to add a new model base, benchmark, or dataset entry.

## Add a model base

1. **Implement the module**
   - For core estimators, add the class under `src/psann/`.
   - For LM bases, add the transformer class under `psannlm/lm/models/`.
2. **Register it**
   - LM: update `psannlm/lm/models/registry.py` to add the new base name.
   - Core estimators: export it in `src/psann/__init__.py` if it is public.
3. **Document expected params**
   - Add constructor options to `docs/API.md` or `docs/lm.md`.
4. **Add tests**
   - Core estimator smoke test in `tests/`.
   - LM forward test in `tests/lm/`.

## Add a benchmark

1. **Create or update a script** under `scripts/`.
2. **Log outputs** into `reports/` or `outputs/` and keep those directories ignored by git.
3. **Add a short entry** to `scripts/README.md` so the benchmark is discoverable.
4. **Optional**: add a CI‑friendly smoke test (CPU‑only) under `tests/`.

## Add a dataset

1. **Place local shards** under `datasets/` or point to a HF dataset in scripts.
2. **Document the source** (URL, license, preprocessing) in the relevant doc (`docs/lm.md`, `docs/benchmarks/`).
3. **Keep data out of git** (use `.gitignore` and `datasets/README.md`).
