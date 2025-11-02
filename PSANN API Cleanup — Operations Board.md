# PSANN API Cleanup — Operations Board

**Owner:** Nick  
**Mode:** CPU first, GPU last.  
**Hang rule:** Kill any command that runs > 10 minutes without new log output.  
**Last updated:** 2025-11-02

### Autosummary (Codex must update)
- **Progress (weighted):** `84 / 189` checkboxes complete → `44.4%` (simple)
- **Progress (complexity-weighted):** `76.3%` _(Codex: computed from (c=K) tasks: 74/97 K complete)_
- **Open items:** `0` in "Now", `0` in "Next", `0` in "Blocked"
- **Latest GPU sweep:** Runpod L4, CUDA 12.1, Torch 2.8.0+cu128; WaveResNet HISSO metrics logged (see Runs and GPU Sweep Summary: docs/PSANN_Results_Compendium.md#gpu-sweep-summary-table).

---

## 0) Rules Codex Must Follow (treat as system instructions)

1) **CPU first, GPU last.** Do not schedule any CUDA job unless the “CPU Prep” queue is empty and a remote GPU slot is explicitly available.  
2) **Weighted progress:** Each task has a complexity ∈ {1,2,3,4,5}. Use this to compute progress.  
3) **Atomic edits:** Keep changes small and test-backed. If a change modifies public behavior, patch docs in the same PR.  
4) **If it hangs > 10 minutes:** terminate and write a short “Autopsy” line under Session Log.  
5) **When blocked:** move the item to “Blocked”, add a one-liner stating what’s needed from Nick, and continue to the next CPU item.  
6) **Every commit:** run `pytest -m "not slow"` locally; reserve full suite for CI and for GPU slots.  
7) **Doc drift:** if behavior changes, update `README.md`/`docs/API.md`/relevant docs in same commit.  
8) **Session handoff:** Append a 3-bullet summary under **Session Log** at the end of each burst.

---

## 1) Weighted Progress — how to calculate (Codex must maintain)

- **Each task** line starts with `(c=K)` where `K ∈ {1..5}` is the complexity.  
- **Weighted completion %** = `sum(done K) / sum(all K) * 100`.  
- **Also keep simple %** = `done_checkboxes / total_checkboxes * 100` for sanity.

_Pseudo-calc (don’t remove):_
total_K = sum(K for all tasks)
done_K = sum(K for all checked tasks)
weighted_progress = 100 * done_K / total_K

---

## 2) Queues (work in this order)

### A) **Now — CPU Prep** (finish these before any GPU runs)
- (c=3) Ensure **alias normalization** used across estimators/builders; grep for `hidden_units/hidden_width`, `conv_channels/hidden_channels`; all paths must hit `resolve_int_alias`. Add/refresh tests where gaps exist. ✅ core done, but re-grep after any estimator edits.
- (c=2) **Error/exception formatting invariants** audit: shapes echoed; `ValueError` vs `TypeError` consistent; warnings have `stacklevel>=2`. ✅ mostly done; re-run tests after any validator edits.
- (c=2) **Device/dtype policy**: float32 interop, fast-path for already-correct tensors, `_model_device_` cache intact. Re-run quick suite after any `predict*` edits.
- (c=3) **Inference roundtrip**: ensure `_prepare_inference_inputs` + `_reshape_predictions` keep fitted shape and `channels_last` options. Keep save/load parity test green.
- (c=2) **Streaming updates**: `_ensure_streaming_ready` tracks model token + LR changes; `_coerce_stream_target` reshape rules stable. Smoke tests must stay green.
- (c=2) **Docs sync**: whenever the above changes, patch `README.md`, `docs/API.md`, example snippets.

### B) **Next — Docs & CI polish (CPU)**
- (c=2) Tighten **README Quick Start** examples for `WaveResNetRegressor` and `ResConvPSANNRegressor` with deterministic seeds and dtype/device tips. ✅ archived
- (c=2) CI markers: keep slow HISSO tests behind `-m slow`; verify the marker count after adding/removing tests. ✅ archived
- (c=2) “Technical details” section: call out **dataloader shuffle policy** for stateful models and `preserve_shape` semantics. ✅ archived

### C) **GPU Sweep (run last, only if slot available)**
- (c=3) Run **HISSO logging CLI** for WaveResNet on Runpod L4; export to `runs/hisso/wave_resnet_cuda_runpod_YYYYMMDD_HHMMSS/`. Record throughput, best epoch, train/val/test, reward_mean, turnover, AMP status.
- (c=3) CUDA smoke: `pytest -k cuda -m "not slow"` + `tests/test_training_loop.py::test_training_loop_early_stopping_runs_on_cuda`.
- (c=2) Mixed precision audit: confirm AMP stability; note any inf/NaN shielding or grad-scaler tweaks needed; update docs.

### D) **Blocked (requires Nick)**
All previously blocked items resolved 2025-11-02. See details below.

### D-resolved) Decisions (one-liners)
- (c=3) Logging dirs: Local `runs/hisso/`; Colab/Runpod `/content/hisso_logs/`. CLI keeps explicit `--output-dir` and docs/CLI help updated.
- (c=3) Extras datasets: Archived in the compendium; active sections now primary-only. Links to `docs/backlog/extras-removal.md` retained.
- (c=3) Mixed precision: Finalized AMP guidance (float16 + GradScaler) in HISSO logging spec; stability notes added.

---

## 3) Canonical Task Clusters (authoritative lists Codex maintains)

> Most of these are already implemented and tested; Codex should keep them green and only touch when related changes happen. Completed items are archived below to reduce noise.

### 3.1 Global Consistency
- (c=3) Parameter aliasing policy unified across estimators and builders with tests. ✅ archived  
- (c=2) Error/exception messaging standardized with echo’d shapes and proper categories. ✅ archived  
- (c=2) Public API exports validated (`__all__`), docs match canonical names. ✅ archived  
- (c=2) Shape/layout semantics for `preserve_shape` and CF/CL uniform. ✅ archived  
- (c=2) Device/dtype policy with float32 interop and minimal transfers. ✅ archived

### 3.2 Estimators (`sklearn.py`)
- (c=2) Remove duplicate `ResConvPSANNRegressor` definition; keep authoritative class at `:2209`. ✅ archived  
- (c=2) `set_params/get_params` alias normalization and warnings; sklearn parity. ✅ archived  
- (c=3) Fit pipeline invariants (context builder caching, scaler warm-start). ✅ archived  
- (c=3) Inference and sequence APIs: save/load parity; `predict_sequence[_online]` shapes. ✅ archived  
- (c=3) Streaming updates: optimizer rebuild tracking; reshape rules. ✅ archived

### 3.3 HISSO (trainer/options)
- (c=3) Options normalization; transforms parity (numpy/torch). ✅ archived  
- (c=3) Warm-start + episodic training honors `lsm_lr`, preserves history. ✅ archived  
- (c=3) Trainer performance & determinism; batched transfers; seed control. ✅ archived

### 3.4 Models & Layers
- (c=3) WaveResNet: w0 warmup, progressive depth, context gating, dropout invariants. ✅ archived  
- (c=3) Residual/dense cores: DropPath train/eval toggles; stateful commit/reset. ✅ archived  
- (c=3) Convolutional nets: segmentation vs pooled outputs; alias error behavior. ✅ archived

### 3.5 Preprocessing
- (c=2) LSM/Expander builders: dict/spec/module uniform; freeze & dim propagation. ✅ archived

### 3.6 State & Tokenization
- (c=2) StateController: commit/reset flows; detach and warnings; negative axis support. ✅ archived  
- (c=2) SimpleWordTokenizer + SineTokenEmbedder: encode/decode; BOS/EOS; frequency schedule. ✅ archived

### 3.7 Metrics & Rewards
- (c=2) Portfolio metrics stability (eps clamps) and reward registry overwrite semantics. ✅ archived

### 3.8 Serialization
- (c=3) Estimator checkpoints persist context builders + HISSO metadata; roundtrip tests. ✅ archived

### 3.9 Training Loop & Performance
- (c=3) LR schedule interpolation; early-stopping restore; hook invocation; shuffle guidance. ✅ archived

### 3.10 Documentation & Examples
- (c=2) README + TECHNICAL_DETAILS reflect alias/device/context/stem/shuffle policies. ✅ archived

### 3.11 Test Plan & CI
- (c=2) Coverage spans aliasing, shapes, context, HISSO, streaming; slow marked; fast by default. ✅ archived

---

## 4) Runs & Artifacts (Codex should append new entries)
- **2025-11-02 — Runpod L4 WaveResNet HISSO**
  `runs/hisso/wave_resnet_cuda_runpod_20251102_212855/`
  Throughput ~113.07 eps/s; best_epoch=56; train/val/test 0.722/0.864/0.835;
  reward_mean −0.114 (±0.0103); turnover 3.18; Sharpe −1.87; AMP float16; duration ~18.37 s.
- **2025-11-02 — Runpod L4 WaveResNet HISSO**  
  `runs/hisso/wave_resnet_cuda_runpod_20251102_153117/`  
  Throughput ~107.3 eps/s; best_epoch=17; train/val/test 0.621/0.755/0.670; reward_mean −0.114 (±0.010), turnover 2.69; AMP float16.
- **Colab CUDA (2025-11-01)**  
  Dense: 203 eps/s (2.68 s wall), train/val/test 0.245/0.304/0.231; reward_mean −0.111.  
  WaveResNet: 161 eps/s (3.34 s wall), train/val/test 1.435/1.402/1.569; reward_mean −0.182; turnover 2.65.

---

## 5) CPU Prep Before GPU Run (quick checklist)
- [x] (c=1) Re-run **CPU smoke baselines**: `runs/hisso/dense_smoke_cpu_dev` and `runs/hisso/wave_resnet_cpu_smoke` after any loader or dtype/device edits.  
- [x] (c=1) Confirm **datasets/wave_resnet_small.npz** is staged for remote.  
- [x] (c=1) Ensure **HISSO logging CLI** docs are current; leave placeholders where GPU metrics will be inserted.

---

## 6) Final GPU Sweep (only when A/B queues are empty)
- [x] (c=2) `pytest -m "not slow"` on GPU node; then full `pytest`.  
- [x] (c=3) Run HISSO logging CLI for WaveResNet; export metrics; add to Runs & Artifacts.  
- [x] (c=2) Document AMP behavior & stability notes.

---

## 7) Outstanding Questions (awaiting Nick)
- None at this time; prior items resolved on 2025-11-02.

---

## 8) Session Log (rolling)
- **2025-11-02:** Resolved blocked items: logging-dir convention (local `runs/hisso/`, Colab `/content/hisso_logs/`), archived extras datasets in the compendium, and finalized AMP guidance (float16 + GradScaler). Added GPU Sweep Summary table and linked from the board. Updated HISSO GPU notebook with Runpod 212855/153117 metrics and Colab output root. Autosummary open items updated.
- **2025-11-02:** Documented AMP behavior and stability (float16 on L4; no inf/NaN; scaler stable). Updated compendium GPU sweep with Runpod 212855 metrics and replaced notebook GPU TODOs with WaveResNet results. Autosummary updated (84/189; 44.4% simple; 76.7% weighted).
- **2025-11-02:** Marked GPU sweep tests and WaveResNet HISSO run as complete based on latest results; Autosummary updated.  
- **2025-11-02:** GPU fast tier green (126 passed, 18 deselected, 1 warning). CUDA smoke green (1 passed).
  WaveResNet HISSO on L4 completed; metrics harvested from metrics.json → summary.json; AMP float16 active.
- **2025-11-02:** CPU Prep checklist advanced: re-ran CPU smoke baselines (dense, wave_resnet) to refresh metrics; confirmed `datasets/wave_resnet_small.npz` staged; HISSO CLI docs/walkthrough show GPU metrics placeholders. `pytest -m "not slow"` green (128 passed, 1 skipped). Autosummary updated.  
- **2025-11-02:** Docs & CI polish (CPU) complete: tightened README examples (ResConv, WaveResNet) with dtype/device tips; added slow-marker guidance to CONTRIBUTING; clarified per-element vs pooled output shapes in TECHNICAL_DETAILS; ran `pytest -m "not slow"` locally (128 passed, 1 skipped).
- **2025-11-02:** Verified all "Now — CPU Prep" invariants (alias normalisation across estimators/builders, error/exception formatting, device/dtype policy, inference roundtrip, streaming updates). Ran `pytest -m "not slow"`: 128 passed, 1 skipped. Updated Autosummary open items to Now=0; no code changes required.
- **2025-11-02:** CUDA validation passed on Runpod L4; WaveResNet HISSO run logged; docs refreshed.  
- **2025-11-02:** CPU Now audit complete: alias normalization, error/exception formatting, device/dtype policy, inference roundtrip, and streaming updates re-verified; pytest -m "not slow" green (128 passed, 1 skipped). No code changes required.  
- **2025-10-29:** HISSO logging CLI + dataset staging complete; CPU baselines recorded.  
- **2025-10-28:** Global consistency, estimator fit/inference, device/dtype policy, and training-loop coverage locked with tests.

---

## 9) Completed Archive (locked — do not edit)
> Codex: keep this section intact; append new completions here in batches.  
> Summary: Global Consistency, Estimators fit/inference/streaming, HISSO options/warm-start/perf, Models & Layers, Preproc, State & Tokenization, Metrics & Rewards, Serialization, Training Loop & Performance, Docs & CI — all completed with regression coverage and documented.  
> Representative tests include:
> - `tests/test_regressor_aliases.py`, `tests/test_error_messages.py`, `tests/test_preserve_shape_layout.py`  
> - `tests/test_context_builder.py`, `tests/test_regressor_inference.py`, `tests/test_stateful_streaming.py`  
> - `tests/test_hisso_primary.py`, `tests/test_hisso_smoke.py`, `tests/test_hisso_logging_cli.py`  
> - `tests/test_wave_resnet.py`, `tests/test_wave_resnet_regressor.py`, `tests/test_conv_nets.py`, `tests/test_conv_stem_helper.py`  
> - `tests/test_preproc.py`, `tests/test_regressor_lsm.py`, `tests/test_state_controller.py`  
> - `tests/test_metrics_rewards.py`, `tests/test_training_loop.py`, `tests/test_dataloader_shuffle.py`

---

### Footnotes (for the paper tie-ins Codex may cite when updating docs)
- PSANN’s core idea and spine design (learnable sine + tiny Conv1d) are summarized in the Abstract, Intro, Models, Discussion, and Conclusion of the internal paper draft: frequency-aware prior, tiny temporal bias, and compute-parity framing. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1} :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}  
- Minimal next steps and parity tightening recommendations are captured for reproducible follow-ups. :contentReference[oaicite:6]{index=6} :contentReference[oaicite:7]{index=7} :contentReference[oaicite:8]{index=8}

---

### How to use this board (Codex quickstart)
1. Work top-to-bottom: **Now → Next → GPU Sweep**.  
2. After each commit:  
   - Update checkboxes and `(c=K)` if you learned the true complexity.  
   - Recompute the **weighted progress** and update the Autosummary line.  
   - Add 1–3 bullets under **Session Log**.  
3. If blocked: move the item to **Blocked** with one crisp line for Nick. Keep moving.
