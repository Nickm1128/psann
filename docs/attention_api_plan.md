# Attention Configuration Plan

## Goal
Expose a unified `attention` configuration knob across PSANN estimators and the LM stack so users can append self-attention layers at model initialization without bespoke wrappers. Defaults keep current behaviour (no attention) while enabling `{"kind": "mha", ...}` style specs everywhere.

## Proposed Work
1. **Shared Config/Builder**
   - Add `AttentionConfig` dataclass plus `ensure_attention_config` helper (similar to `StateConfig`).
   - Provide a minimal registry/builder that currently supports `"none"` and `"mha"` (PyTorch `nn.MultiheadAttention`).

2. **PSANNRegressor Integration**
   - Extend `PSANNRegressor.__init__` with `attention=` kwarg (accepting dict/dataclass).
   - Implement `_wrap_with_attention_dense` / `_wrap_with_attention_conv` helpers that inject attention between the base core and the readout.
   - Ensure per-element + preserve-shape paths either support attention or clearly error with guidance.

3. **Residual / Wave ResNet Variants**
   - ResPSANN / ResConvPSANN should inherit the same behaviour by reusing the base wrappers.
   - WaveResNetRegressor overrides `_build_dense_core`; adjust it to wrap the WaveResNet backbone output in attention when configured.

4. **LM Transformer Alignment (Optional but recommended)**
   - Thread `AttentionConfig` through `psannLM` / `ModelConfig`.
   - Swap hard-coded `nn.MultiheadAttention` in `transformer_respsann.py` / `transformer_waveresnet.py` for the shared builder.

5. **Docs & Tests**
   - Update `docs/API.md` and `README.md` with examples (`attention={"kind": "mha", "num_heads": 4}`).
   - Add regression tests that enable attention on PSANNRegressor and WaveResNetRegressor, covering shape mismatches and forward passes.

## Execution Notes
- Start with the shared config so all downstream modules import the same helpers.
- Roll integration out incrementally: base estimator → residuals → wave → LM.
- Maintain backwards compatibility by defaulting `attention` to `"none"`.
- Document any unsupported combinations (e.g., attention with flat inputs but unspecified sequence length) clearly in error messages and docs.
