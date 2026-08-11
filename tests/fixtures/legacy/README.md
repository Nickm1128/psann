# Retained public legacy checkpoint

`psann-0.12.7-regressor.pt` was created by the exact public PSANN `0.12.7`
wheel recorded in `psann-0.12.7-regressor.json`. The sidecar contains the wheel
URL and SHA256, the checkpoint SHA256, deterministic inputs, training settings,
and producer predictions used by the compatibility tests.

Rebuild the fixture from the pinned wheel with:

```bash
python tools/generate_legacy_fixture.py
```

The generator creates an isolated environment, verifies the wheel hash, trains
and saves through the `0.12.7` public API, and verifies that the producer can load
its own result. Pass `--wheel PATH` to rebuild without downloading after obtaining
the recorded wheel through an approved channel.

## Security boundary

The `.pt` file uses Python pickle through `torch.save`. Loading any pickle can
execute code. PSANN's compatibility test explicitly opts into trusted legacy
loading because this retained file has pinned provenance and a checked SHA256.
Never extend that trust to unknown or user-supplied checkpoints. Native `.psann`
deployment artifacts remain the safe default.
