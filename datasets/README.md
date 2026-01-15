# datasets/

This directory contains **small, versioned fixtures** used by quick smoke tests and benchmark configs.

- Large datasets should not live in this repo. Use `eval_data/` (ignored) or external storage.
- This folder is mostly ignored by git, except for explicitly whitelisted small fixtures.

Current fixtures:
- `wave_resnet_small.npz` — small dataset used by HISSO smoke configs under `configs/hisso/`.

