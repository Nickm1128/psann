# PSANN Research Notebooks

This directory hosts exploratory and reproducibility notebooks that complement the library and docs.

## Available notebooks

- [**PSANN_Parity_and_Probes.ipynb**](PSANN_Parity_and_Probes.ipynb) &nbsp;[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nickm1128/psann/blob/main/notebooks/PSANN_Parity_and_Probes.ipynb)\
  Demonstrates the compute-parity experiment suite discussed in `data_descriptions.txt` and the accompanying reports. Designed for Google Colab.
- [**HISSO_Logging_CLI_Walkthrough.ipynb**](HISSO_Logging_CLI_Walkthrough.ipynb) &nbsp;[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nickm1128/psann/blob/main/notebooks/HISSO_Logging_CLI_Walkthrough.ipynb)\
  CPU-first draft for the HISSO logging CLI workflow. Captures the command template, explains the emitted artifacts, and includes TODO markers for GPU-derived metrics/screenshots that will be filled in after the remote sweep.
- [**HISSO_Logging_GPU_Run.ipynb**](HISSO_Logging_GPU_Run.ipynb) &nbsp;[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nickm1128/psann/blob/main/notebooks/HISSO_Logging_GPU_Run.ipynb)\
  Colab-ready automation that installs the published `psann` wheel, synthesises datasets, and executes the HISSO logging CLI on CUDA to collect metrics, event logs, and checkpoints for the final GPU sweep.

## How to run

1. Launch the notebook in Google Colab (GPU runtime recommended) or a local environment with a recent CUDA-ready PyTorch install.
2. Execute the setup cell to install the published `psann` package (`pip install psann`)—cloning the repo is optional.
3. Provide access to the datasets referenced by the experiment plan (mount Drive in Colab or point `DATA_ROOT` to a local directory).
4. Review and adjust the configuration cell (`GLOBAL_CONFIG`, toggles) so the run fits your available time and GPU budget.
5. Enable the heavier training sections only when you are ready for 30–45 minutes of runtime; the lightweight diagnostics finish in ~10 minutes on T4.

## Logging directories

- Recommended locations: `runs/hisso/` for local shells; `/content/hisso_logs/` on Colab/Runpod.
- Pass `--output-dir` to the HISSO logging CLI (`python -m psann.scripts.hisso_log_run`) to control where metrics, checkpoints, and events are written.

## Notebook hygiene

- Outputs are cleared before commit to keep diffs small.
- `.ipynb_checkpoints/` folders remain ignored repo-wide via `.gitignore`.
- When adding new notebooks, please update this README with a short description, expected runtime, and execution guidance.
