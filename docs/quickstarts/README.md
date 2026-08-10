# Workplace Quick Starts

These task-oriented guides use the stable `workplace-v1` API frozen for the
`1.1.0rc1` candidate:

- [Regression](regression.md): named tabular inputs, target scaling, validation, and a
  safe artifact.
- [Classification](classification.md): binary probabilities and thresholds plus
  multiclass top-k output.
- [Deployment](deployment.md): bounded batch inference and the optional HTTP service.
- [Resume](resume.md): restart deterministic training from a `.psann-train`
  checkpoint.
- [SHAP](shap.md): explain the deployed raw-input contract with an explicit
  background.

Install the combined quick-start environment with:

```bash
python -m pip install "psann[sklearn,serve,explain]" pandas
```

The base package does not install pandas, FastAPI, or SHAP. Add only the extras used
by your workload.
