# Release-Candidate Warning Register

Status: Blocking policy for the `1.1.0rc1` workplace certification suite

Last reviewed: 2026-08-10

The six-scenario certification command runs with `PYTHONWARNINGS=error`. The accepted
warning count is therefore **zero** on both CPU and CUDA:

```bash
PYTHONWARNINGS=error python -m psann.platform.certification \
  --output reports/certification/cpu \
  --device cpu
```

Any new warning is a release blocker until it is:

1. fixed at the source;
2. converted to a deliberate typed error/capability result; or
3. documented below with a narrow message/category filter, owner, upstream issue,
   and expiry release.

## Current dispositions

| Surface | Disposition | Reason |
| --- | --- | --- |
| Six workplace certification scenarios | Zero emitted warnings; enforced as errors | Release evidence must be unambiguous |
| Explicit-context SHAP | Typed `ExplanationCapabilityError`, tested | This is a capability boundary, not a degraded success |
| Registered custom backbone export | Experimental artifact capability metadata | Reduced support is machine-visible, not warning-only |
| Unavailable release device | Hard failure under `fallback_policy="error"` | CPU/CUDA certification cannot silently fall back |
| Legacy pickle checkpoint loading | Accepted `LegacyCheckpointWarning` outside certification | Operator must explicitly acknowledge trusted deserialization |
| Third-party SWIG/import deprecations seen in the broader suite | Accepted outside certification, upstream-owned | They do not occur in the installed-wheel certification command |
| FastAPI 0.112 / Starlette status-name deprecation | Accepted with one exact message/category import-only filter | The supported FastAPI floor imports a deprecated status alias; PSANN neither calls nor exposes it, and current FastAPI removes the warning |

No broad warning suppression is configured for the certification module. The one
accepted filter matches the exact FastAPI-floor/Starlette status message and module.
Any other upstream warning fails the release workflow and this register must be
updated only after investigation.
