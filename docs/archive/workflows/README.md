# Archived GitHub Workflows

Status: Archive

Archived: 2026-08-11

These workflow definitions are retained for historical reference outside
`.github/workflows/`. GitHub Actions does not discover or execute them, and they are
not current release, support, or promotion evidence.

| Archived definition | Last observed problem | Historical run |
| --- | --- | --- |
| `release-certification.yml` | The workflow failed before creating any jobs. | [Run 31435720525](https://github.com/Nickm1128/psann/actions/runs/31435720525) |
| `security.yml` | The dependency audit found `setuptools` 79.0.1 affected by PYSEC-2026-3447; the fixed version is 83.0.0. | [Run 31435572293](https://github.com/Nickm1128/psann/actions/runs/31435572293) |
| `hisso-benchmark.yml` | The benchmark comparison disagreed on Windows-style versus POSIX-style baseline paths. | [Run 31435572239](https://github.com/Nickm1128/psann/actions/runs/31435572239) |

The repository owner chose to archive these workflows rather than repair them. The
ordinary pull-request CI, accelerator, performance, and tag-gated container workflows
remain active. Local certification and security commands may still be used for
development, but they do not replace independent evidence for an exact pushed commit.

Consequently, `1.1.0rc1` is not promotable and must not be tagged or published to
PyPI on the basis of these archived definitions. Restoring any workflow requires
repairing it, moving it back under `.github/workflows/`, restoring its repository
contract tests and documentation, and obtaining a green run for the exact candidate
commit.
