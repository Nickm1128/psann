# Release Identity

Status: Selected for the workplace release candidate; not yet published or tagged

Last reviewed: 2026-08-10

This document is the naming authority for the workplace candidate. Package versions
use PEP 440 without a `v` prefix. Git, GitHub Release, container, and SBOM identities
use the matching `v`-prefixed tag.

## Selected identity

| Surface | Release candidate | General availability |
| --- | --- | --- |
| `psann` package | `1.1.0rc1` | `1.1.0` |
| `psannlm` package | `1.1.0rc1` | `1.1.0` |
| Git and GitHub tag | `v1.1.0rc1` | `v1.1.0` |
| GitHub Release title | `PSANN 1.1.0rc1` | `PSANN 1.1.0` |
| Serving image | `ghcr.io/nickm1128/psann/serve:v1.1.0rc1` | `ghcr.io/nickm1128/psann/serve:v1.1.0` |
| Changelog section | `1.1.0rc1` | Promote the reviewed candidate notes to `1.1.0` |

The historical `v1.0.0` Git tag is reserved permanently. It must never be moved,
deleted, recreated, or used as the tag for this release line.

The two selected package versions are coordinated but remain distribution-owned:
`psann.__version__` comes from the core wheel and `psannlm.__version__` comes from the
LM wheel. The `1.1` LM line declares and enforces `psann>=1.1.0rc1,<1.2`; an
incompatible installed core is rejected rather than silently reused.

## Availability decision

On 2026-08-10, the public indexes reported `0.12.7` as the latest release of both
`psann` and `psannlm`. Neither `1.1.0rc1` nor `1.1.0` existed on PyPI, and the Git
remote exposed no `v1.1.0rc1` or `v1.1.0` tag. Recheck immediately before publishing;
package versions and Git tags are immutable identities once published.

## Evidence and artifact names

Before a tag exists, candidate workflow artifacts use both the selected package
version and the full candidate commit SHA:

- `release-source-gates-1.1.0rc1-<sha>`;
- `release-candidate-1.1.0rc1-<sha>`;
- `release-candidate-cuda-1.1.0rc1-<sha>`;
- `psann-dependency-security-1.1.0rc1-<run-id>`;
- `psann-security-1.1.0rc1-<run-id>`.

After the reviewed commit is tagged, the container workflow derives the serving
image and release SBOM names from `v1.1.0rc1` (or `v1.1.0` for GA):

- `psann-wheel-v1.1.0rc1.spdx.json`;
- `psann-sdist-v1.1.0rc1.spdx.json`;
- `psannlm-wheel-v1.1.0rc1.spdx.json`;
- `psannlm-sdist-v1.1.0rc1.spdx.json`;
- `psann-serving-v1.1.0rc1.spdx.json`.

The native `.psann` artifact format remains `1.0`. Artifact-format and workplace API
contract versions are compatibility protocols, not package release numbers, and do
not change solely because the distribution version moves to `1.1.0rc1`.

## Promotion rule

The selected tag does not exist yet. A maintainer may create it only after every
blocking Phase 9 item is committed on the exact candidate SHA, the release-candidate
workflow is green, and its CPU, CUDA, Windows, security, container, SBOM, package,
API, and compatibility evidence has been reviewed. PyPI and GHCR publication remain
explicit maintainer actions.
