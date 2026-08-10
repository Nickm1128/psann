# Release Process

This is the maintained release procedure for the `psann` and `psannlm`
distributions.

## Preconditions

1. Work from a clean checkout of the intended release commit.
2. Confirm the release is covered by `docs/support_policy.md`.
3. Update `CHANGELOG.md`, `docs/migration.md`, and any changed compatibility claims.
4. Install the development environment:

   ```bash
   python -m pip install -e .[dev,sklearn]
   ```

5. Run the blocking gates:

   ```bash
   python tools/quality.py lint
   python tools/check_public_api.py
   python -m pytest -m "not slow and not gpu"
   python tools/run_coverage.py
   python tools/repo_hygiene_audit.py --strict-long-files
   ```

6. Obtain recent scheduled accelerator evidence for any release that advertises CUDA
   support.
7. Review the current dependency audit, container high/critical scan, and source/image
   SBOM artifacts.
8. Run the workplace benchmark comparison. Correctness is blocking; any performance
   warning must be investigated, rebaselined intentionally, or recorded as accepted.

For the workplace 1.1 candidate, dispatch **Workplace release-candidate
certification** for the exact commit. It repeats the clean source gates, builds both
distributions, runs all six scenarios from the installed core wheel with warnings as
errors from a temporary working directory outside the checkout (preventing source or
generated-report shadowing), verifies the retained public `0.12.7` trusted migration
fixture and current native schema migrations, executes representative CPU and CUDA
soaks, and reuses the supply-chain and reference-container workflows. Download and
review every uploaded evidence artifact.

Release branches named `codex/release-*` run the same certification automatically on
pull requests so a new workflow can be proven before it first reaches the default
branch. After merge, maintainers may also dispatch it manually with explicit version
and soak inputs. Neither path tags or publishes a package.

Normal pull-request CI additionally builds and installs the candidate wheel on Linux
and Windows for Python 3.11, 3.12, and 3.13, including sklearn, service, and SHAP
extras. Separate Python 3.11 jobs consume the maintained core floor, workstation
snapshot, SHAP floor, and unconstrained-current profiles. A release is not promotable
if any advertised platform/minor/dependency profile is skipped or allowed to fail.

Scoped coverage follows [`quality_policy.md`](quality_policy.md): core 70%, PSANN-LM
Alpha 35%, and the release helper 60% are blocking; aggregate scripts coverage is
observational. The current and public `0.12.7` API manifests must both pass before
building.

## Versioning

The selected release identity is `1.1.0rc1` for both distributions and
`v1.1.0rc1` for the candidate Git/GitHub tag. General availability is reserved as
`1.1.0` / `v1.1.0`. The historical `v1.0.0` tag is permanent and must never be moved,
recreated, or reused. [`release_identity.md`](release_identity.md) is the naming
authority for package, workflow-artifact, container, SBOM, and release-note surfaces.

`src/psann/_version.py` and `psannlm/_version.py` are the package-owned version
sources. The release helper synchronizes them for coordinated releases and rejects
drift. At runtime, `psann.__version__` reports the core distribution version and
`psannlm.__version__` reports the LM distribution version; neither aliases the other.

The `1.1` PSANN-LM package declares and enforces `psann>=1.1.0rc1,<1.2`. Any future
release outside that band requires an intentional metadata and runtime-compatibility
update before the release helper will proceed.

Preview a bump without modifying files:

```bash
python scripts/release.py --part patch --dry-run
```

Prepare and build without uploading:

```bash
python scripts/release.py --version 1.1.0rc1 --skip-upload
```

For an offline local rehearsal only:

```bash
python scripts/release.py --version 1.1.0rc1 --skip-upload --skip-remote-checks
```

`--skip-remote-checks` cannot be combined with upload. The release helper otherwise
requires all of these preflights before modifying version files:

- a clean tracked and untracked Git worktree;
- synchronized core and LM version sources;
- an exact `CHANGELOG.md` release heading;
- an LM dependency band that accepts the selected core version;
- an unused local and remote `v<version>` tag;
- an unused version for both projects on PyPI.

It then synchronizes the two version sources, removes only known build outputs,
builds one wheel and one source distribution per project, verifies their identities,
runs Twine checks, and executes the installed-wheel package smoke test. `--skip-build`
reuses artifacts only when the complete artifact set already matches the selected
version.

## Artifact Verification

Before publishing:

```bash
python tools/package_smoke.py
```

Inspect both `dist/` and `psannlm/dist/` and confirm:

- one wheel and one source distribution exist for each project;
- coordinated candidate artifacts use the selected version for both distributions;
- the core wheel does not contain `psannlm`, notebooks, scripts, datasets, or reports;
- the LM wheel imports only after the core wheel is installed;
- `psann.__version__` and `psannlm.__version__` each match their own installed
  distribution metadata;
- the installed core version satisfies the LM wheel's declared compatibility band;
- artifact/model metadata contains fingerprints but no credentials or raw training
  rows;
- release packages and the reference image have retained SBOM evidence.

## Publishing

Publishing is an explicit maintainer action. Use a PyPI token from the environment or
keyring; never place it in source files, command history, model manifests, or logs.

```bash
python scripts/release.py \
  --version 1.1.0rc1 \
  --confirm-upload 1.1.0rc1
```

The exact confirmation value is mandatory and is checked after version selection but
before any release preflight or upload. The helper performs no tagging or GitHub/GHCR
publication.

After upload, install both packages from the public index in a new environment, repeat
the import smoke test, create the Git tag and GitHub release, and attach the changelog
plus compatibility evidence.

The certification workflow never tags, uploads to PyPI, or publishes a container. A
maintainer may create `v1.1.0rc1` only after its promotion job is green and the exact
commit, CPU/CUDA/Windows reports, vulnerability results, container smoke, API-freeze
result, and migration tests are linked in the release record. The tag must point to
that reviewed commit and may never be moved. Promote to `1.1.0` / `v1.1.0` only after
the candidate evidence and public-install checks are accepted; never reuse the
historical `v1.0.0` tag.
