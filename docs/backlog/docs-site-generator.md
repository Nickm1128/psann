# Docs Site Generator Evaluation

Status: Backlog &mdash; revisit once the HISSO refactor stabilises.

## Motivation

- Centralise API, tutorials, and benchmark write-ups with navigation/search.
- Publish notebook-backed guides without checking large outputs into git.
- Automate deployment (e.g., GitHub Pages) as part of the release pipeline.

## Current blockers

- The documentation set is still in flux while HISSO failures are triaged; a
  static site today would churn weekly.
- No single source of truth for configuration snippets yet (README, docs/*.md,
  and notebooks all overlap).
- Need to validate whether MkDocs or Sphinx can coexist with the Hatch build
  excludes already trimming docs from distributions.

## Recommendation

- Revisit after the HISSO regression fixes land and the API stabilises.
- Capture structured doc requirements (API reference, tutorials, gallery) ahead
  of time to choose between MkDocs Material and Sphinx.
- When ready, add a dedicated `docs` extras group (mkdocs-material, mkdocstrings)
  and ensure CI builds the site for smoke-testing before publishing.
