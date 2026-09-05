# Deprecation policy for 0.x

The canonical 0.13.0 task/configuration API is authoritative for documentation, examples, wildcard exports, and new code. Direct legacy regression constructors, flat LSM/HISSO inputs, lowercase LM APIs, old base names, CLI shims, and supported checkpoint migration routes remain usable throughout the remaining 0.x line. This release does not perform a breaking 1.x removal.

Compatibility facades normalize once into the canonical runtime and emit one caller-located `DeprecationWarning` at the specified entry boundary. Python hides many deprecation warnings by default; use `-W default` while migrating. Do not issue a warning per batch, parameter, or checkpoint reconstruction stage. Direct import availability is preserved; construction/use is the warning boundary where specified.

Legacy names are excluded from canonical wildcard surfaces and CLI help. They may be documented in [migration](migration.md), which provides old-to-new mappings and checkpoint limitations. Conflicting old/new representations and unsupported policies reject early instead of silently changing runtime behavior.

Any future removal needs a separately documented breaking-release decision. The historical `v1.0.0` Git tag does not end this 0.x compatibility commitment and is unchanged. Current package metadata is 0.13.0; release preparation does not imply publication.
