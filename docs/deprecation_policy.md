# Deprecation policy for 2.x

The canonical 2.0 task/configuration API is the sole authoritative interface for documentation, examples, wildcard exports, CLI help, and new code. Direct legacy regression constructors, flat LSM/HISSO inputs, lowercase LM APIs, old base names, CLI shims, and supported checkpoint readers remain available only as migration compatibility routes during 2.x. They are not alternative public APIs.

Compatibility facades normalize once into the canonical runtime and emit one caller-located `DeprecationWarning` at the specified entry boundary. Python hides many deprecation warnings by default; use `-W default` while migrating. Do not issue a warning per batch, parameter, or checkpoint reconstruction stage. Direct import availability is preserved; construction/use is the warning boundary where specified.

Regression facade `set_params`, `sklearn.clone`, and supported checkpoint loading reconstruct internally without repeating the constructor warning. A new direct constructor call still emits its own warning at the caller.

Legacy names are excluded from canonical wildcard surfaces and CLI help. They may be documented in [migration](migration.md), which provides old-to-new mappings and checkpoint limitations. Conflicting old/new representations and unsupported policies reject early instead of silently changing runtime behavior.

Removing a compatibility route requires a separately documented breaking-release decision. No compatibility route is guaranteed beyond the 2.x line. The historical `v1.0.0` Git tag remains unchanged; package version 2.0.0 starts the canonical release track described here.
