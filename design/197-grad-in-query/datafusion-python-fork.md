# datafusion-python fork patch (for `grad()` in queries, #197)

To run `grad()` inside a SQL query from Python, the `datafusion` Python package
must be built against the **same** patched `datafusion-ffi` as our cdylib — both
binaries have to carry the new `simplify` vtable slot. That means a custom
`datafusion-python` wheel built from a fork.

- Fork: `alxmrs/datafusion-python`
- Apply: `datafusion-python-fork.patch` (fills the existing `[patch.crates-io]`
  placeholder in the root `Cargo.toml`, pointing every datafusion-* dependency
  at the `alxmrs/datafusion` fork branch `claude/xarray-sql-grad-patch-0hdbqm`).

```bash
cd datafusion-python        # your fork
git apply path/to/datafusion-python-fork.patch
maturin build --release     # produces the wheel xarray-sql installs
```

datafusion-python `main` is already on **datafusion 54 / pyo3 0.28**, and its
root `Cargo.toml` even ships an empty `[patch.crates-io]` placeholder for exactly
this. So in the clean case the patch is *only* that placeholder fill — no source
changes.

## Important: base the datafusion fork on the 54.0.0 *release*, not main

This patch assumes the `alxmrs/datafusion` fork's `simplify` commit sits on the
**datafusion `54.0.0` release tag**. As currently pushed, the fork looks to be
based on datafusion **main**, which has drifted from the 54.0.0 release that
datafusion-python `main` targets. Building datafusion-python against the
main-based fork fails to compile the wheel crate (`crates/core`) with ~9 API
mismatches, e.g.:

- `datafusion::sql::TableReference` import moved
- `PhysicalExtensionCodec::try_decode/try_encode` gained a parameter
- `LogicalPlan::RecursiveQuery` gained a `schema` field
- `From<Box<CreateExternalTable>>` / `From<Box<CreateFunction>>` shape changes
- a non-exhaustive `match` on the `dml` plan enum

…plus arrow **58→59** and the `datafusion-spark` `core` feature now being
non-default. These are upstream-version differences, **not** the simplify
change. Porting datafusion-python to datafusion-main is a real effort and out of
scope for this feature.

**Recommendation:** rebase the one `simplify` commit onto the `54.0.0` tag:

```bash
# in the alxmrs/datafusion fork
git rebase --onto 54.0.0 <commit-before-simplify> claude/xarray-sql-grad-patch-0hdbqm
# (the simplify change is additive; expect a clean replay)
```

With the fork on 54.0.0, both downstream patches become trivial:

- **datafusion-python:** just this `[patch.crates-io]` fill (no arrow bump, no
  spark feature) — verified to resolve cleanly; the wheel crate then builds
  because it's the combination datafusion-python main already supports.
- **xarray-sql:** the `third_party/datafusion` submodule re-points to the
  rebased commit, and `Cargo.toml` reverts `arrow` 59 → 58.3 (matching the
  54.0.0 release). The 21 Rust tests already pass against the fork's ffi.

## What was verified here

- The patch **resolves** cleanly against the fork (`cargo generate-lockfile`,
  414 packages, no version conflicts) once `datafusion-spark`'s `core` feature
  is enabled / arrow aligned — i.e. dependency wiring is correct.
- The remaining build failures are all datafusion main-vs-54.0.0 API drift in
  datafusion-python's own code, which the rebase above removes.
