# Syncing xarray-sql to the `datafusion` fork (54)

The canonical FFI `simplify`-forwarding change (issue #197) lives in the fork:

> `alxmrs/datafusion` @ `claude/xarray-sql-grad-patch-0hdbqm`
> (datafusion 54, arrow 59; `datafusion/ffi/src/udf/mod.rs`)

datafusion-ffi **54 migrated `abi_stable` → `stabby`** (`SString`/`SVec`/
`FFI_Result`/`sresult!`), so the 54 patch is a rewrite of the 52 one in
`datafusion-ffi-simplify.patch`, not a port. The fork is the source of truth;
the 52 patch here is kept only as the design reference. The fork's wire format
matches the 52 design: count-prefixed datafusion-proto `Expr` bytes for the
args, a tag byte (`1` Simplified + expr bytes / `0` Original) for the result,
and a `TaskContext` built from `all_default_functions()` for
`Expr::from_bytes_with_ctx` on both sides.

## How it's wired (this branch)

The fork is pinned as a **git submodule** and the whole datafusion family is
patched to it:

- `.gitmodules` → `third_party/datafusion` tracks branch
  `claude/xarray-sql-grad-patch-0hdbqm` (shallow), at commit `b3fa8a5`.
- `Cargo.toml`:
  ```toml
  [patch.crates-io]
  datafusion           = { path = "third_party/datafusion/datafusion/core" }
  datafusion-ffi       = { path = "third_party/datafusion/datafusion/ffi" }
  datafusion-proto     = { path = "third_party/datafusion/datafusion/proto" }
  datafusion-substrait = { path = "third_party/datafusion/datafusion/substrait" }
  ```
  The whole family is patched (not just `datafusion-ffi`) so a single source
  resolves consistently — the fork bumped `dashmap`/`once_cell`/arrow, so a
  single-crate patch produced duplicate `datafusion-catalog`/arrow instances.
- `arrow = "59.0.0"` and `pyo3 = "0.28"` in `Cargo.toml` to match the fork
  (the fork is datafusion 54 on arrow 59, ahead of crates.io's arrow-58.3).

### Building it

```bash
git submodule update --init --depth 1 third_party/datafusion
cargo test --lib    # 21 tests, incl. simplify_poc + proto_abi_poc
```

CI must check out submodules (`actions/checkout` with `submodules: recursive`);
GitHub runners can fetch the public fork normally.

## Why a submodule + path patch (not a cargo git patch)

A `[patch.crates-io] datafusion-ffi = { git = "…/alxmrs/datafusion", branch }`
is cleaner, **but** this development session's cargo cannot fetch it: git's
`insteadOf` routes github.com through an xarray-sql-scoped credential proxy that
403s other repos, and cargo honors that rewrite even with the global gitconfig
disabled. The submodule's working tree is fetched once via the general egress
proxy, after which the **path** patch needs no git fetch — so the build and a
committable `Cargo.lock` are both reproducible here. (On a fork-reachable
environment the git-patch form is equivalent; switch if preferred.)

## Validation status (Rust side, green on the fork)

`cargo test --lib` (21 tests, incl. the recursive-CTE `simplify_poc` and the
`proto_abi_poc` round-trips), `cargo clippy --all-features --tests -- -D
warnings`, and `cargo fmt --check` all pass building against the fork's
`datafusion-ffi`.

## Still required for the end-to-end feature

The submodule makes our **cdylib** build against the patched ffi. To run
`grad()` inside a query from Python, two pieces remain:

1. **A `datafusion-python` 54 wheel built against this same forked
   `datafusion-ffi`** (task #13) — the Python `datafusion` package and our
   cdylib must both carry the new vtable slot. PyPI has no datafusion 54.
2. **Register `grad`/`jvp`/`vjp` as FFI `ScalarUDF`s** from the cdylib whose
   `simplify` is the `differentiate` engine (matching the fork's wire format),
   then delete the Substrait bridge and rewrite `benchmarks/grad_descent.py` as
   a single `SELECT` with `grad(...)` inside the recursive CTE (task #14).
