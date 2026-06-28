# DataFusion 52 → 54 migration notes (grad-ffi branch)

This branch (`claude/xarray-sql-grad-ffi-73ovqq`) is bumped to **datafusion 54**
to align with upstreaming the FFI `simplify` patch and with Claude Code mobile's
proxy. The core/main feature branch stays on 52 (it ships against the PyPI
`datafusion` wheel, which tops out at 53 — there is no 54 on PyPI yet).

## Dependency bumps (`Cargo.toml`)

| crate | 52 | 54 |
| --- | --- | --- |
| `datafusion`, `-ffi`, `-proto`, `-substrait` | `52.0.0` | `54.0.0` |
| `arrow` | `57.2.0` | `58.3.0` (datafusion 54 requires `^58.3`) |
| `pyo3`, `pyo3-build-config` | `0.26` | `0.28` (arrow 58's `pyarrow` feature needs pyo3 `^0.28`) |

## Source API changes fixed

1. **`ScalarUDFImpl::as_any` / `TableProvider::as_any` removed from the traits.**
   Delete the `fn as_any(&self) -> &dyn Any` overrides (and the now-unused
   `std::any::Any` imports). Affected: `MarkerUdf` and the `SimplifyingGrad`
   test UDF in `src/autograd.rs`; `PrunableStreamingTable` in `src/lib.rs`.

2. **`Cast` now stores the target type as a `FieldRef`, not a `DataType`.**
   `cast.data_type` → `cast.field.data_type().clone()`. `Cast::new(expr, dt)`
   still takes a `DataType`. (`src/autograd.rs` `linearize` Cast arm.)

3. **`simplify` signature: `&dyn SimplifyInfo` → `&SimplifyContext`.** The
   `SimplifyInfo` trait is gone; `simplify` now takes the concrete
   `SimplifyContext` (itself reworked: built via `SimplifyContext::builder()` /
   `Default`, no longer `new(&props)`). Import
   `datafusion::logical_expr::simplify::{ExprSimplifyResult, SimplifyContext}`.

4. **datafusion-proto: `Expr::from_bytes_with_registry(&dyn FunctionRegistry)`
   → `Expr::from_bytes_with_ctx(&TaskContext)`.** Build a context with the
   default functions via `SessionContext::new().task_ctx()`. (`to_bytes` is
   unchanged.)

5. **pyo3 0.28 (`src/lib.rs` logical-extension-codec capsule extraction):**
   - `Bound::downcast::<T>()` is deprecated → `Bound::cast::<T>()` (drop-in,
     returns `&Bound<T>`).
   - `PyCapsule::name()` returns a `CapsuleName`; read it with the `unsafe`
     `as_cstr()` if you need the string.
   - `PyCapsule::reference::<T>()` is deprecated → `pointer_checked(Some(name))`
     returns the validated payload pointer; cast it to `&T` yourself.

## Implications for the `datafusion-ffi` simplify patch

The patch in `datafusion-ffi-simplify.patch` was written against **52.2.0** and
must be regenerated against **54** (in progress in the fork). The 54-relevant
deltas to carry over:

- The forwarded `simplify` signature on `ForeignScalarUDF` and the provider
  wrapper must use `&SimplifyContext` (not `&dyn SimplifyInfo`). Because
  `SimplifyContext` is concrete, "not forwarding it" now means constructing a
  default `SimplifyContext` on the provider side rather than a schema-less one.
- Deserialization uses `from_bytes_with_ctx(&TaskContext)` instead of
  `from_bytes_with_registry`; the provider/consumer build a `TaskContext` from
  the default functions rather than a hand-rolled `FunctionRegistry`. This is
  simpler than the 52 version (no custom registry struct needed).

## Validation status on 54 (Rust side)

`cargo test --lib` (21 tests, incl. `simplify_poc` recursive-CTE and
`proto_abi_poc`), `cargo clippy --all-features --tests -- -D warnings`, and
`cargo fmt --check` are all green on datafusion 54. The Python integration path
(pytest, `maturin develop`) is blocked until a datafusion-python 54 wheel exists
from the fork — same blocker as the FFI work itself.
