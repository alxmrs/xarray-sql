# `grad()` inside any query — FFI `simplify` forwarding (#197)

Today `grad()`/`jvp()`/`vjp()` are rewritten by round-tripping the whole logical
plan through Substrait (see `xarray_sql/sql.py` and `grad_rewrite` in
`src/lib.rs`). Substrait can't represent every plan shape, so `grad()` can't
appear inside **recursive CTEs**, **DML**, or **scalar subqueries** — which is
what blocks a fully-declarative training loop with `grad` *in* the query
(tracked in [#197](https://github.com/alxmrs/xarray-sql/issues/197)).

The fix (Option 1 in #197): make `grad` a **self-rewriting UDF**. If a foreign
(FFI-registered) scalar UDF could implement `simplify`, `grad` would
differentiate its argument during the optimizer's `SimplifyExpressions` pass —
which runs for *every* plan shape, with no Substrait involved. That lets us
delete the Substrait bridge entirely and write the OLS loop as one `SELECT`.

The one upstream gap: `datafusion-ffi`'s `FFI_ScalarUDF` / `ForeignScalarUDF`
forward `name`, `signature`, `return_field_from_args`, `coerce_types`, and
`invoke_with_args` — but **not `simplify`**, so foreign UDFs get the default
(identity) and can't rewrite themselves.

## What's in here

- `datafusion-ffi-simplify.patch` — the surgical change to `datafusion-ffi`
  (`datafusion/ffi/src/udf/mod.rs`) that adds a `simplify` slot to the
  `FFI_ScalarUDF` vtable and forwards it. Generated against **datafusion-ffi
  52.2.0**.

## How the patch works

`simplify(Vec<Expr>, &dyn SimplifyInfo)` can't cross the C ABI as Rust types, so
the argument `Expr`s travel as **datafusion-proto bytes** (already a dependency
of `datafusion-ffi`):

1. **Consumer** (`ForeignScalarUDF::simplify`, in datafusion-python's process)
   encodes its `args` with `encode_exprs` and calls the new vtable fn pointer.
2. **Provider** (`simplify_fn_wrapper`, in our cdylib) decodes the args, calls
   the underlying `ScalarUDF::simplify`, and encodes the result
   (`Simplified(expr)` as tag `1` + expr bytes, `Original` as tag `0`).
3. **Consumer** decodes the result back into an `ExprSimplifyResult`.

Resolving a deserialized `Expr` that references built-in functions (`sin`,
`power`, …) needs a `FunctionRegistry` populated with them, on **both** sides.
The patch builds one from `datafusion_functions::all_default_functions()`
(cached in a `OnceLock`), which is why it makes `datafusion-functions` a
non-optional dependency (see the Cargo.toml change below).

### Known limitation

`SimplifyInfo` is **not** forwarded; the provider is handed a schema-less
`SimplifyContext`. UDFs whose `simplify` consults the schema or execution props
are not supported across FFI. `grad`/`jvp`/`vjp` differentiate structurally and
don't need it, so this is fine for our use case. Forwarding `SimplifyInfo` (a
set of callbacks across the ABI) would be the follow-up needed for a general
upstream contribution.

## Cargo.toml change the patch assumes

In `datafusion/ffi/Cargo.toml`, make `datafusion-functions` a regular dependency
(it is currently optional, enabled only by the `integration-tests` feature):

```diff
-datafusion-functions = { workspace = true, optional = true }
+datafusion-functions = { workspace = true }
```

and drop `"datafusion-functions"` from the `integration-tests` feature list. The
vtable field must always be present for ABI stability, so the registry it needs
can't be feature-gated.

## Validation status (all green, in this repo + locally)

The risky assumptions were retired with tests before any fork work:

| Question | Where | Result |
| --- | --- | --- |
| Does `simplify` fire inside a recursive CTE's recursive term? | `src/autograd.rs` `simplify_poc` | ✅ grad-in-recursive-CTE OLS loop converges; the marker's `invoke` errors, so success proves simplify ran |
| Does an `Expr` survive proto bytes and stay differentiable (both directions)? | `src/autograd.rs` `proto_abi_poc` | ✅ structural identity + identical derivative |
| Does the FFI `simplify`-forwarding ABI work across the foreign boundary? | patched `datafusion-ffi` `udf::tests::test_ffi_udf_simplify_forwarding` | ✅ a foreign UDF's simplify fires; `deriv(sin(x))` → `cos(sin(x))` across the boundary |

The third test uses datafusion-ffi's own `mock_foreign_marker_id` hook to force
the foreign code path (in-process the FFI is otherwise bypassed).

## Integration path (remaining work — #197)

Both the Python `datafusion` package **and** our cdylib must see the new vtable
slot, so both must build against the patched `datafusion-ffi`.

1. **Land the patch in a fork.** The change is in `apache/datafusion`
   (`datafusion/ffi`), not `datafusion-python`. Either fork `apache/datafusion`
   and apply the patch, or vendor the patched `datafusion-ffi` into the
   `datafusion-python` fork.
2. **Pin it as a git submodule** of this repo and point Cargo at it:
   ```toml
   [patch.crates-io]
   datafusion-ffi = { path = "third_party/datafusion/datafusion/ffi" }
   ```
   A submodule (vs. a vendored copy or loose patch) pins an exact upstream
   commit and keeps the fork's history separate — the stable home for this.
3. **Rebuild the `datafusion-python` wheel** against the patched
   `datafusion-ffi` (`[patch.crates-io]` + `maturin build`), and have
   xarray-sql depend on that wheel.
4. **Register `grad`/`jvp`/`vjp` as FFI UDFs** exported from our cdylib, with
   `simplify` = the `differentiate` engine in `src/autograd.rs`. Then **delete
   the Substrait bridge** (`grad_rewrite`, `_sql_with_autograd`,
   `_table_schemas`, the `protoc` CI steps) and rewrite `benchmarks/grad_descent.py`
   as a single declarative `SELECT` with `grad(...)` inside the recursive CTE.

Steps 1–3 need an environment that can reach the fork; they can't run in the
xarray-sql-scoped session where the patch was developed.
