# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "xarray_sql",
#   "xarray",
#   "numpy",
# ]
#
# [tool.uv.sources]
# xarray_sql = { path = "..", editable = true }
# ///
"""Gradient descent as a single declarative SQL query.

Fits a line ``y ~= a*x + b`` by minimising the mean squared error — with the
**entire training loop expressed as one recursive CTE**, no Python iteration and
no precompiled update rule. ``grad(...)`` lives *inside* the recursion:

    WITH RECURSIVE params(step, a, b) AS (
      SELECT 0, 0.0, 0.0
      UNION ALL
      SELECT params.step + 1,
             params.a - lr * AVG(grad(loss, a)),
             params.b - lr * AVG(grad(loss, b))
      FROM params CROSS JOIN d
      WHERE params.step < STEPS
      GROUP BY params.step, params.a, params.b)
    SELECT step, a, b FROM params ORDER BY step

Each recursion appends the next generation, descending along the gradient that
``grad`` computes from the loss formula directly. ``AVG(grad(loss, a))`` is the
relational ``d/da (Σ loss) / N`` — differentiation through the aggregate is just
linearity. So gradient, update, and iteration are all one declarative query; the
optimisation trajectory is the rows of that query.

``grad`` is differentiated as a SQL source-to-source rewrite *before* the query
is planned, so the marker works inside the recursive CTE (and any other query
shape) with no Substrait round-trip. The loss is written once, as ordinary SQL,
and the engine differentiates it symbolically — the relational equivalent of
``jax.vmap(jax.grad(f))``, since each row is an independent evaluation point.

Run standalone:

    uv run benchmarks/grad_descent.py
"""

from __future__ import annotations

import numpy as np
import xarray as xr

import xarray_sql as xql

# Per-row loss r^2 with residual r = y - (a*x + b). The columns a, b come from
# the recursive `params` relation; x, y come from the data table `d`.
RESIDUAL = "(y - (a * x + b))"
LOSS = f"{RESIDUAL} * {RESIDUAL}"
LR = 0.4
STEPS = 200


def main() -> None:
    rng = np.random.default_rng(0)
    n = 500
    x = rng.uniform(0.0, 1.0, n)
    a_true, b_true = 2.0, -1.0
    y = a_true * x + b_true + rng.normal(0.0, 0.01, n)

    ctx = xql.XarrayContext()
    ctx.from_dataset(
        "d",
        xr.Dataset(
            {"x": (("i",), x), "y": (("i",), y)}, coords={"i": np.arange(n)}
        ),
        chunks={"i": n},
    )

    # The entire training loop is one declarative recursive query: each step
    # appends the next generation, descending along the gradient that grad()
    # computes from the loss — differentiated inside the recursion itself.
    trajectory = ctx.sql(
        f"""
        WITH RECURSIVE params(step, a, b) AS (
          SELECT 0 AS step, CAST(0.0 AS DOUBLE) AS a, CAST(0.0 AS DOUBLE) AS b
          UNION ALL
          SELECT params.step + 1                        AS step,
                 params.a - {LR} * AVG(grad({LOSS}, a))  AS a,
                 params.b - {LR} * AVG(grad({LOSS}, b))  AS b
          FROM params CROSS JOIN d
          WHERE params.step < {STEPS}
          GROUP BY params.step, params.a, params.b
        )
        SELECT step, a, b FROM params ORDER BY step
        """
    ).to_pandas()

    print("trajectory (every 40th generation):")
    print(trajectory.iloc[::40].to_string(index=False))

    a, b = float(trajectory["a"].iloc[-1]), float(trajectory["b"].iloc[-1])
    a_ols, b_ols = np.polyfit(x, y, 1)
    print(
        f"\nSQL gradient descent:  a={a:.4f}  b={b:.4f}  ({len(trajectory)} generations)"
    )
    print(f"least-squares (numpy): a={a_ols:.4f}  b={b_ols:.4f}")
    assert abs(a - a_ols) < 1e-2 and abs(b - b_ols) < 1e-2
    print(
        "\nOK: a single recursive-CTE query with grad() inside fit the line "
        "to the OLS solution."
    )


if __name__ == "__main__":
    main()
