# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "xarray-sql",
#   "xarray",
#   "numpy",
#   "scipy",
# ]
# ///
"""Regridding — interpolation to a new grid is a sparse matmul, i.e. a JOIN.

Regridding (resampling a field from one grid onto another) is the operation we
most associate with the *array* paradigm — xESMF/ESMF, ``apply_ufunc``,
``.interp()``. But every linear regridding scheme (bilinear, conservative,
nearest) is mathematically a **sparse matrix–vector product**: each output cell
is a weighted sum of a few input cells. And a sparse matrix is just a table of
``(dst_id, src_id, weight)`` rows. So *applying* a regridding is::

    SELECT w.dst_id, SUM(s.value * w.weight) AS regridded
    FROM   weights w JOIN src s ON s.cell_id = w.src_id
    GROUP BY w.dst_id

— a JOIN against the weight table plus a weighted GROUP BY. This is the most
relational the "array" paradigm ever gets: the operation we reach for xESMF to
do is a join.

**Where the array paradigm still earns its keep:** *generating* the weights is
the genuinely geometric part (cell overlaps, interpolation stencils, spherical
coordinates). Here we build bilinear weights with a few lines of numpy; for
conservative remapping on real grids you would let ESMF/xESMF compute them once
and hand the resulting sparse matrix to SQL as a table. SQL *applies* the
weights; it does not invent the geometry.

We validate the SQL matmul against xarray's own bilinear ``.interp()``.
No network; fully synthetic.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

import xarray_sql as xql

from _harness import assert_grid_close, run_case, show_sql, timed


def _linear_weights(
    src: np.ndarray, dst: np.ndarray
) -> list[tuple[int, int, float]]:
    """1-D linear-interpolation weights: (dst_index, src_index, weight) triples.

    Each target point falls between two source points and borrows from both,
    with weights summing to 1 — the 1-D building block of bilinear regridding.
    """
    triples = []
    for t, x in enumerate(dst):
        i = int(np.clip(np.searchsorted(src, x) - 1, 0, len(src) - 2))
        span = src[i + 1] - src[i]
        hi = (x - src[i]) / span
        triples.append((t, i, 1.0 - hi))
        triples.append((t, i + 1, hi))
    return triples


def _bilinear_weight_table(
    slat: np.ndarray, slon: np.ndarray, tlat: np.ndarray, tlon: np.ndarray
) -> xr.Dataset:
    """Build the sparse bilinear weight matrix as a (dst_id, src_id, weight) table.

    The 2-D weight is the outer product of the 1-D lat and lon weights; src_id
    and dst_id are row-major flattenings of the source and target grids.
    """
    nslon, ntlon = len(slon), len(tlon)
    lat_w = _linear_weights(slat, tlat)
    lon_w = _linear_weights(slon, tlon)
    dst_ids, src_ids, weights = [], [], []
    for tj, si, wlat in lat_w:
        for tk, sj, wlon in lon_w:
            dst_ids.append(tj * ntlon + tk)
            src_ids.append(si * nslon + sj)
            weights.append(wlat * wlon)
    n = len(weights)
    return xr.Dataset(
        {
            "dst_id": (["pair"], np.array(dst_ids, dtype="int64")),
            "src_id": (["pair"], np.array(src_ids, dtype="int64")),
            "weight": (["pair"], np.array(weights, dtype="float64")),
        },
        coords={"pair": np.arange(n)},
    ).chunk({"pair": n})


def main() -> None:
    # Coarse source grid with a smooth field; finer target grid strictly inside.
    slat = np.linspace(0.0, 10.0, 11)
    slon = np.linspace(0.0, 10.0, 11)
    field = (
        np.sin(slat[:, None] / 3.0) * np.cos(slon[None, :] / 4.0)
        + 0.1 * slat[:, None]
    )
    src_da = xr.DataArray(
        field, coords={"lat": slat, "lon": slon}, dims=["lat", "lon"]
    )

    tlat = np.linspace(0.3, 9.7, 25)
    tlon = np.linspace(0.4, 9.6, 30)
    print(
        f"  regrid {len(slat)}×{len(slon)} → {len(tlat)}×{len(tlon)} (bilinear)"
    )

    # Source field as a flat (cell_id, value) table.
    src_table = xr.Dataset(
        {"value": (["cell_id"], field.ravel())},
        coords={"cell_id": np.arange(field.size)},
    ).chunk({"cell_id": field.size})
    weights = _bilinear_weight_table(slat, slon, tlat, tlon)
    print(
        f"  weight matrix: {weights.sizes['pair']:,} nonzeros "
        f"({len(tlat) * len(tlon)} targets × 4 corners)"
    )

    ctx = xql.XarrayContext()
    ctx.from_dataset("src", src_table, chunks={"cell_id": field.size})
    ctx.from_dataset("weights", weights, chunks={"pair": weights.sizes["pair"]})

    sql = """
        SELECT w.dst_id,
               SUM(s.value * w.weight) AS regridded
        FROM weights w
        JOIN src s ON s.cell_id = w.src_id
        GROUP BY w.dst_id
        ORDER BY w.dst_id
    """
    show_sql(sql)

    # The result is keyed by dst_id (row-major over the target grid); reshape
    # it back to the (lat, lon) field it represents.
    with timed("SQL regrid (weight-table JOIN + weighted SUM)"):
        flat = ctx.sql(sql).to_dataset(dims=["dst_id"]).regridded
        got = xr.DataArray(
            flat.values.reshape(len(tlat), len(tlon)),
            dims=["lat", "lon"],
            coords={"lat": tlat, "lon": tlon},
        )

    # Array reference: xarray's own bilinear interpolation onto the target grid.
    with timed("xarray .interp reference"):
        ref = src_da.interp(lat=tlat, lon=tlon, method="linear")

    assert_grid_close("bilinear regrid", got, ref, rtol=1e-9, atol=1e-9)

    print(
        f"\n  {got.size:,} target cells regridded; "
        f"value range [{float(got.min()):.3f}, {float(got.max()):.3f}]."
    )


if __name__ == "__main__":
    raise SystemExit(run_case(main, "Regridding: sparse weight-table JOIN"))
