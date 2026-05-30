"""
Step 0 — CPU reference for ray-triangle intersection (Möller–Trumbore).

Used as ground truth for the CUDA kernel in later steps.

Convention:
  - Triangle V0, V1, V2 (CCW from +z looking down at z=0 plane)
  - Ray:  P(t) = origin + t * direction,  t > 0
  - Barycentric returned as (w, u, v) where  P = w*V0 + u*V1 + v*V2,  w = 1 - u - v
  - `u` corresponds to edge V0→V1, `v` to edge V0→V2
  - Distance to nearest edge in barycentric space = min(w, u, v)
"""

from __future__ import annotations

import numpy as np

EPS = 1e-8


def ray_triangle_intersect(
    origins: np.ndarray,        # (N, 3)
    directions: np.ndarray,     # (N, 3)
    v0: np.ndarray,             # (3,)
    v1: np.ndarray,             # (3,)
    v2: np.ndarray,             # (3,)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized Möller–Trumbore.

    Returns:
      hit:   (N,) bool       — True if ray hits the triangle at t > EPS
      t:     (N,) float      — parametric distance along the ray (NaN if miss)
      bary:  (N, 3) float    — (w, u, v) barycentric coordinates (NaN if miss)
    """
    origins = np.asarray(origins, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)

    e1 = v1 - v0                          # (3,)
    e2 = v2 - v0                          # (3,)

    pvec = np.cross(directions, e2)       # (N, 3)
    det = pvec @ e1                       # (N,)

    # Parallel rays (det ≈ 0) — no unique intersection
    parallel = np.abs(det) < EPS
    inv_det = np.where(parallel, 0.0, 1.0 / np.where(parallel, 1.0, det))

    tvec = origins - v0                   # (N, 3)
    u = np.einsum("ij,ij->i", tvec, pvec) * inv_det

    qvec = np.cross(tvec, e1)             # (N, 3)
    v = np.einsum("ij,ij->i", directions, qvec) * inv_det

    t = (qvec @ e2) * inv_det
    w = 1.0 - u - v

    hit = (~parallel) & (u >= 0.0) & (v >= 0.0) & (w >= 0.0) & (t > EPS)

    nan = np.full_like(t, np.nan)
    t_out = np.where(hit, t, nan)
    bary = np.stack([w, u, v], axis=-1)
    bary_out = np.where(hit[:, None], bary, np.nan)

    return hit, t_out, bary_out


def edge_distance_bary(bary: np.ndarray) -> np.ndarray:
    """
    Per-ray distance to the nearest triangle edge in *barycentric* units.
    A value of 0 means the hit is exactly on an edge; 1/3 is the centroid.
    NaN propagates for missed rays.
    """
    return np.min(bary, axis=-1)


# ---------------------------------------------------------------------------
# Hand-verifiable tests
# ---------------------------------------------------------------------------

def _fmt(x: np.ndarray) -> str:
    if np.all(np.isnan(x)):
        return "—"
    return "[" + ", ".join(f"{c:+.4f}" if not np.isnan(c) else "  nan " for c in x) + "]"


def run_tests() -> None:
    V0 = np.array([0.0, 0.0, 0.0])
    V1 = np.array([1.0, 0.0, 0.0])
    V2 = np.array([0.0, 1.0, 0.0])

    # Each test: (name, origin, direction, expected_hit, expected_bary_or_None)
    # All rays here shoot in -z; the triangle lies on z = 0.
    cases = [
        ("centroid hit",            [1/3, 1/3, +1.0], [0, 0, -1], True,  [1/3, 1/3, 1/3]),
        ("near vertex V0",          [0.0, 0.0, +1.0], [0, 0, -1], True,  [1.0, 0.0, 0.0]),
        ("near vertex V1",          [1.0, 0.0, +1.0], [0, 0, -1], True,  [0.0, 1.0, 0.0]),
        ("near vertex V2",          [0.0, 1.0, +1.0], [0, 0, -1], True,  [0.0, 0.0, 1.0]),
        ("edge midpoint V1V2",      [0.5, 0.5, +1.0], [0, 0, -1], True,  [0.0, 0.5, 0.5]),
        ("just inside edge V1V2",   [0.495, 0.495, +1.0], [0, 0, -1], True, [0.010, 0.495, 0.495]),
        ("just outside edge V1V2",  [0.51, 0.51, +1.0], [0, 0, -1], False, None),
        ("outside (x<0)",           [-0.01, 0.5, +1.0], [0, 0, -1], False, None),
        ("outside (y<0)",           [0.5, -0.01, +1.0], [0, 0, -1], False, None),
        ("parallel to plane",       [0.3, 0.3, +1.0], [1, 0,  0], False, None),
        ("ray points away",         [0.3, 0.3, +1.0], [0, 0, +1], False, None),
        ("oblique hit (centroid)",  [1/3 + 1.0, 1/3 + 1.0, +1.0], [-1, -1, -1], True, [1/3, 1/3, 1/3]),
    ]

    origins = np.array([c[1] for c in cases], dtype=np.float64)
    dirs    = np.array([c[2] for c in cases], dtype=np.float64)

    hit, t, bary = ray_triangle_intersect(origins, dirs, V0, V1, V2)
    edge_d = edge_distance_bary(bary)

    print(f"Triangle: V0={V0.tolist()}  V1={V1.tolist()}  V2={V2.tolist()}\n")
    header = f"{'#':>2}  {'case':<26} {'hit':>5}  {'t':>8}  {'bary (w,u,v)':<28} {'min(bary)':>10}  result"
    print(header)
    print("-" * len(header))

    all_ok = True
    for i, (name, _, _, exp_hit, exp_bary) in enumerate(cases):
        ok = bool(hit[i]) == exp_hit
        if exp_hit and exp_bary is not None:
            ok = ok and np.allclose(bary[i], exp_bary, atol=1e-6)

        t_str = f"{t[i]:8.4f}" if hit[i] else "   —    "
        ed_str = f"{edge_d[i]:+.4f}" if hit[i] else "   —   "
        flag = "OK " if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"{i:>2}  {name:<26} {str(bool(hit[i])):>5}  {t_str}  {_fmt(bary[i]):<28} {ed_str:>10}  {flag}")

    print()
    print("ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED")


# ---------------------------------------------------------------------------
# Sanity check: batched edge-biased sampling (preview for step 1+)
# ---------------------------------------------------------------------------

def edge_biased_sample(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate N rays that are biased toward the triangle's edges, all shooting in -z.
    This previews the sampling distribution used by later CUDA tests (N = 10^6).
    """
    rng = np.random.default_rng(seed)
    # Sample (u, v) uniformly on the triangle, then nudge toward the closest edge
    r1 = np.sqrt(rng.random(n))
    r2 = rng.random(n)
    u = 1.0 - r1
    v = r1 * (1.0 - r2)
    w = r1 * r2
    bary = np.stack([w, u, v], axis=-1)
    # Pull each point a fraction toward its nearest edge (set the smallest coord toward 0)
    idx = np.argmin(bary, axis=-1)
    pull = 0.5  # 0 = no bias, 1 = collapse onto the edge
    bary[np.arange(n), idx] *= (1.0 - pull)
    bary /= bary.sum(axis=-1, keepdims=True)

    V0 = np.array([0.0, 0.0, 0.0])
    V1 = np.array([1.0, 0.0, 0.0])
    V2 = np.array([0.0, 1.0, 0.0])
    points = bary[:, 0:1] * V0 + bary[:, 1:2] * V1 + bary[:, 2:3] * V2  # (n, 3)

    origins = points + np.array([0.0, 0.0, 1.0])
    directions = np.tile(np.array([0.0, 0.0, -1.0]), (n, 1))
    return origins, directions


def run_sampling_preview(n: int = 100_000) -> None:
    V0 = np.array([0.0, 0.0, 0.0])
    V1 = np.array([1.0, 0.0, 0.0])
    V2 = np.array([0.0, 1.0, 0.0])
    origins, dirs = edge_biased_sample(n)
    hit, _, bary = ray_triangle_intersect(origins, dirs, V0, V1, V2)
    hit_rate = hit.mean()
    edge_d = edge_distance_bary(bary[hit])
    print(f"\nedge-biased sampling preview (N={n}):")
    print(f"  hit rate         : {hit_rate:.4f}  (expect ~1.0; rays are constructed to hit)")
    print(f"  mean min(bary)   : {edge_d.mean():.5f}  (smaller = closer to edges)")
    print(f"  median min(bary) : {np.median(edge_d):.5f}")
    print(f"  fraction within 0.02 of an edge : {(edge_d < 0.02).mean():.4f}")


if __name__ == "__main__":
    run_tests()
    run_sampling_preview()