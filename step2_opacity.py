"""
Step 2 — Smooth opacity probability α (deterministic, no sampling yet).

The CUDA kernel turns each ray's in-plane hit point into a signed perpendicular
distance d to the nearest triangle edge, then α = sigmoid(d / σ).

Validates the kernel against a NumPy reference on:
  (a) a perpendicular sweep across an edge  — α must follow sigmoid(d/σ) and the
      10–90% transition width must be ln(81)·σ ≈ 4.394·σ
  (b) the edge-biased sampling distribution  — elementwise α match vs CPU (N≈1e5)

Run on any CUDA machine (Colab T4 is fine). macOS fails at load_extension.
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import torch

from step0_cpu_reference import ray_triangle_intersect, edge_biased_sample

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SIGMA = 0.01

V0 = np.array([0.0, 0.0, 0.0])
V1 = np.array([1.0, 0.0, 0.0])
V2 = np.array([0.0, 1.0, 0.0])


def _load_extension():
    from torch.utils.cpp_extension import load
    return load(
        name="diffsoup_step2",
        sources=[os.path.join(THIS_DIR, "step2_cuda_opacity.cu")],
        verbose=True,
    )


# ---------------------------------------------------------------------------
# CPU reference for the opacity window — mirrors the kernel exactly.
# ---------------------------------------------------------------------------

def opacity_reference(origins, dirs, v0, v1, v2, sigma):
    """Returns (alpha, sdist, bary) with the same convention as the kernel.

    Misses the plane (parallel / behind) -> alpha 0, sdist & bary NaN.
    """
    origins = np.asarray(origins, dtype=np.float64)
    dirs = np.asarray(dirs, dtype=np.float64)

    hit_plane, t, bary = _intersect_plane(origins, dirs, v0, v1, v2)

    w, u, v = bary[:, 0], bary[:, 1], bary[:, 2]

    # |n| = 2*Area, altitude_i = |n| / L_i, d_i = bary_i * altitude_i
    n = np.cross(v1 - v0, v2 - v0)
    two_area = np.linalg.norm(n)
    L0 = np.linalg.norm(v2 - v1)     # edge opposite V0
    L1 = np.linalg.norm(v2 - v0)     # edge opposite V1
    L2 = np.linalg.norm(v1 - v0)     # edge opposite V2

    d0 = w * (two_area / L0)
    d1 = u * (two_area / L1)
    d2 = v * (two_area / L2)
    d = np.minimum(np.minimum(d0, d1), d2)

    alpha = _sigmoid(d / sigma)

    nan = np.full(origins.shape[0], np.nan)
    alpha = np.where(hit_plane, alpha, 0.0)
    sdist = np.where(hit_plane, d, nan)
    bary_out = np.where(hit_plane[:, None], bary, np.nan)
    return alpha, sdist, bary_out


def _intersect_plane(origins, dirs, v0, v1, v2, eps=1e-8):
    """Like step0 but does NOT clamp to inside the triangle; only needs t>eps."""
    e1, e2 = v1 - v0, v2 - v0
    pvec = np.cross(dirs, e2)
    det = pvec @ e1
    parallel = np.abs(det) < eps
    inv_det = np.where(parallel, 0.0, 1.0 / np.where(parallel, 1.0, det))
    tvec = origins - v0
    u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
    qvec = np.cross(tvec, e1)
    v = np.einsum("ij,ij->i", dirs, qvec) * inv_det
    t = (qvec @ e2) * inv_det
    w = 1.0 - u - v
    hit_plane = (~parallel) & (t > eps)
    bary = np.stack([w, u, v], axis=-1)
    return hit_plane, t, bary


def _sigmoid(x):
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


# ---------------------------------------------------------------------------
# CUDA call
# ---------------------------------------------------------------------------

def cuda_opacity(ext, origins_np, dirs_np, sigma, dtype=torch.float32):
    origins = torch.as_tensor(origins_np, dtype=dtype, device="cuda")
    dirs = torch.as_tensor(dirs_np, dtype=dtype, device="cuda")
    v0 = torch.as_tensor(V0, dtype=dtype, device="cuda")
    v1 = torch.as_tensor(V1, dtype=dtype, device="cuda")
    v2 = torch.as_tensor(V2, dtype=dtype, device="cuda")
    alpha, sdist, bary = ext.opacity_forward(origins, dirs, v0, v1, v2, float(sigma))
    torch.cuda.synchronize()
    return alpha.cpu().numpy(), sdist.cpu().numpy(), bary.cpu().numpy()


# ---------------------------------------------------------------------------
# (a) Perpendicular sweep across edge V0V1 (the x-axis edge, y = 0).
#     A point at (x0, y, 0) inside has perpendicular distance d = y to that edge.
#     Sweep y from -5σ to +5σ; α should trace sigmoid(y/σ).
# ---------------------------------------------------------------------------

def test_sweep(ext, sigma=SIGMA, n=801):
    x0 = 0.3                                   # safely between the two other edges
    ys = np.linspace(-5 * sigma, 5 * sigma, n)
    pts = np.stack([np.full(n, x0), ys, np.zeros(n)], axis=-1)
    origins = pts + np.array([0.0, 0.0, 1.0])
    dirs = np.tile(np.array([0.0, 0.0, -1.0]), (n, 1))

    alpha_g, sdist_g, _ = cuda_opacity(ext, origins, dirs, sigma)
    alpha_ref = _sigmoid(ys / sigma)            # exact: d == y on this sweep

    max_err = np.nanmax(np.abs(alpha_g - alpha_ref))

    # Empirical 10–90% transition width should equal ln(81)·σ.
    expected_width = math.log(81.0) * sigma
    y_lo = np.interp(0.10, alpha_g, ys)
    y_hi = np.interp(0.90, alpha_g, ys)
    width = y_hi - y_lo
    width_err = abs(width - expected_width)

    a_center = float(np.interp(0.0, ys, alpha_g))
    monotone = bool(np.all(np.diff(alpha_g) >= -1e-7))

    print(f"\n(a) perpendicular sweep across edge V0V1  (σ={sigma}, N={n})")
    print(f"  α vs sigmoid(y/σ) max abs err : {max_err:.3e}   (tol 2e-3)  "
          f"{'OK' if max_err < 2e-3 else 'FAIL'}")
    print(f"  α on the edge (y=0)           : {a_center:.5f}   (expect 0.5)")
    print(f"  α deep inside (+5σ)           : {alpha_g[-1]:.5f}   α outside (−5σ): {alpha_g[0]:.5f}")
    print(f"  10–90% band width             : {width:.5f}   expect {expected_width:.5f}  "
          f"(err {width_err:.2e})  {'OK' if width_err < 0.1 * sigma else 'FAIL'}")
    print(f"  monotonically increasing      : {monotone}")

    ok = (max_err < 2e-3) and (abs(a_center - 0.5) < 1e-2) \
        and (width_err < 0.1 * sigma) and monotone
    return ok


# ---------------------------------------------------------------------------
# (b) Edge-biased distribution: elementwise α match vs CPU reference.
# ---------------------------------------------------------------------------

def test_against_cpu(ext, sigma=SIGMA, N=100_000, seed=0, atol=2e-3):
    origins, dirs = edge_biased_sample(N, seed=seed)

    alpha_c, sdist_c, _ = opacity_reference(origins, dirs, V0, V1, V2, sigma)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    alpha_g, sdist_g, _ = cuda_opacity(ext, origins, dirs, sigma)
    elapsed = (time.perf_counter() - t0) * 1000.0

    a_err = np.abs(alpha_c - alpha_g)
    s_err = np.abs(np.nan_to_num(sdist_c) - np.nan_to_num(sdist_g))
    a_ok = a_err.max() < atol
    s_ok = s_err.max() < 1e-4

    print(f"\n(b) edge-biased dist vs CPU  (N={N}, GPU {elapsed:.1f} ms)")
    print(f"  α    max abs err : {a_err.max():.3e}   (tol {atol:.0e})  {'OK' if a_ok else 'FAIL'}")
    print(f"  sdist max abs err: {s_err.max():.3e}   (tol 1e-4)  {'OK' if s_ok else 'FAIL'}")
    print(f"  α mean           : {alpha_g.mean():.4f}   frac(α<0.5 → near/outside edge): "
          f"{(alpha_g < 0.5).mean():.4f}")
    return a_ok and s_ok


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA is not available — step 2 requires an NVIDIA GPU.", file=sys.stderr)
        print(f"  torch={torch.__version__}  platform={sys.platform}", file=sys.stderr)
        return 2

    print(f"torch {torch.__version__}  /  device: {torch.cuda.get_device_name(0)}")
    print("compiling step2_cuda_opacity.cu ...")
    ext = _load_extension()
    print("compiled. running tests.")

    ok_a = test_sweep(ext)
    N = int(os.environ.get("DIFFSOUP_N", "100000"))
    ok_b = test_against_cpu(ext, N=N)

    print()
    if ok_a and ok_b:
        print("STEP 2 OK — opacity window is smooth, σ-calibrated, and matches CPU reference.")
        return 0
    print("STEP 2 FAILED — see above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
