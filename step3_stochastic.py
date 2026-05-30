"""
Step 3 — Stochastic opacity decision.

Each ray draws ξ ~ U(0,1] (Philox) and emits a Bernoulli sample opaque ⇔ (ξ < α).
The mean over many samples at a fixed point must converge to α — that is the whole
point: E[sample] = α is smooth in V, so it carries a usable gradient (step 4), while a
hard inside/outside test would not.

Validates the kernel on:
  (a) convergence  — replicate a point M times; |mean − α| within Monte-Carlo error,
                     for several target α values
  (b) reproducibility — same seed ⇒ bit-identical samples; different seed ⇒ differs,
                        but the mean is unchanged within MC error (= no race / correct streams)
  (c) smoothness   — sweep across an edge, binned mean traces sigmoid(d/σ); full 10^6 sweep

Run on any CUDA machine (Colab T4). macOS fails at load_extension.
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import torch

from step0_cpu_reference import edge_biased_sample

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SIGMA = 0.01

V0 = np.array([0.0, 0.0, 0.0])
V1 = np.array([1.0, 0.0, 0.0])
V2 = np.array([0.0, 1.0, 0.0])


def _load_extension():
    from torch.utils.cpp_extension import load
    return load(
        name="diffsoup_step3",
        sources=[os.path.join(THIS_DIR, "step3_cuda_stochastic.cu")],
        verbose=True,
    )


def cuda_stochastic(ext, origins_np, dirs_np, sigma, seed, dtype=torch.float32):
    origins = torch.as_tensor(origins_np, dtype=dtype, device="cuda")
    dirs = torch.as_tensor(dirs_np, dtype=dtype, device="cuda")
    v0 = torch.as_tensor(V0, dtype=dtype, device="cuda")
    v1 = torch.as_tensor(V1, dtype=dtype, device="cuda")
    v2 = torch.as_tensor(V2, dtype=dtype, device="cuda")
    alpha, sample = ext.stochastic_forward(origins, dirs, v0, v1, v2, float(sigma), int(seed))
    torch.cuda.synchronize()
    return alpha.cpu().numpy(), sample.cpu().numpy()


def _rays_at_perp_distance(d_values, x0=0.3):
    """Rays that hit edge V0V1 (y=0) at signed perpendicular distance d (= y here)."""
    d_values = np.asarray(d_values, dtype=np.float64)
    pts = np.stack([np.full(d_values.shape, x0), d_values, np.zeros_like(d_values)], axis=-1)
    origins = pts + np.array([0.0, 0.0, 1.0])
    dirs = np.tile(np.array([0.0, 0.0, -1.0]), (d_values.shape[0], 1))
    return origins, dirs


# ---------------------------------------------------------------------------
# (a) convergence of the sample mean to α
# ---------------------------------------------------------------------------

def test_convergence(ext, sigma=SIGMA, M=200_000, seed=1234):
    targets = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    # d such that sigmoid(d/σ) = target
    d = sigma * np.log(targets / (1.0 - targets))

    print(f"\n(a) convergence  (M={M} samples per target, σ={sigma})")
    print(f"  {'target α':>9} {'kernel α':>9} {'mean':>9} {'|mean−α|':>10} {'4σ_MC':>9}  result")
    all_ok = True
    for tgt, dd in zip(targets, d):
        origins, dirs = _rays_at_perp_distance(np.full(M, dd))
        alpha_g, sample_g = cuda_stochastic(ext, origins, dirs, sigma, seed)
        a = float(alpha_g[0])
        mean = float(sample_g.mean())
        mc = 4.0 * math.sqrt(a * (1.0 - a) / M)   # ~4 std-errors band
        ok = abs(mean - a) < mc
        all_ok = all_ok and ok
        print(f"  {tgt:9.2f} {a:9.4f} {mean:9.4f} {abs(mean-a):10.2e} {mc:9.2e}  "
              f"{'OK' if ok else 'FAIL'}")
    return all_ok


# ---------------------------------------------------------------------------
# (b) reproducibility / independent streams
# ---------------------------------------------------------------------------

def test_reproducibility(ext, sigma=SIGMA, M=200_000):
    origins, dirs = _rays_at_perp_distance(np.zeros(M))   # all on the edge, α=0.5
    a1, s1 = cuda_stochastic(ext, origins, dirs, sigma, seed=7)
    a2, s2 = cuda_stochastic(ext, origins, dirs, sigma, seed=7)    # same seed
    a3, s3 = cuda_stochastic(ext, origins, dirs, sigma, seed=99)   # different seed

    same_seed_identical = np.array_equal(s1, s2)
    frac_diff = float((s1 != s3).mean())
    # independent fair coins differ ~50% of the time
    streams_ok = 0.45 < frac_diff < 0.55
    means = (s1.mean(), s3.mean())
    means_ok = abs(means[0] - 0.5) < 5e-3 and abs(means[1] - 0.5) < 5e-3

    print(f"\n(b) reproducibility  (M={M} rays on the edge, α=0.5)")
    print(f"  same seed → bit-identical samples : {same_seed_identical}  "
          f"{'OK' if same_seed_identical else 'FAIL'}")
    print(f"  diff seed → fraction differing     : {frac_diff:.4f}   (expect ~0.50)  "
          f"{'OK' if streams_ok else 'FAIL'}")
    print(f"  means (seed7, seed99)              : {means[0]:.4f}, {means[1]:.4f}   "
          f"{'OK' if means_ok else 'FAIL'}")
    return same_seed_identical and streams_ok and means_ok


# ---------------------------------------------------------------------------
# (c) smoothness — binned sample mean across the edge follows sigmoid(d/σ)
# ---------------------------------------------------------------------------

def test_smoothness(ext, sigma=SIGMA, N=1_000_000, seed=2024):
    # Spread N rays uniformly in y over [-5σ, 5σ] crossing edge V0V1.
    ys = np.linspace(-5 * sigma, 5 * sigma, N)
    origins, dirs = _rays_at_perp_distance(ys)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    alpha_g, sample_g = cuda_stochastic(ext, origins, dirs, sigma, seed)
    elapsed = (time.perf_counter() - t0) * 1000.0

    # Bin by y and compare bin mean vs sigmoid(y/σ).
    nb = 40
    edges = np.linspace(-5 * sigma, 5 * sigma, nb + 1)
    idx = np.clip(np.digitize(ys, edges) - 1, 0, nb - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts = np.bincount(idx, minlength=nb)
    sums = np.bincount(idx, weights=sample_g, minlength=nb)
    bin_mean = sums / np.maximum(counts, 1)
    bin_ref = 1.0 / (1.0 + np.exp(-centers / sigma))

    valid = counts > 100
    rmse = float(np.sqrt(np.mean((bin_mean[valid] - bin_ref[valid]) ** 2)))
    # monotone (allowing MC wiggle)
    smoothed = bin_mean[valid]
    monotone_ish = float(np.mean(np.diff(smoothed) >= -0.05)) > 0.9

    print(f"\n(c) smoothness sweep  (N={N}, {nb} bins, GPU {elapsed:.1f} ms)")
    print(f"  binned mean vs sigmoid  RMSE : {rmse:.4f}   (tol 0.02)  "
          f"{'OK' if rmse < 0.02 else 'FAIL'}")
    print(f"  bin mean @ edge (y≈0)        : "
          f"{bin_mean[nb//2]:.3f} / {bin_mean[nb//2-1]:.3f}   (expect ~0.5)")
    print(f"  overall sample rate          : {sample_g.mean():.4f}   "
          f"mean α: {alpha_g.mean():.4f}  (should match within MC)")
    print(f"  roughly monotone increasing  : {monotone_ish}")
    return (rmse < 0.02) and monotone_ish


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA is not available — step 3 requires an NVIDIA GPU.", file=sys.stderr)
        print(f"  torch={torch.__version__}  platform={sys.platform}", file=sys.stderr)
        return 2

    print(f"torch {torch.__version__}  /  device: {torch.cuda.get_device_name(0)}")
    print("compiling step3_cuda_stochastic.cu ...")
    ext = _load_extension()
    print("compiled. running tests.")

    ok_a = test_convergence(ext)
    ok_b = test_reproducibility(ext)
    N = int(os.environ.get("DIFFSOUP_N", "1000000"))
    ok_c = test_smoothness(ext, N=N)

    print()
    if ok_a and ok_b and ok_c:
        print("STEP 3 OK — stochastic samples converge to α, streams are independent & reproducible.")
        return 0
    print("STEP 3 FAILED — see above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
