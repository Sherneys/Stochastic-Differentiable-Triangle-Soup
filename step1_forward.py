"""
Step 1 — Deterministic forward pass on CUDA.

Loads the .cu kernel via torch.utils.cpp_extension.load (JIT, no setup.py),
then validates it against the CPU reference from step 0 on:
  (a) the 12 hand-verifiable cases
  (b) the edge-biased sampling distribution (N = 1e5 by default, scale to 1e6 for the full check)

Run on any CUDA-equipped machine (laptop GPU, lab box, Colab). macOS will fail at load_extension.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

from step0_cpu_reference import (
    ray_triangle_intersect,
    edge_distance_bary,
    edge_biased_sample,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_extension():
    from torch.utils.cpp_extension import load
    return load(
        name="diffsoup_step1",
        sources=[os.path.join(THIS_DIR, "step1_cuda_forward.cu")],
        verbose=True,
    )


def cuda_intersect(ext, origins_np, dirs_np, V0, V1, V2, dtype=torch.float32):
    origins = torch.as_tensor(origins_np, dtype=dtype, device="cuda")
    dirs    = torch.as_tensor(dirs_np,    dtype=dtype, device="cuda")
    v0 = torch.as_tensor(V0, dtype=dtype, device="cuda")
    v1 = torch.as_tensor(V1, dtype=dtype, device="cuda")
    v2 = torch.as_tensor(V2, dtype=dtype, device="cuda")
    hit, t, bary = ext.intersect_forward(origins, dirs, v0, v1, v2)
    torch.cuda.synchronize()
    return hit.cpu().numpy(), t.cpu().numpy(), bary.cpu().numpy()


# ---------------------------------------------------------------------------
# (a) 12 hand-verifiable cases — same as step 0
# ---------------------------------------------------------------------------

def test_hand_cases(ext) -> bool:
    V0 = np.array([0.0, 0.0, 0.0])
    V1 = np.array([1.0, 0.0, 0.0])
    V2 = np.array([0.0, 1.0, 0.0])

    cases = [
        ("centroid hit",           [1/3, 1/3, +1.0],           [0, 0, -1], True,  [1/3, 1/3, 1/3]),
        ("near vertex V0",         [0.0, 0.0, +1.0],           [0, 0, -1], True,  [1.0, 0.0, 0.0]),
        ("near vertex V1",         [1.0, 0.0, +1.0],           [0, 0, -1], True,  [0.0, 1.0, 0.0]),
        ("near vertex V2",         [0.0, 1.0, +1.0],           [0, 0, -1], True,  [0.0, 0.0, 1.0]),
        ("edge midpoint V1V2",     [0.5, 0.5, +1.0],           [0, 0, -1], True,  [0.0, 0.5, 0.5]),
        ("just inside edge V1V2",  [0.495, 0.495, +1.0],       [0, 0, -1], True,  [0.010, 0.495, 0.495]),
        ("just outside edge V1V2", [0.51, 0.51, +1.0],         [0, 0, -1], False, None),
        ("outside (x<0)",          [-0.01, 0.5, +1.0],         [0, 0, -1], False, None),
        ("outside (y<0)",          [0.5, -0.01, +1.0],         [0, 0, -1], False, None),
        ("parallel to plane",      [0.3, 0.3, +1.0],           [1, 0,  0], False, None),
        ("ray points away",        [0.3, 0.3, +1.0],           [0, 0, +1], False, None),
        ("oblique hit (centroid)", [1/3 + 1.0, 1/3 + 1.0, 1.0], [-1, -1, -1], True, [1/3, 1/3, 1/3]),
    ]
    origins = np.array([c[1] for c in cases], dtype=np.float64)
    dirs    = np.array([c[2] for c in cases], dtype=np.float64)

    hit_g, t_g, bary_g = cuda_intersect(ext, origins, dirs, V0, V1, V2, dtype=torch.float32)

    header = f"{'#':>2}  {'case':<26} {'hit':>5}  {'t':>9}  {'bary (w,u,v)':<28}  result"
    print(header)
    print("-" * len(header))

    all_ok = True
    for i, (name, _, _, exp_hit, exp_bary) in enumerate(cases):
        ok = bool(hit_g[i]) == exp_hit
        if exp_hit and exp_bary is not None:
            ok = ok and np.allclose(bary_g[i], exp_bary, atol=1e-5)
        t_str = f"{t_g[i]:9.5f}" if hit_g[i] else "    —    "
        bs = "[" + ", ".join(f"{c:+.4f}" if not np.isnan(c) else "  nan " for c in bary_g[i]) + "]"
        flag = "OK " if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"{i:>2}  {name:<26} {str(bool(hit_g[i])):>5}  {t_str}  {bs:<28}  {flag}")

    print()
    print(("ALL 12 CASES PASSED" if all_ok else "SOME CASES FAILED") + " (CUDA)")
    return all_ok


# ---------------------------------------------------------------------------
# (b) Edge-biased sampling vs CPU reference
# ---------------------------------------------------------------------------

def test_against_cpu(ext, N: int = 100_000, seed: int = 0,
                     atol_bary: float = 1e-5, atol_t: float = 1e-4) -> bool:
    V0 = np.array([0.0, 0.0, 0.0])
    V1 = np.array([1.0, 0.0, 0.0])
    V2 = np.array([0.0, 1.0, 0.0])

    origins, dirs = edge_biased_sample(N, seed=seed)

    # CPU ground truth (float64)
    hit_c, t_c, bary_c = ray_triangle_intersect(origins, dirs, V0, V1, V2)

    # GPU under test (float32 — kernel default)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    hit_g, t_g, bary_g = cuda_intersect(ext, origins, dirs, V0, V1, V2, dtype=torch.float32)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(f"\nedge-biased sampling: N={N}  (GPU intersect + copy back: {elapsed_ms:.1f} ms)")

    hit_match = np.array_equal(hit_c, hit_g)
    print(f"  hit flags match exactly      : {hit_match}")

    # Where both agree it's a hit, compare numerics.
    both_hit = hit_c & hit_g
    nb = int(both_hit.sum())
    if nb == 0:
        print("  no overlapping hits to compare — abort")
        return False

    bary_err = np.abs(bary_c[both_hit] - bary_g[both_hit])
    t_err = np.abs(t_c[both_hit] - t_g[both_hit])
    bary_ok = bary_err.max() < atol_bary
    t_ok    = t_err.max()    < atol_t

    print(f"  bary max abs err (N={nb})    : {bary_err.max():.3e}   (tol {atol_bary:.0e})  {'OK' if bary_ok else 'FAIL'}")
    print(f"  t    max abs err             : {t_err.max():.3e}   (tol {atol_t:.0e})  {'OK' if t_ok else 'FAIL'}")

    edge_d = edge_distance_bary(bary_g[hit_g])
    print(f"  GPU hit rate                 : {hit_g.mean():.4f}")
    print(f"  GPU mean min(bary)           : {edge_d.mean():.5f}   (smaller = nearer edges)")
    print(f"  GPU fraction within 0.02     : {(edge_d < 0.02).mean():.4f}")

    return hit_match and bary_ok and t_ok


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA is not available in this environment — step 1 requires an NVIDIA GPU.", file=sys.stderr)
        print(f"  torch={torch.__version__}  platform={sys.platform}", file=sys.stderr)
        return 2

    print(f"torch {torch.__version__}  /  device: {torch.cuda.get_device_name(0)}")
    print("compiling step1_cuda_forward.cu ...")
    ext = _load_extension()
    print("compiled. running tests.\n")

    ok1 = test_hand_cases(ext)
    # Bump N to 1_000_000 for the full plan-spec sweep; 100k keeps iteration fast.
    N = int(os.environ.get("DIFFSOUP_N", "100000"))
    ok2 = test_against_cpu(ext, N=N)

    print()
    if ok1 and ok2:
        print("STEP 1 OK — CUDA forward pass matches CPU reference within tolerance.")
        return 0
    print("STEP 1 FAILED — see above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
