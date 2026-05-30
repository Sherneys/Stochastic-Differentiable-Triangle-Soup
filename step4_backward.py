"""
Steps 4–6 — Backward pass, gradient correctness, and race-condition check.

Step 4: opacity_backward accumulates dL/dV = Σ_i grad_alpha_i · dα_i/dV into a 9-float buffer
        with atomicAdd (all N rays hit the same 3 vertices ⇒ maximal contention).

Step 5 (gradient check): with L = Σ_i α_i(V), compare the kernel's analytic ∂L/∂V against
        central finite differences of the GPU forward. Done in float64 so there is NO
        Monte-Carlo noise (gradient flows through the smooth α, not the stochastic sample),
        and FD/analytic should agree to ~1e-5 relative.

Step 6 (race check): run the float32 backward K times on identical input. atomicAdd makes the
        sum correct but FP add-ordering varies run to run — that jitter must be tiny and the
        mean must match a float64 reference. A real race would show large, unstable swings.

Run on any CUDA machine (Colab T4). macOS fails at load_extension.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

from step0_cpu_reference import edge_biased_sample

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SIGMA = 0.01

V0 = [0.0, 0.0, 0.0]
V1 = [1.0, 0.0, 0.0]
V2 = [0.0, 1.0, 0.0]


def _load_extension():
    from torch.utils.cpp_extension import load
    return load(
        name="diffsoup_step4",
        sources=[os.path.join(THIS_DIR, "step4_cuda_backward.cu")],
        verbose=True,
    )


def _verts(dtype):
    return (torch.tensor(V0, dtype=dtype, device="cuda"),
            torch.tensor(V1, dtype=dtype, device="cuda"),
            torch.tensor(V2, dtype=dtype, device="cuda"))


# ---------------------------------------------------------------------------
# Step 5 — finite-difference gradient check (deterministic, float64)
# ---------------------------------------------------------------------------

def test_gradient(ext, N=200_000, sigma=SIGMA, eps=1e-6, seed=0):
    dtype = torch.float64
    origins_np, dirs_np = edge_biased_sample(N, seed=seed)
    origins = torch.as_tensor(origins_np, dtype=dtype, device="cuda")
    dirs = torch.as_tensor(dirs_np, dtype=dtype, device="cuda")
    v0, v1, v2 = _verts(dtype)
    verts = [v0, v1, v2]

    def S(vs):
        return ext.opacity_forward(origins, dirs, vs[0], vs[1], vs[2], float(sigma)).sum().item()

    # analytic dL/dV with L = Σ α  ->  grad_alpha = 1
    grad_alpha = torch.ones(N, dtype=dtype, device="cuda")
    analytic = ext.opacity_backward(origins, dirs, v0, v1, v2, float(sigma), grad_alpha)
    analytic = analytic.cpu().numpy()           # (3,3)

    fd = np.zeros((3, 3))
    for vi in range(3):
        for c in range(2):                       # x,y only (in-plane); z grad is 0 by construction
            base = verts[vi].clone()
            vp = base.clone(); vp[c] += eps
            vm = base.clone(); vm[c] -= eps
            vs_p = list(verts); vs_p[vi] = vp
            vs_m = list(verts); vs_m[vi] = vm
            fd[vi, c] = (S(vs_p) - S(vs_m)) / (2 * eps)

    scale = max(np.abs(fd).max(), 1e-12)
    abs_err = np.abs(analytic[:, :2] - fd[:, :2])
    rel = abs_err.max() / scale
    z_zero = np.abs(analytic[:, 2]).max()

    print(f"\n(step 5) gradient check  (N={N}, float64, ε={eps:g})")
    print(f"  L = Σ_i α_i,  grad shape (3 verts × xyz)")
    names = ["V0", "V1", "V2"]
    print(f"  {'':4}{'analytic ∂L/∂x':>16}{'FD ∂L/∂x':>14}{'analytic ∂L/∂y':>16}{'FD ∂L/∂y':>14}")
    for vi in range(3):
        print(f"  {names[vi]:4}{analytic[vi,0]:16.4f}{fd[vi,0]:14.4f}"
              f"{analytic[vi,1]:16.4f}{fd[vi,1]:14.4f}")
    print(f"  max abs err (x,y)         : {abs_err.max():.3e}")
    print(f"  max relative err          : {rel:.3e}   (tol 1e-4)  {'OK' if rel < 1e-4 else 'FAIL'}")
    print(f"  z-gradient (expect 0)     : {z_zero:.3e}  {'OK' if z_zero < 1e-12 else 'FAIL'}")
    return (rel < 1e-4) and (z_zero < 1e-12)


# ---------------------------------------------------------------------------
# Step 6 — race / determinism check (float32 atomics)
# ---------------------------------------------------------------------------

def test_race(ext, N=1_000_000, sigma=SIGMA, K=50, seed=0):
    origins_np, dirs_np = edge_biased_sample(N, seed=seed)

    # float64 reference (atomic FP jitter ~1e-12 → effectively ground truth)
    o64 = torch.as_tensor(origins_np, dtype=torch.float64, device="cuda")
    d64 = torch.as_tensor(dirs_np, dtype=torch.float64, device="cuda")
    v0d, v1d, v2d = _verts(torch.float64)
    g64 = ext.opacity_backward(o64, d64, v0d, v1d, v2d, float(sigma),
                               torch.ones(N, dtype=torch.float64, device="cuda")).cpu().numpy()

    o32 = torch.as_tensor(origins_np, dtype=torch.float32, device="cuda")
    d32 = torch.as_tensor(dirs_np, dtype=torch.float32, device="cuda")
    v0f, v1f, v2f = _verts(torch.float32)
    ones32 = torch.ones(N, dtype=torch.float32, device="cuda")

    runs = []
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(K):
        g = ext.opacity_backward(o32, d32, v0f, v1f, v2f, float(sigma), ones32)
        torch.cuda.synchronize()
        runs.append(g.cpu().numpy())
    elapsed = (time.perf_counter() - t0) * 1000.0 / K
    runs = np.stack(runs)                         # (K, 3, 3)

    spread = runs.max(0) - runs.min(0)            # run-to-run swing per component
    mag = np.abs(g64) + 1e-8
    rel_spread = (spread / mag).max()
    rel_bias = (np.abs(runs.mean(0) - g64) / mag).max()
    bit_exact = bool(np.all(runs.max(0) == runs.min(0)))

    print(f"\n(step 6) race / determinism  (N={N}, {K} runs, float32, {elapsed:.1f} ms/run)")
    print(f"  run-to-run spread (rel)   : {rel_spread:.3e}")
    print(f"  bias vs float64 ref (rel) : {rel_bias:.3e}")
    print(f"  bit-exact across runs     : {bit_exact}  "
          f"(False is normal — FP add-ordering, not a race)")
    # No race ⇔ jitter is at FP-ordering scale (small) AND mean matches the f64 reference.
    no_race = (rel_spread < 1e-3) and (rel_bias < 1e-3)
    print(f"  verdict                   : {'NO RACE (jitter is FP-ordering only)' if no_race else 'POSSIBLE RACE — investigate'}"
          f"  {'OK' if no_race else 'FAIL'}")
    return no_race


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA is not available — steps 4–6 require an NVIDIA GPU.", file=sys.stderr)
        print(f"  torch={torch.__version__}  platform={sys.platform}", file=sys.stderr)
        return 2

    print(f"torch {torch.__version__}  /  device: {torch.cuda.get_device_name(0)}")
    print("compiling step4_cuda_backward.cu ...")
    ext = _load_extension()
    print("compiled. running tests.")

    ok5 = test_gradient(ext)
    N = int(os.environ.get("DIFFSOUP_N", "1000000"))
    ok6 = test_race(ext, N=N)

    print()
    if ok5 and ok6:
        print("STEPS 4–6 OK — gradient matches finite differences and atomics are race-free.")
        return 0
    print("STEPS 4–6 FAILED — see above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
