# DiffSoup — Stochastic Differentiable Triangle (Challenge 1)

A CUDA implementation of a **differentiable** triangle using *stochastic opacity masking* — shoot rays at a triangle, turn the distance-to-edge into an opacity probability, make a stochastic decision, then push gradients back to update the vertex positions on the GPU without any race condition.

> Challenge from **Umetani Lab**, based on **DiffSoup** (Tojo, Bickel, Umetani — CVPR 2026)

```
α = sigmoid(d / σ)   →   sample ~ Bernoulli(α)   →   ∂L/∂V  (atomicAdd)
```

---

## TL;DR — what problem does this solve?

Standard rendering tests "is this point inside the triangle?" as a 0/1 answer (a step function) → the derivative is 0 almost everywhere → **you can't train it with gradient descent**.

The fix: soften the edge with `α = sigmoid(d/σ)` (smooth, differentiable), then **sample** `Bernoulli(α)` so the image stays crisp — while `E[sample] = α` lets the gradient flow through the smooth `α` (a cousin of the score-function / REINFORCE estimator).

For a beginner-friendly, section-by-section walkthrough see **`DiffSoup_อธิบายโค้ด.pdf`** (in Thai).

---

## Test case

| Parameter | Value |
|---|---|
| Triangle | `V₀(0,0,0)`, `V₁(1,0,0)`, `V₂(0,1,0)` (on the z=0 plane) |
| Smoothness | `σ = 0.01` |
| Rays | `N = 1,000,000` (biased toward the **edges** — where the gradient lives) |
| Output | correct `∂L/∂V₀, ∂L/∂V₁, ∂L/∂V₂`, race-free |

---

## File layout

The project is built **incrementally and verifiably** — every step is checked against a CPU reference before moving on.

| File | Role |
|---|---|
| `step0_cpu_reference.py` | Möller–Trumbore on CPU (NumPy) — the **ground truth** + edge-biased ray generator |
| `step1_cuda_forward.cu` / `step1_forward.py` | **Forward pass**: ray-triangle intersection, 1 ray per thread |
| `step2_cuda_opacity.cu` / `step2_opacity.py` | **Opacity window**: `α = sigmoid(signed_edge_distance / σ)` |
| `step3_cuda_stochastic.cu` / `step3_stochastic.py` | **Stochastic decision**: `Bernoulli(α)` via Philox RNG |
| `step4_cuda_backward.cu` / `step4_backward.py` | **Backward pass**: `dL/dV` via `atomicAdd` + gradient/race checks |
| `build_colab.py` | Bundles all sources into `step1_colab.ipynb` for Colab |
| `step1_colab.ipynb` | Ready-to-run notebook (generated) |
| `challenge1_plan.md` | Plan + pitfalls to watch out for (Thai) |
| `DiffSoup_อธิบายโค้ด.pdf` | Section-by-section code walkthrough (Thai) |

---

## Running it

> ⚠️ Requires an **NVIDIA GPU** — macOS won't work (it fails at `load_extension`). Use Google Colab (free T4), a lab machine, or any CUDA-capable box.

### Option A — Google Colab (recommended)

```bash
# Generate the notebook from the latest sources
python build_colab.py        # → step1_colab.ipynb
```

Upload `step1_colab.ipynb` to Colab → **Runtime → Change runtime type → GPU** → Run all.
(The notebook writes every source file into the working dir via `%%writefile`, then runs each step's tests for you.)

### Option B — a CUDA-capable machine

```bash
pip install torch numpy ninja      # ninja is required by torch's JIT load

python step0_cpu_reference.py      # CPU sanity (runs anywhere)
python step1_forward.py            # Forward vs CPU
python step2_opacity.py            # Opacity window
python step3_stochastic.py         # Stochastic + RNG
python step4_backward.py           # Backward + gradient + race checks
```

Control the ray count with an env var (some steps default to 1e5 for fast iteration):

```bash
DIFFSOUP_N=1000000 python step1_forward.py     # full plan-spec sweep
```

The first run of each `.cu` compiles for ~30–60 s (JIT); subsequent runs are cached.

---

## Pipeline + per-step verification

```
Step 0  CPU reference ────────────────► 12 hand-verifiable cases pass
Step 1  Forward (Möller–Trumbore) ────► bary/t match CPU within tolerance
Step 2  α = sigmoid(d/σ) ──────────────► soft edge, 10–90% band = ln(81)·σ ≈ 4.39σ
Step 3  Bernoulli(α) + Philox ─────────► mean → α (Monte Carlo), reproducible
Step 4  Backward + atomicAdd ──────────► (step 5) matches finite differences
                                         (step 6) jitter is FP-ordering, not a race
```

### Key pitfall (from `challenge1_plan.md`)

`atomicAdd` on `float` is **inherently non-deterministic** — the floating-point add order varies run to run, so results aren't bit-exact. **This is not a race condition** (the sum is always correct). Step 6 is designed to tell the two apart:
- small spread + mean matches the float64 reference → **FP ordering (normal)**
- large/erratic swings → **a real race (bug)**

---

## Core concepts to understand

1. **Möller–Trumbore** — recovers barycentric `(w,u,v)` and `t` of the hit; `min(w,u,v)→0` means you're on an edge.
2. **Stochastic opacity masking** — why sampling makes the *expected* gradient non-zero.
3. **Splitting forward/backward** — forward samples (crisp image), backward differentiates the smooth expectation `E[sample]=α(V)`.
4. **atomicAdd** — all rays share the same 3 vertices ⇒ heavy contention ⇒ atomics are needed to avoid a race.
5. **Philox RNG** — counter-based; `(seed, ray_index)` → independent, reproducible streams for 10⁶ threads.

---

## Rebuild the explainer PDF (optional)

```bash
python _pdfbuild/make.py     # → DiffSoup_อธิบายโค้ด.pdf
```

Uses HTML + headless Chrome with Google fonts (Google Sans / Noto Sans Thai / Google Sans Code) embedded as base64 (needs Google Chrome installed + internet at build time).

---

## References

- DiffSoup project page — https://kenji-tojo.github.io/publications/diffsoup/
- GitHub (official code) — https://github.com/kenji-tojo/diffsoup
- Tojo, Bickel, Umetani — *DiffSoup*, CVPR 2026
