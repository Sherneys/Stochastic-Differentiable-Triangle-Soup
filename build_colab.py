"""
Build step1_colab.ipynb from the three source files in this directory.
Run once; commit/upload the resulting .ipynb to Colab.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def read(name: str) -> str:
    with open(os.path.join(HERE, name), "r", encoding="utf-8") as f:
        return f.read()


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": text,
        "outputs": [],
        "execution_count": None,
    }


def writefile_cell(filename: str, body: str) -> dict:
    # Use %%writefile -q so Colab writes the file without echoing the path.
    return code(f"%%writefile {filename}\n{body}")


def main() -> None:
    step0 = read("step0_cpu_reference.py")
    cu1   = read("step1_cuda_forward.cu")
    step1 = read("step1_forward.py")
    cu2   = read("step2_cuda_opacity.cu")
    step2 = read("step2_opacity.py")
    cu3   = read("step3_cuda_stochastic.cu")
    step3 = read("step3_stochastic.py")
    cu4   = read("step4_cuda_backward.cu")
    step4 = read("step4_backward.py")

    cells = [
        md(
            "# DiffSoup — Stochastic Differentiable Triangle (Steps 1–6)\n"
            "\n"
            "Step 1: 1 ray / thread Möller–Trumbore intersection on GPU.\n"
            "Step 2: smooth opacity α = sigmoid(signed edge distance / σ), deterministic.\n"
            "Step 3: stochastic Bernoulli(α) decision via Philox RNG; mean over samples → α.\n"
            "Steps 4–6: backward pass (dL/dV via atomicAdd), finite-difference gradient check,\n"
            "and a race / determinism check on the float32 atomics.\n"
            "All validated against the CPU reference.\n"
            "\n"
            "**Runtime → Change runtime type → GPU** (T4 is fine).\n"
            "Then run the cells top-to-bottom.\n"
        ),
        md("## 1. Environment check (installs ninja — required by torch JIT load)"),
        code(
            "!pip install -q ninja\n"
            "import sys, torch\n"
            "print('python :', sys.version.split()[0])\n"
            "print('torch  :', torch.__version__)\n"
            "print('cuda?  :', torch.cuda.is_available())\n"
            "assert torch.cuda.is_available(), 'No GPU detected — set Runtime → GPU and re-run.'\n"
            "print('device :', torch.cuda.get_device_name(0))\n"
            "!ninja --version\n"
            "!nvcc --version | tail -n2\n"
        ),
        md("## 2. Drop source files into the Colab working dir"),
        writefile_cell("step0_cpu_reference.py", step0),
        writefile_cell("step1_cuda_forward.cu", cu1),
        writefile_cell("step1_forward.py", step1),
        writefile_cell("step2_cuda_opacity.cu", cu2),
        writefile_cell("step2_opacity.py", step2),
        writefile_cell("step3_cuda_stochastic.cu", cu3),
        writefile_cell("step3_stochastic.py", step3),
        writefile_cell("step4_cuda_backward.cu", cu4),
        writefile_cell("step4_backward.py", step4),
        md("## 3. Sanity check — CPU reference"),
        code("!python step0_cpu_reference.py\n"),
        md(
            "## 4. Step 1 — JIT-compile the CUDA kernel and validate against CPU\n"
            "\n"
            "First run compiles `step1_cuda_forward.cu` (≈30–60 s); subsequent runs are cached.\n"
            "Set `DIFFSOUP_N=1000000` to do the full plan-spec sweep (1e6 rays).\n"
        ),
        code("!DIFFSOUP_N=100000 python step1_forward.py\n"),
        md("### Optional — full 10^6 ray sweep"),
        code("!DIFFSOUP_N=1000000 python step1_forward.py\n"),
        md(
            "## 5. Step 2 — Smooth opacity window α = sigmoid(d / σ)\n"
            "\n"
            "Sweeps a ray perpendicular across an edge: α must fall smoothly from ~1 (inside) "
            "through 0.5 (on the edge) to ~0 (outside), with a 10–90% band width of ln(81)·σ ≈ 4.39σ "
            "for σ=0.01. Then checks elementwise α against the CPU reference.\n"
        ),
        code("!DIFFSOUP_N=100000 python step2_opacity.py\n"),
        md(
            "## 6. Step 3 — Stochastic decision (Bernoulli sampling + Philox RNG)\n"
            "\n"
            "Each ray draws ξ ~ U(0,1] and emits opaque ⇔ (ξ < α). Checks that the sample mean "
            "converges to α, that the same seed is bit-reproducible while different seeds give "
            "independent streams (~50% differ), and that a full 10^6-ray sweep across the edge "
            "traces sigmoid(d/σ). Runs the full 1e6 sweep by default.\n"
        ),
        code("!python step3_stochastic.py\n"),
        md(
            "## 7. Steps 4–6 — Backward pass, gradient check, race check\n"
            "\n"
            "The gradient flows through the smooth α (E[sample]=α), so it carries a usable, "
            "low-variance signal. Step 5 confirms the kernel's analytic ∂L/∂V matches central "
            "finite differences in float64 (no Monte-Carlo noise). Step 6 runs the float32 "
            "backward 50× on identical input: atomicAdd keeps the sum correct, so the run-to-run "
            "swing should sit at FP-ordering scale (not a race) and the mean must match the "
            "float64 reference.\n"
        ),
        code("!python step4_backward.py\n"),
    ]

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"name": "diffsoup_colab.ipynb", "provenance": []},
        },
        "cells": cells,
    }

    out = os.path.join(HERE, "step1_colab.ipynb")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"wrote {out}  ({os.path.getsize(out):,} bytes)")


if __name__ == "__main__":
    main()
