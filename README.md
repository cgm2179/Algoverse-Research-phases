# Algoverse Research — Phases

Experiments for a research project on **local, backprop-free (zeroth-order)
post-training** of neural networks with low-rank (**LoRA**) adapters.

The work is staged so that the cheapest experiment capable of *falsifying* a
claim runs first — before any GPU or TPU compute is spent.

---

## Tier 0 — Alignment-law validation (CPU, ~1 minute)

The foundational check: a zeroth-order (forward-pass-only) gradient estimator
recovers the true gradient only up to an alignment that depends on an *effective
dimension*. Tier 0 measures that alignment on a real nonlinear loss and compares
it to prediction.

| File | What it is |
|------|------------|
| [`Tier0_alignment_laws_walkthrough.ipynb`](Tier0_alignment_laws_walkthrough.ipynb) | Explained, run-top-to-bottom notebook (markdown + code). |
| [`exp0_alignment_laws.py`](exp0_alignment_laws.py) | The same experiment as a plain script (saves a plot). |

### What it tests

For an estimator that averages `M` random probes, the squared cosine between the
estimated gradient and the true gradient is predicted to follow:

- **weight perturbation:**  `cos²θ = M / (M + P + 1)`,  with `P = r(m+n)` the LoRA weight count
- **node perturbation:**  `cos²θ = M / (M + d̄ + 1)`,  with `d̄ = (1−β)·r + β·m` an energy-weighted node dimension
  (`β` = fraction of adapter gradient energy in the `B` matrix, measured from the reference gradient)

The script builds a small LoRA-adapted nonlinear network, computes exact
gradients by backprop as a reference, then measures both estimators against the
two predicted curves. A **pass** shows the node-perturbation curve rising much
faster than weight perturbation — i.e. a substantially smaller effective
dimension (`d̄ ≪ P`) and the same alignment reached with far fewer probes.

### Run it

**Colab:** upload the notebook → *Runtime ▸ Change runtime type ▸ CPU* → *Run all*.

**Local:**
```bash
pip install numpy matplotlib
python exp0_alignment_laws.py     # prints the table, writes exp0_alignment_laws.png
```

---

## Tier 1 — Alignment laws + rank sweep under a real denoising loss (GPU-ready)

[`Tier1_diffusion_lora_walkthrough.ipynb`](Tier1_diffusion_lora_walkthrough.ipynb) — the
Tier-0 estimators wired into an actual diffusion **denoising loss** with a
LoRA-adapted layer inside a small denoiser (PyTorch, MNIST with a synthetic
fallback). Two experiments: (1) the alignment laws hold under the denoising loss
(node `d̄ ≪ P`); (2) a rank sweep whose captured **loss** saturates at the task's
intrinsic rank (Eckart–Young knee), with the honest caveat that the loss knee —
not the raw Frobenius-energy knee — is the operative one under an anisotropic
Hessian.

> Uses a per-element (dimension-invariant) loss and a base-convergence check;
> summing the loss over pixels makes the effective learning rate scale with image
> size and breaks pretraining at MNIST resolution.

---

## Tier 2 — Estimation cost vs rank (GPU-ready)

[`Tier2_estimation_cost_vs_rank.ipynb`](Tier2_estimation_cost_vs_rank.ipynb) — measures
the effective dimension `D_eff(r)` of each estimator across LoRA ranks (by fitting
`1/cos²θ = 1 + (D+1)/M`) and shows that `M*(r)`, the probes needed to reach a
target alignment, is linear in rank with the predicted slopes: **weight `∝ r(m+n)`**
vs **node `∝ (1−β)r`** — a per-rank estimation-cost gap of order `m+n`.

---

## Roadmap

- **Tier 3 (TPU v5e/v6e):** move to an open-weight diffusion backbone; identify
  which layer carries a local residual channel and post-train that segment with
  node perturbation, budgeting probes from the measured `d̄`.
