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

## Roadmap

- **Tier 1 (GPU):** the same estimators wired into a small diffusion denoiser
  with LoRA adapters, on a T4/L4.
