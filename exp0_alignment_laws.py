"""
exp0_alignment_laws.py  —  Tier-0 experiment (CPU only, ~seconds)

Falsification test for the two alignment laws behind the PIHL post-training plan,
on a REAL nonlinear loss with a LoRA-adapted layer and FINITE-mu zeroth-order
probes (not the mu->0 idealization). Ground-truth gradients come from exact
manual backprop; zeroth-order estimates come from forward passes only.

Validates:
  (1) weight-perturbation alignment:  cos^2(theta) = M / (M + P + 1),   P = r(m+n)
  (2) node-perturbation   alignment:  cos^2(theta) = M / (M + dbar + 1),
                                       dbar = (1-beta) r + beta m,
                                       beta = ||dL/dB||^2 / (||dL/dA||^2 + ||dL/dB||^2)

Model (one adapted linear layer + nonlinearity + linear head):
    z = A x            (bottleneck, r-dim)
    u = B z            (adapter output, m-dim)
    p = W0 x + u       (pre-activation, m-dim)
    h = tanh(p)
    out = C h
    L = 0.5 * ||out - t||^2

Dependencies: numpy, matplotlib.   Run:  python exp0_alignment_laws.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# ---- dimensions ----
n, m, r, k = 100, 64, 8, 10      # in-dim, out-dim(adapted), LoRA rank, head-dim
mu = 1e-4                         # finite-difference step (float64 keeps bias tiny)

# ---- frozen backbone + trainable adapter + fixed head/input/target ----
W0 = rng.standard_normal((m, n)) / np.sqrt(n)
A  = rng.standard_normal((r, n)) / np.sqrt(n)
B  = rng.standard_normal((m, r)) / np.sqrt(r)
C  = rng.standard_normal((k, m)) / np.sqrt(m)
x  = rng.standard_normal(n)
t  = rng.standard_normal(k)

def forward(A_, B_, z_override=None, u_override=None):
    """Forward pass; optionally inject perturbed node activations."""
    z = A_ @ x
    if z_override is not None:
        z = z_override
    u = B_ @ z
    if u_override is not None:
        u = u_override
    p = W0 @ x + u
    h = np.tanh(p)
    out = C @ h
    return 0.5 * np.sum((out - t) ** 2), z, u, p, h, out

# ---- exact gradients via manual backprop (ground truth) ----
L0, z0, u0, p0, h0, out0 = forward(A, B)
dout = out0 - t
dh   = C.T @ dout
dp   = dh * (1 - h0 ** 2)          # tanh'
g_u  = dp                          # dL/du   (p = W0 x + u)
g_z  = B.T @ g_u                   # dL/dz   (u = B z)
gA   = np.outer(g_z, x)            # dL/dA
gB   = np.outer(g_u, z0)           # dL/dB
g    = np.concatenate([gA.ravel(), gB.ravel()])
gn2  = g @ g

E_A, E_B = np.sum(gA ** 2), np.sum(gB ** 2)
beta = E_B / (E_A + E_B)
dbar = (1 - beta) * r + beta * m
P    = r * n + m * r               # = r(m+n)

print(f"dims: n={n} m={m} r={r} | P = r(m+n) = {P}")
print(f"beta = {beta:.4f}   ->   dbar = (1-beta)r + beta m = {dbar:.3f}")
print(f"weight-space effective dim P = {P}   |   node-space effective dim dbar = {dbar:.2f}"
      f"   ({P/dbar:.1f}x smaller)\n")

# ---- zeroth-order estimators (finite mu, antithetic two-point) ----
def zo_weight_probe():
    xiA = rng.standard_normal((r, n)); xiB = rng.standard_normal((m, r))
    Lp, *_ = forward(A + mu * xiA, B + mu * xiB)
    Lm, *_ = forward(A - mu * xiA, B - mu * xiB)
    d = (Lp - Lm) / (2 * mu)
    return np.concatenate([(d * xiA).ravel(), (d * xiB).ravel()])

def zo_node_probe():
    # site 1: perturb bottleneck z (r-dim) -> estimate g_z (total derivative)
    xi_z = rng.standard_normal(r)
    Lp, *_ = forward(A, B, z_override=z0 + mu * xi_z)
    Lm, *_ = forward(A, B, z_override=z0 - mu * xi_z)
    gz_hat = ((Lp - Lm) / (2 * mu)) * xi_z
    # site 2: perturb adapter output u (m-dim) -> estimate g_u
    xi_u = rng.standard_normal(m)
    Lp, *_ = forward(A, B, u_override=u0 + mu * xi_u)
    Lm, *_ = forward(A, B, u_override=u0 - mu * xi_u)
    gu_hat = ((Lp - Lm) / (2 * mu)) * xi_u
    eA = np.outer(gz_hat, x); eB = np.outer(gu_hat, z0)
    return np.concatenate([eA.ravel(), eB.ravel()])

# ---- measure signal-to-energy cos^2 vs M, compare to predictions ----
def alignment_curve(probe_fn, Ms, reps):
    out = {}
    for M in Ms:
        num = den = 0.0
        for _ in range(reps):
            acc = np.zeros_like(g)
            for _ in range(M):
                acc += probe_fn()
            acc /= M
            num += acc @ g
            den += acc @ acc
        num /= reps; den /= reps
        out[M] = (num ** 2) / (gn2 * den)
    return out

Ms = [1, 4, 16, 64, 256]
reps = 3000
cw = alignment_curve(zo_weight_probe, Ms, reps)
cn = alignment_curve(zo_node_probe,   Ms, reps)

print(f"{'M':>4} | {'weight cos^2':>12} {'M/(M+P+1)':>11} | {'node cos^2':>11} {'M/(M+dbar+1)':>13}")
for M in Ms:
    print(f"{M:>4} | {cw[M]:>12.4f} {M/(M+P+1):>11.4f} | {cn[M]:>11.4f} {M/(M+dbar+1):>13.4f}")

# ---- plot ----
Mgrid = np.array(Ms)
plt.figure(figsize=(7, 4.5))
plt.plot(Mgrid, [cw[M] for M in Ms], "o", color="C3", label="weight ZO (measured)")
plt.plot(Mgrid, Mgrid/(Mgrid+P+1), "-", color="C3", alpha=.6, label=f"M/(M+P+1), P={P}")
plt.plot(Mgrid, [cn[M] for M in Ms], "s", color="C0", label="node ZO (measured)")
plt.plot(Mgrid, Mgrid/(Mgrid+dbar+1), "-", color="C0", alpha=.6, label=f"M/(M+dbar+1), dbar={dbar:.1f}")
plt.xscale("log"); plt.xlabel("probe budget M"); plt.ylabel(r"alignment $\cos^2\theta$")
plt.title("Alignment laws: node perturbation vs weight perturbation")
plt.legend(fontsize=8); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig("exp0_alignment_laws.png", dpi=130)
print("\nsaved plot -> exp0_alignment_laws.png")
