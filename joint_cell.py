#!/usr/bin/env python3
"""011 P1-3/P1-4: a joint design with nonzero bias and a certified nuisance layer.

The main designs of sim.py have a rational m_d(L), so a fixed-degree sieve
retains approximation error and A7(ii) fails; the polynomial design of
poly_cell.py satisfies A7(ii)-(iv) but is zero-bias because e depends on L only.
This script adds the missing joint cell.

DGP (the P9 structure (B), with a nonconstant polynomial mu_d^0):
    L ~ U(-1.5, 1.5),  S in {-1,+1} symmetric and independent of L
    B = S,  delta = 0.3,  X = L + delta * B
    e(L,B) = 1/2 + delta * b * S,          b = 0.15
    mu_d(L,B) = mu_d^0(L) + delta * a * S, a = 1.5
    mu_1^0(L) = 1 + 0.5 L,  mu_0^0(L) = 0.2 L
    Y = mu_D(L,B) + xi,  xi ~ U(-1,1)  (bounded outcome noise)

Because e depends on B through S, the bias is nonzero:
    theta_L - theta_0 = 4 delta^2 a b = 0.081.
Because e is constant in L (ebar = 1/2), the observed regressions are
    m_1(L) = 1 + 0.5 L + 2 delta^2 a b,   m_0(L) = 0.2 L - 2 delta^2 a b,
both linear; a fixed-degree Legendre sieve represents them exactly, so A7(ii)
holds.  The degree is fixed (4), so A7(iv) holds.  The outcome is bounded on a
compact support, so Remark C0 applies: the design lies in P_n^C and tests
Proposition P8(iii) with a nonzero structural bias.

Run from this directory:
    python3 joint_cell.py
Writes results/results_joint.csv.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from numpy.random import SeedSequence, default_rng

import sim as base

Z975 = 1.959963984540054
MASTER_SEED = 20261020
R = 1000
N_GRID = (4000, 16000)
DELTA = 0.3
L_HALF = 1.5
A = 1.5          # residual slope in mu_d
B = 0.15         # residual slope in e
M_MU = 2.0       # path gradient bound: sup|mu_d^0'| + a = 0.5 + 1.5
M_E = B
SIGMA_B = 1.0
MAX_Y = 1.0 + 0.5 * L_HALF + DELTA * A + 1.0   # |mu_1| + |xi| bound
EPS = 0.4        # e in [0.455, 0.545]
MU4 = 1.0
M2 = 0.0
M2P = 0.0


def e_fn(l, s, delta=DELTA):
    return 0.5 + delta * B * s


def mu1(l, s, delta=DELTA):
    return 1.0 + 0.5 * l + delta * A * s


def mu0(l, s, delta=DELTA):
    return 0.2 * l + delta * A * s


def truths(delta=DELTA):
    # theta_0 = E[mu_1 - mu_0] = 1 (E[L] = E[S] = 0)
    th0 = 1.0
    # m_d(L) = mu_d^0(L) +/- 2 delta^2 a b
    shift = 2.0 * delta**2 * A * B
    thL = 1.0 + 2.0 * shift
    gam = 4.0 * A * B              # delta^2 * Gamma = 4 delta^2 A B
    # V = E[(m1-m0-thetaL)^2] + E[sigma_1^2/ebar + sigma_0^2/(1-ebar)]
    var_l = (0.3 * L_HALF) ** 2 / 3.0      # Var(0.3 L), L ~ U(-1.5,1.5)
    mix_var = (delta * A) ** 2 * (1.0 - (delta * B) ** 2)
    noise_var = 1.0 / 3.0                   # U(-1,1)
    V = var_l + 4.0 * (mix_var + noise_var)  # ebar = 1/2
    return float(th0), float(thL), float(gam), float(V), float(shift)


def simulate(n, rng, delta=DELTA):
    L = rng.uniform(-L_HALF, L_HALF, n)
    S = rng.choice([-1.0, 1.0], size=n)
    D = (rng.uniform(size=n) < e_fn(L, S, delta)).astype(float)
    Y = np.where(D == 1, mu1(L, S, delta), mu0(L, S, delta)) + rng.uniform(-1.0, 1.0, n)
    max_y = 1.0 + 0.5 * L_HALF + delta * A + 1.0
    th, se, Vhat = base.aipw_estimate(L, D, Y, rng, K=5, deg=4,
                                     clip=(EPS / 2, 1 - EPS / 2),
                                     y_clip=max_y)
    return th, se, Vhat


def main():
    th0, thL, gam, Vtrue, shift = truths()
    gamma_bar = M_MU * M_E * SIGMA_B * (1 / EPS + 1 / (1 - EPS))
    c1 = 2 * M2 * M_E * MU4**0.75        # M2 = M2P = 0
    c_tilde = (1 / EPS + 1 / (1 - EPS)) * c1
    radius = DELTA**2 * (gamma_bar + c_tilde * DELTA)
    max_y = MAX_Y
    rows = []
    for n in N_GRID:
        rng = default_rng(SeedSequence(MASTER_SEED).spawn(1)[0])
        est = np.empty(R)
        se = np.empty(R)
        for r in range(R):
            est[r], se[r], _ = simulate(n, rng)
        bias = est.mean() - th0
        bias_L = est.mean() - thL
        rmse = math.sqrt(np.mean((est - th0) ** 2))
        naive = np.abs(est - th0) <= Z975 * se
        rob = np.abs(est - th0) <= Z975 * se + radius
        rows.append(dict(
            design="joint_bounded_noise", n=n, delta=DELTA, R=R,
            theta0=th0, thetaL=thL, gamma=gam, bias_0=bias,
            bias_0_se=est.std(ddof=1) / math.sqrt(R),
            bias_L=bias_L, bias_L_se=est.std(ddof=1) / math.sqrt(R),
            rmse_0=rmse, coverage_naive=float(naive.mean()),
            coverage_naive_se=float(math.sqrt(naive.mean() * (1 - naive.mean()) / R)),
            coverage_rob=float(rob.mean()),
            mean_half_naive=float((Z975 * se).mean()),
            radius_abs=radius, mean_half_rob=float((Z975 * se + radius).mean()),
            M_mu=M_MU, M_e=M_E, Sigma_B=SIGMA_B, eps=EPS, mu4=MU4,
            M2=M2, M2P=M2P, Gamma_bar=gamma_bar, C_tilde=c_tilde,
            V_true=Vtrue, max_abs_Y=max_y, sieve_degree=4, folds=5,
            C0_verified="yes: bounded Y, fixed-degree sieve, compact support",
        ))
        print(f"n={n}: bias={bias:+.4f} (se {rows[-1]['bias_0_se']:.4f}) "
              f"bias_L={bias_L:+.4f} naive={naive.mean():.3f} rob={rob.mean():.3f} "
              f"half_naive={(Z975*se).mean():.4f} radius={radius:.4f}")
    out = Path(__file__).resolve().parent / "results" / "results_joint.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
