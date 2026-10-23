#!/usr/bin/env python3
"""P1-2: one design that satisfies A7(ii)-(iv) exactly.

Polynomial outcome means (degree 2) and a linear propensity, estimated with a
fixed degree-4 Legendre sieve and 5-fold cross-fitting.  The sieve represents the
outcome regression exactly, so the approximation error is zero and A7(ii) holds;
the degree is fixed, so A7(iv) holds.  This supplements, not replaces, the
rational-m_d designs of sim.py.

Run from this directory:
    python3 poly_cell.py
Writes sim/results/results_poly.csv.
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


def e_fn(s):
    # Depends on L only: e(s) = e(L).  This keeps both observed regressions
    # polynomial (the outcome means are polynomial and B is symmetric), which is
    # what A7(ii) needs; the design is therefore zero-bias by construction.
    return 0.5 + 0.15 * s


def mu1(s):
    return 1.0 + 2.0 * s + 0.3 * s**2


def mu0(s):
    return 0.5 * s + 0.1 * s**2


def truths():
    L = np.linspace(-L_HALF, L_HALF, 400001)
    th0 = 0.0
    for p, b in ((0.5, 1.0), (0.5, -1.0)):
        s = L + DELTA * b
        th0 += p * np.trapezoid(mu1(s) - mu0(s), L) / (2 * L_HALF)
    # e depends on L only, so the observed regressions are the conditional
    # means of the polynomial outcome functions: E[mu_d(L+delta B)|L].
    ebar = e_fn(L)
    m1 = 1.0 + 2.0 * L + 0.3 * (L**2 + DELTA**2)
    m0 = 0.5 * L + 0.1 * (L**2 + DELTA**2)
    thL = np.trapezoid(m1 - m0, L) / (2 * L_HALF)
    # e is independent of B, so Gamma = 0 exactly; the design is a coverage /
    # nuisance-rate test, not a bias test.
    gam = 0.0
    return float(th0), float(thL), float(gam)


def simulate(n, rng):
    L = rng.uniform(-L_HALF, L_HALF, n)
    B = rng.choice([-1.0, 1.0], size=n)
    X2 = DELTA * B
    s = L + X2
    D = (rng.uniform(size=n) < e_fn(L)).astype(float)
    Y = np.where(D == 1, mu1(s), mu0(s)) + rng.normal(size=n)
    th, se, Vhat = base.aipw_estimate(L, D, Y, rng, K=5, deg=4)
    return th, se, Vhat


def main():
    th0, thL, gam = truths()
    # Class constants on the support |s| <= 1.8 (delta=0.3, |L|<=1.5).
    M_mu = math.sqrt(2) * (2.0 + 0.6 * 1.8)
    M_e = math.sqrt(2) * 0.15
    M2 = 2.0 * 0.6
    eps = 0.2
    SigmaB = 1.0
    Gamma_bar = M_mu * M_e * SigmaB * (1 / eps + 1 / (1 - eps))
    rows = []
    for n in N_GRID:
        rng = default_rng(SeedSequence(MASTER_SEED).spawn(1)[0])
        est = np.empty(R)
        se = np.empty(R)
        for r in range(R):
            est[r], se[r], _ = simulate(n, rng)
        bias = est.mean() - th0
        rmse = math.sqrt(np.mean((est - th0) ** 2))
        naive = np.abs(est - th0) <= Z975 * se
        radius = DELTA**2 * Gamma_bar
        rob = np.abs(est - th0) <= Z975 * se + radius
        rows.append(dict(
            design="poly_fixed_degree", n=n, delta=DELTA, R=R,
            theta0=th0, thetaL=thL, gamma=gam, bias_0=bias,
            bias_0_se=est.std(ddof=1) / math.sqrt(R), rmse_0=rmse,
            coverage_naive=float(naive.mean()),
            coverage_naive_se=float(math.sqrt(naive.mean() * (1 - naive.mean()) / R)),
            coverage_rob=float(rob.mean()),
            mean_half_naive=float((Z975 * se).mean()),
            radius_abs=radius, mean_half_rob=float((Z975 * se + radius).mean()),
            M_mu=M_mu, M_e=M_e, M2=M2, eps=eps, Gamma_bar=Gamma_bar,
            sieve_degree=4, folds=5,
        ))
        print(f"n={n}: bias={bias:.5f} naive={naive.mean():.3f} rob={rob.mean():.3f} "
              f"half_naive={(Z975*se).mean():.4f} radius={radius:.4f}")
    out = Path(__file__).resolve().parent / "results" / "results_poly.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
