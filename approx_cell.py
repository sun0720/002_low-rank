#!/usr/bin/env python3
"""014 P1-3: certified nuisance-rate evidence and small-overlap boundary cells.

Three panels, all built on the certified joint design of joint_cell.py (P9
structure (B), bounded U(-1,1) outcome noise, fixed-degree sieve, compact
support):

  A. rate-compliant approximation error (in class)
     The observed regressions and the observed propensity carry a degree-5
     Legendre component with coefficient eps_n = n^{-3/8}.  A degree-4 outcome
     sieve and a linear propensity therefore have approximation error of order
     eps_n, which is o(n^{-1/4}); the product of the two errors is
     eps_n^2 = n^{-3/4} = o(n^{-1/2}).  A7(ii) holds with a nonzero error, and
     every cell remains in P_n^C.  The structural bias is unchanged because
     the Legendre component has mean zero and is common to both arms.

  B. small overlap (in class)
     The residual slope b in e(L,B) = 1/2 + delta b S is increased.  The actual
     overlap constant is eps = 1/2 - delta b, and the certified radius uses
     M_e = b and that eps.  This is a numerical instance of the paper's own
     epsilon -> 0 degeneration: the interval is valid and becomes uninformative.

  C. nuisance-rate boundary (outside class, negative control)
     Same construction as panel A but with eps_n = n^{-1/4}, so the product of
     the two approximation errors is n^{-1/2}, which violates A7(ii)'s
     o(n^{-1/2}) requirement.  The cell documents the boundary; it is not a
     member of P_n^C and is not used as evidence for the theorem.

Run from this directory:
    python3 approx_cell.py
Writes results/results_approx.csv.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from numpy.random import SeedSequence, default_rng

import sim as base

Z975 = 1.959963984540054
MASTER_SEED = 20261022
R = 500
N_GRID = (4000, 16000)
DELTA = 0.3
L_HALF = 1.5
A = 1.5            # residual slope in mu_d
SIGMA_B = 1.0
M_MU = 2.0         # sup|mu_d^0'| + a = 0.5 + 1.5
MU4 = 1.0
M2 = 0.0
M2P = 0.0


def p5(x):
    """Legendre P5 on [-1, 1]; mean zero under the uniform law."""
    return (63.0 * x**5 - 70.0 * x**3 + 15.0 * x) / 8.0


def e_fn(l, s, b, delta=DELTA, eps_n=0.0):
    return 0.5 + delta * b * s + eps_n * p5(l / L_HALF)


def mu1(l, s, delta=DELTA):
    return 1.0 + 0.5 * l + delta * A * s


def mu0(l, s, delta=DELTA):
    return 0.2 * l + delta * A * s


def simulate(n, rng, b, eps_n):
    L = rng.uniform(-L_HALF, L_HALF, n)
    S = rng.choice([-1.0, 1.0], size=n)
    D = (rng.uniform(size=n) < e_fn(L, S, b, eps_n=eps_n)).astype(float)
    eps_y = eps_n
    Y = (np.where(D == 1, mu1(L, S), mu0(L, S))
         + eps_y * p5(L / L_HALF) + rng.uniform(-1.0, 1.0, n))
    # Class-constant clipping at eps/2 is set by the caller's certified eps.
    eps_cert = 0.5 - DELTA * b - eps_n
    max_y = 1.0 + 0.5 * L_HALF + DELTA * A + abs(eps_y) + 1.0
    th, se, Vhat = base.aipw_estimate(L, D, Y, rng, K=5, deg=4,
                                     clip=(eps_cert / 2, 1 - eps_cert / 2),
                                     y_clip=max_y)
    return th, se, Vhat


def cell(n, b, eps_n, rng, panel, class_member, rate_note):
    # Certified overlap constant: actual infimum of e, rounded down slightly.
    eps_cert = 0.5 - DELTA * b - eps_n
    assert eps_cert > 0.0, "overlap must remain positive"
    assert eps_cert >= 1e-2, "cells below one percent overlap are not reported"
    # Truth: theta_0 = 1; theta_L = 1 + 4 delta^2 a b (Legendre term mean zero).
    th0 = 1.0
    gam = 4.0 * A * b
    thL = th0 + DELTA**2 * gam
    gamma_bar = M_MU * b * SIGMA_B * (1 / eps_cert + 1 / (1 - eps_cert))
    c1 = 2 * M2 * b * MU4**0.75
    c_tilde = (1 / eps_cert + 1 / (1 - eps_cert)) * c1
    radius = DELTA**2 * (gamma_bar + c_tilde * DELTA)
    est = np.empty(R)
    se = np.empty(R)
    for r in range(R):
        est[r], se[r], _ = simulate(n, rng, b, eps_n)
    bias = est.mean() - th0
    bias_L = est.mean() - thL
    naive = np.abs(est - th0) <= Z975 * se
    rob = np.abs(est - th0) <= Z975 * se + radius
    return dict(
        panel=panel, n=n, R=R, delta=DELTA, b_slope=b, a_slope=A,
        eps_cert=eps_cert, eps_n=eps_n, eps_n_rate=rate_note,
        approx_product_rate=eps_n**2,
        theta0=th0, thetaL=thL, gamma=gam, bias_pred=DELTA**2 * gam,
        bias_0=bias, bias_0_se=est.std(ddof=1) / math.sqrt(R),
        bias_L=bias_L, bias_L_se=est.std(ddof=1) / math.sqrt(R),
        coverage_naive=float(naive.mean()),
        coverage_naive_se=float(math.sqrt(naive.mean() * (1 - naive.mean()) / R)),
        coverage_rob=float(rob.mean()),
        coverage_rob_se=float(math.sqrt(rob.mean() * (1 - rob.mean()) / R)),
        mean_half_naive=float((Z975 * se).mean()),
        radius_abs=radius, mean_half_rob=float((Z975 * se + radius).mean()),
        M_mu=M_MU, M_e=b, Sigma_B=SIGMA_B, mu4=MU4, M2=M2, M2P=M2P,
        class_member=class_member, sieve_degree=4, folds=5,
    )


def main() -> None:
    rows = []
    for n in N_GRID:
        eps_n = n ** (-3.0 / 8.0)
        rng = default_rng(SeedSequence(MASTER_SEED).spawn(1)[0])
        rows.append(cell(n, 0.15, eps_n, rng, "A_rate_compliant",
                         "yes: nonzero o(n^-1/4) sieve error, product o(n^-1/2)",
                         "n^-3/8"))

    for n in N_GRID:
        for b in (0.15, 0.6, 1.2, 1.5):
            rng = default_rng(SeedSequence(MASTER_SEED + 1).spawn(1)[0])
            rows.append(cell(n, b, 0.0, rng, "B_small_overlap",
                             "yes: exact sieves, certified eps = 1/2 - delta b",
                             "0"))

    for n in N_GRID:
        eps_n = n ** (-1.0 / 4.0)
        rng = default_rng(SeedSequence(MASTER_SEED + 2).spawn(1)[0])
        rows.append(cell(n, 0.15, eps_n, rng, "C_rate_boundary",
                         "no: product n^-1/2 is not o(n^-1/2)",
                         "n^-1/4"))

    out = Path(__file__).resolve().parent / "results" / "results_approx.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for row in rows:
        print(f"{row['panel']:<18} n={row['n']:<6} b={row['b_slope']:<5} "
              f"eps={row['eps_cert']:.3f} eps_n={row['eps_n']:.4f} "
              f"bias={row['bias_0']:+.4f} (pred {row['bias_pred']:+.4f}) "
              f"naive={row['coverage_naive']:.3f} rob={row['coverage_rob']:.3f} "
              f"radius={row['radius_abs']:.4f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
