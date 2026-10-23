#!/usr/bin/env python3
"""DEC-013-03: certified joint design along the critical sequence.

The fixed-delta joint cell in joint_cell.py tests the bias identity at
delta = 0.3.  This script keeps the same certified design (P9 structure (B),
bounded U(-1,1) noise, fixed-degree sieve, compact support) and moves it along
the critical sequence

    delta_n = c n^{-1/4},   K_n = sqrt(n) delta_n^2 = c^2,

for a grid of c and n.  Every cell is a member of the same certified class:
the observed regressions are linear, a degree-4 Legendre sieve is exact,
A7(ii)-(iv) hold, Y is bounded, and Remark C0 applies.  The reported objects
are bias, naive and bias-aware coverage, shift, width, and Monte Carlo error.

Run from this directory:
    python3 joint_grid.py
Writes sim/results/results_joint_grid.csv.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from numpy.random import SeedSequence, default_rng

import joint_cell as jc

Z975 = 1.959963984540054
MASTER_SEED = 20261021
R = 500
N_GRID = (4000, 16000)
C_GRID = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
DELTA_BAR = 0.3


def main() -> None:
    gamma_bar = jc.M_MU * jc.M_E * jc.SIGMA_B * (1 / jc.EPS + 1 / (1 - jc.EPS))
    c1 = 2 * jc.M2 * jc.M_E * jc.MU4**0.75
    c_tilde = (1 / jc.EPS + 1 / (1 - jc.EPS)) * c1
    rows = []
    for n in N_GRID:
        for c in C_GRID:
            delta = c * n ** (-0.25)
            if delta > DELTA_BAR:
                continue
            th0, thL, gam, Vtrue, shift = jc.truths(delta)
            radius = delta**2 * (gamma_bar + c_tilde * delta)
            max_y = 1.0 + 0.5 * jc.L_HALF + delta * jc.A + 1.0
            rng = default_rng(SeedSequence(MASTER_SEED).spawn(1)[0])
            est = np.empty(R)
            se = np.empty(R)
            for r in range(R):
                est[r], se[r], _ = jc.simulate(n, rng, delta)
            bias = est.mean() - th0
            bias_L = est.mean() - thL
            rmse = math.sqrt(np.mean((est - th0) ** 2))
            naive = np.abs(est - th0) <= Z975 * se
            rob = np.abs(est - th0) <= Z975 * se + radius
            K = math.sqrt(n) * delta**2
            rows.append(dict(
                design="joint_critical_sequence", n=n, c=c, delta=delta,
                K_n=K, R=R, theta0=th0, thetaL=thL, gamma=gam,
                bias_pred=delta**2 * gam, bias_0=bias,
                bias_0_se=est.std(ddof=1) / math.sqrt(R),
                bias_L=bias_L,
                bias_L_se=est.std(ddof=1) / math.sqrt(R),
                rmse_0=rmse,
                coverage_naive=float(naive.mean()),
                coverage_naive_se=float(math.sqrt(naive.mean() * (1 - naive.mean()) / R)),
                coverage_rob=float(rob.mean()),
                coverage_rob_se=float(math.sqrt(rob.mean() * (1 - rob.mean()) / R)),
                mean_half_naive=float((Z975 * se).mean()),
                radius_abs=radius, mean_half_rob=float((Z975 * se + radius).mean()),
                shift_K_Gamma=K * gam,
                V_true=Vtrue, max_abs_Y=max_y, delta_bar=DELTA_BAR,
                class_member="yes: bounded Y, fixed-degree sieve, linear m_d",
                sieve_degree=4, folds=5,
            ))
            print(f"n={n} c={c:.2f} delta={delta:.4f} K={K:.3f} "
                  f"bias={bias:+.4f} (pred {delta**2*gam:+.4f}) "
                  f"naive={naive.mean():.3f} rob={rob.mean():.3f} "
                  f"radius={radius:.4f}")
            out = Path(__file__).resolve().parent / "results" / "results_joint_grid.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    out = Path(__file__).resolve().parent / "results" / "results_joint_grid.csv"
    print("wrote", out)


if __name__ == "__main__":
    main()
