#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""h001 · 低秩局部误设下 oracle-L cross-fitted AIPW 的 Monte Carlo（econometric-simu ⑨节产物）。

复现命令：python3 sim.py

本文件只检验 theory/theory.md ⑫ 交来的可证伪预言 D1–D6，不新增、不改写任何预言或假定。
DGP 逐条回指 model/blueprint.md ⑥ 的冻结假定 A1 · A2 · A3a · A4a · A4b · A5 · A7；
污染机制取 blueprint.md ② ③ 的加性局部误设 X = L + δ_n B（B 内生，进入 e 与 μ_d）。

三条铁律焊死在代码里：
  1. 主种子 + SeedSequence(master).spawn(R) 派生 R 条独立流（复制之间不相关）。
  2. 每个估计量返回 (点估计, 标准误)；覆盖率用估计出来的 V̂，不用真 V。
  3. 覆盖率等随机量一律报 Monte Carlo 标准误；分位数用 bootstrap 报 SE。

产物：sim/results/results.csv · results_p9.csv · summary.txt · *.png
"""
from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path

import numpy as np
from numpy.random import SeedSequence, default_rng

# ── 0. 常量：DGP 参数与类常数（③ 逐条回指 A#）────────────────────────────
L_HALF = 1.5          # L ~ U(-1.5, 1.5)，支撑有界 ⇒ A2 的 ε 可逐点核对
A0, A1 = 0.5, 0.15    # e(x) = A0 + A1 x（线性概率；A2 · A4a · A4b）
E_EPS = 0.15          # A2：ε ≤ e ≤ 1-ε（δ_max=0.7 时 e∈[0.17,0.83]）
Z975 = 1.959963984540054
MASTER_SEED = 20261020

# 类常数（A4a/A4b/A5；在 |x|≤2.2 上取上确界，δ_max=0.7）。
# 标量指标 s=X_1+X_2 嵌在 R^2 中：对 x 的梯度是 s-导数的 √2 倍，
# Hessian 算子范数是二阶导数的 2 倍（2026-10-21 外审 009 P1-4 修正）。
M_MU = 4.07 * 2 ** 0.5
M_E = A1 * 2 ** 0.5
SIGMA_B = 1.0         # B∈{-1,+1} ⇒ Var(B|L)=1, E[B⁴]=1
M2 = 1.27 * 2.0
ASYM_P = 0.7          # P2-7b：非对称残差 P(B=+1)=0.7
M2P = 0.0
MU4 = 1.0
GAMMA_BAR = M_MU * M_E * SIGMA_B * (1 / E_EPS + 1 / (1 - E_EPS))
C1_TILDE = 2 * M_MU * M2P * MU4**0.75 + 2 * M2 * M_E * MU4**0.75 + M2 * M2P * MU4
C_TILDE = (1 / E_EPS + 1 / (1 - E_EPS)) * C1_TILDE


def gamma_bar_eff(dbar: float) -> float:
    """blueprint ④ 的 Γ̄_eff = Γ̄ + C δ̄（候选甲）。"""
    return GAMMA_BAR + C_TILDE * dbar


def radius_main(dbar: float) -> float:
    """主 DGP 的诚实半径：δ̄²Γ̄_eff（= 主项 δ̄²Γ̄ + 余项 C̃δ̄³）。"""
    return dbar ** 2 * gamma_bar_eff(dbar)


# ── 0b. 光谱 DGP 的常数（⑥ spectral：秩 r 真正承重）──────────────────────
# X ∈ R^P，保留子空间 span(e_1..e_r)，残差在 (P-r) 维正交补上，谱 λ_j=(j-1)^{-2}。
SPEC_P = 6
SPEC_W = np.array([0.25, 0.70, 0.30, 0.17, 0.11, 0.07])   # 对齐方向 w（‖w‖ 归一化由代码算）
SPEC_SLOPE = 0.12                                          # e = 0.5 + SLOPE · w'X
SPEC_LAM = np.full(SPEC_P + 1, np.nan)
for _j in range(2, SPEC_P + 1):
    SPEC_LAM[_j] = (_j - 1) ** (-2.0)                      # λ_2=1 ≥ λ_3=1/4 ≥ ...
SPEC_L_HALF = 1.5


# ── 1. 主 DGP：X = L + δ B 的加性局部误设（A3a），B 内生（A1 的混淆通道）──
def e_fn(x):
    return A0 + A1 * x


def mu1(x):
    return 1.0 + 2.0 * x + 0.3 * x**2 + 0.05 * x**3


def mu0(x):
    return 0.0 + 0.5 * x + 0.1 * x**2 + 0.02 * x**3


def dmu1(x):
    return 2.0 + 0.6 * x + 0.15 * x**2


def dmu0(x):
    return 0.5 + 0.2 * x + 0.06 * x**2


# 求积节点（L ~ U(-L_HALF, L_HALF)）
_xg, _wg = np.polynomial.legendre.leggauss(300)
LQ = L_HALF * _xg
WQ = 0.5 * _wg


def truths_main(delta: float) -> dict:
    """主 DGP 的 θ0, θL, Γ, V（求积；与 DGP 同一参数化）。"""
    ep, em = e_fn(LQ + delta), e_fn(LQ - delta)
    m1p, m1m = mu1(LQ + delta), mu1(LQ - delta)
    m0p, m0m = mu0(LQ + delta), mu0(LQ - delta)
    w1p, w1m = ep / (ep + em), em / (ep + em)
    m1 = w1p * m1p + w1m * m1m
    var1 = w1p * m1p**2 + w1m * m1m**2 - m1**2
    f = 2 - ep - em
    w0p, w0m = (1 - ep) / f, (1 - em) / f
    m0 = w0p * m0p + w0m * m0m
    var0 = w0p * m0p**2 + w0m * m0m**2 - m0**2
    thL = float(np.sum(WQ * (m1 - m0)))
    th0 = float(np.sum(WQ * 0.5 * (m1p + m1m - m0p - m0m)))
    # Γ = E[g_μ1'Σ_B g_e/ē + g_μ0'Σ_B g_e/(1-ē)]，Σ_B=1，ē=e(L)
    gam = float(np.sum(WQ * (A1 * dmu1(LQ) / e_fn(LQ) + A1 * dmu0(LQ) / (1 - e_fn(LQ)))))
    V = float(np.sum(WQ * (m1 - m0 - thL) ** 2)
              + np.sum(WQ * ((1 + var1) / e_fn(LQ) + (1 + var0) / (1 - e_fn(LQ)))))
    return dict(th0=th0, thL=thL, gam=gam, V=V, bias=thL - th0)


def truths_asym(delta: float, p: float = ASYM_P) -> dict:
    """非对称两点残差 B=+1 (概率 p) / -1 (概率 1-p) 的真值。"""
    q = 1.0 - p
    ep, em = e_fn(LQ + delta), e_fn(LQ - delta)
    m1p, m1m = mu1(LQ + delta), mu1(LQ - delta)
    m0p, m0m = mu0(LQ + delta), mu0(LQ - delta)
    ebar = p * ep + q * em
    m1 = (p * ep * m1p + q * em * m1m) / ebar
    m0 = (p * (1 - ep) * m0p + q * (1 - em) * m0m) / (1 - ebar)
    var1 = (p * ep * (m1p - m1) ** 2 + q * em * (m1m - m1) ** 2) / ebar
    var0 = (p * (1 - ep) * (m0p - m0) ** 2 + q * (1 - em) * (m0m - m0) ** 2) / (1 - ebar)
    thL = float(np.sum(WQ * (m1 - m0)))
    th0 = float(np.sum(WQ * (p * (m1p - m0p) + q * (m1m - m0m))))
    varb = 4.0 * p * q
    gam = float(varb * np.sum(WQ * (A1 * dmu1(LQ) / ebar + A1 * dmu0(LQ) / (1 - ebar))))
    V = float(np.sum(WQ * (m1 - m0 - thL) ** 2)
              + np.sum(WQ * ((1 + var1) / ebar + (1 + var0) / (1 - ebar))))
    return dict(th0=th0, thL=thL, gam=gam, V=V, bias=thL - th0, varb=varb)


def gen_asym(n, rng, delta, p: float = ASYM_P):
    Lc = rng.uniform(-L_HALF, L_HALF, n)
    B = np.where(rng.random(n) < p, 1.0, -1.0)
    X = Lc + delta * B
    D = (rng.uniform(size=n) < e_fn(X)).astype(float)
    Y = np.where(D == 1, mu1(X), mu0(X)) + rng.normal(size=n)
    return Lc, X, D, Y


def truths_off(delta: float) -> dict:
    """D5 机制关掉：e 只看 L（B ⟂ D | L），B 仍进 Y。"""
    m1 = 0.5 * (mu1(LQ + delta) + mu1(LQ - delta))
    m0 = 0.5 * (mu0(LQ + delta) + mu0(LQ - delta))
    var1 = 0.5 * (mu1(LQ + delta) ** 2 + mu1(LQ - delta) ** 2) - m1**2
    var0 = 0.5 * (mu0(LQ + delta) ** 2 + mu0(LQ - delta) ** 2) - m0**2
    thL = float(np.sum(WQ * (m1 - m0)))
    th0 = float(np.sum(WQ * 0.5 * (mu1(LQ + delta) + mu1(LQ - delta)
                                   - mu0(LQ + delta) - mu0(LQ - delta))))
    V = float(np.sum(WQ * (m1 - m0 - thL) ** 2)
              + np.sum(WQ * ((1 + var1) / e_fn(LQ) + (1 + var0) / (1 - e_fn(LQ)))))
    return dict(th0=th0, thL=thL, gam=0.0, V=V, bias=thL - th0)


def gen_main(n, rng, delta):
    Lc = rng.uniform(-L_HALF, L_HALF, n)
    B = rng.choice([-1.0, 1.0], size=n)
    X = Lc + delta * B
    D = (rng.uniform(size=n) < e_fn(X)).astype(float)
    Y = np.where(D == 1, mu1(X), mu0(X)) + rng.normal(size=n)
    return Lc, X, D, Y


def gen_off(n, rng, delta):
    Lc = rng.uniform(-L_HALF, L_HALF, n)
    B = rng.choice([-1.0, 1.0], size=n)
    X = Lc + delta * B
    D = (rng.uniform(size=n) < e_fn(Lc)).astype(float)   # e 只看 L
    Y = np.where(D == 1, mu1(X), mu0(X)) + rng.normal(size=n)
    return Lc, X, D, Y


def gen_a1(n, rng, delta, cD=0.2, cY=2.0):
    """⑧ 负例：违反 A1——U 同时进 D 与 Y。"""
    Lc = rng.uniform(-L_HALF, L_HALF, n)
    B = rng.choice([-1.0, 1.0], size=n)
    X = Lc + delta * B
    U = rng.normal(size=n)
    p = np.clip(A0 + A1 * X + cD * U, 0.02, 0.98)
    D = (rng.uniform(size=n) < p).astype(float)
    Y = np.where(D == 1, mu1(X), mu0(X)) + cY * U + rng.normal(size=n)
    return Lc, X, D, Y


# ── 2. 估计量（④）：target · oracle · naive ────────────────────────────────
def degree_schedule(n: int) -> int:
    """系列阶 k_n = 2 + floor(log10 n)（下限 4）：近似误差随 n 消失，A7 的 r_m→0。"""
    return max(4, 2 + int(math.floor(math.log10(n))))


def _leg_fit_predict(x_tr, y_tr, x_te, deg, scale):
    Vtr = np.polynomial.legendre.legvander(x_tr / scale, deg)
    coef, *_ = np.linalg.lstsq(Vtr, y_tr, rcond=None)
    return np.polynomial.legendre.legval(x_te / scale, coef)


def crossfit_nuisance(cov, D, Y, rng, K=5, deg=4, scale=None,
                      clip=(0.01, 0.99), y_clip=None):
    """K 折 cross-fitting：线性概率 ê(L)，分臂多项式 m̂_d(L)。

    clip: 概率截断区间；y_clip: 非 None 时把 m̂_d 截到 [-y_clip, y_clip]，
    使讨厌函数类在紧支撑上确定地一致有界（供 Remark C0 / A.10 使用）。
    """
    n = len(cov)
    if scale is None:
        scale = max(1.0, float(np.max(np.abs(cov))) * 1.1)
    folds = np.array_split(rng.permutation(n), K)
    m1h = np.empty(n)
    m0h = np.empty(n)
    eh = np.empty(n)
    for k in range(K):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(K) if j != k])
        Xtr = np.column_stack([np.ones(len(tr)), cov[tr]])
        coef, *_ = np.linalg.lstsq(Xtr, D[tr], rcond=None)
        eh[te] = np.clip(coef[0] + coef[1] * cov[te], clip[0], clip[1])
        for arm, out in ((1, m1h), (0, m0h)):
            idx = tr[D[tr] == arm]
            d_use = min(deg, max(1, len(idx) - 2))
            fit = _leg_fit_predict(cov[idx], Y[idx], cov[te], d_use, scale)
            out[te] = fit if y_clip is None else np.clip(fit, -y_clip, y_clip)
    return m1h, m0h, eh


def aipw_estimate(cov, D, Y, rng, K=5, deg=4, clip=(0.01, 0.99), y_clip=None):
    m1h, m0h, eh = crossfit_nuisance(cov, D, Y, rng, K, deg, clip=clip, y_clip=y_clip)
    psi = (m1h - m0h + D * (Y - m1h) / eh - (1 - D) * (Y - m0h) / (1 - eh))
    th = float(psi.mean())
    Vhat = float(np.mean((psi - th) ** 2))
    se = math.sqrt(max(Vhat, 1e-12) / len(D))
    return th, se, Vhat


def est_target(L, D, Y, rng, dbar):
    deg = degree_schedule(len(D))
    th, se, Vhat = aipw_estimate(L, D, Y, rng, K=5, deg=deg)
    half_naive = Z975 * se
    half_rob = half_naive + dbar**2 * gamma_bar_eff(dbar)
    return dict(th=th, se=se, Vhat=Vhat, half_naive=half_naive, half_rob=half_rob)


def est_oracle(X, D, Y, rng):
    th, se, Vhat = aipw_estimate(X, D, Y, rng, K=5, deg=3)
    return dict(th=th, se=se, Vhat=Vhat, half_naive=Z975 * se)


def est_naive_dm(D, Y):
    y1, y0 = Y[D == 1], Y[D == 0]
    th = float(y1.mean() - y0.mean())
    se = math.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    return dict(th=th, se=se, half_naive=Z975 * se)


def est_naive_lm(L, D, Y, rng):
    """naive 的第二读法：不校正污染、直接把 L 当全部协变量做 OLS 回归调整。"""
    Xd = np.column_stack([np.ones(len(L)), L, D])
    coef, *_ = np.linalg.lstsq(Xd, Y, rcond=None)
    resid = Y - Xd @ coef
    s2 = resid @ resid / (len(L) - 3)
    cov = s2 * np.linalg.inv(Xd.T @ Xd)
    return dict(th=float(coef[2]), se=float(math.sqrt(cov[2, 2])),
                half_naive=Z975 * math.sqrt(cov[2, 2]))


# ── 2b. 光谱 DGP：保留前 r 维、残差在 (P-r) 维正交补上（P0 修复的算例）──────
def spec_e(s):
    return 0.5 + SPEC_SLOPE * s


def spec_mu1(s):
    return 1.0 + 2.0 * s + 0.3 * s ** 2 + 0.05 * s ** 3


def spec_mu0(s):
    return 0.5 * s + 0.1 * s ** 2 + 0.02 * s ** 3


def spec_dmu1(s):
    return 2.0 + 0.6 * s + 0.15 * s ** 2


def spec_dmu0(s):
    return 0.5 + 0.2 * s + 0.06 * s ** 2


# 固定律嵌套设计：X_1 = √0.75 S_1，X_j = δ√λ_j^0 S_j (j≥2)；
# 对每个 r，L=(X_1..X_r)，B=(X_{r+1}..X_6)。同一个 δ 下 X 的律不随 r 变。
SPEC_VAR0 = np.concatenate(([0.75], SPEC_LAM[2:]))


def _spec_states_all():
    S = np.array(list(itertools.product([-1.0, 1.0], repeat=SPEC_P)))
    return S, 1.0 / S.shape[0]


def _spec_x(delta, S):
    lam = SPEC_VAR0.copy()
    lam[1:] *= delta ** 2
    return S * np.sqrt(lam)[None, :]


def spec_truths(r, delta):
    """固定律谱 DGP 的 θ0, θL, Γ, V 与类常数（2^6 精确枚举）。"""
    S, _ = _spec_states_all()
    X = _spec_x(delta, S)
    s = X @ SPEC_W
    ev = spec_e(s)
    u1, u0 = spec_mu1(s), spec_mu0(s)
    th0 = float(np.mean(u1 - u0))
    # 按 L=(X_1..X_r) 分组；组内是 2^{6-r} 个 B 状态
    keys = X[:, :r]
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    mdiff, weights, ebar_g, v1_g, v0_g, sL_g = [], [], [], [], [], []
    for k in range(len(uniq)):
        idx = np.where(inv == k)[0]
        ek, u1k, u0k = ev[idx], u1[idx], u0[idx]
        eb = float(ek.mean())
        m1 = float((ek * u1k).mean() / eb)
        m0 = float(((1 - ek) * u0k).mean() / (1 - eb))
        v1 = float((ek * (u1k - m1) ** 2).mean() / eb) + 1.0
        v0 = float(((1 - ek) * (u0k - m0) ** 2).mean() / (1 - eb)) + 1.0
        w = len(idx) / len(S)
        mdiff.append(m1 - m0); weights.append(w)
        ebar_g.append(eb); v1_g.append(v1); v0_g.append(v0)
        sL_g.append(float(uniq[k] @ SPEC_W[:r]))
    weights = np.asarray(weights); mdiff = np.asarray(mdiff)
    thL = float(np.sum(weights * mdiff))
    V = float(np.sum(weights * (mdiff - thL) ** 2)
              + np.sum(weights * (np.asarray(v1_g) / np.asarray(ebar_g)
                                  + np.asarray(v0_g) / (1 - np.asarray(ebar_g)))))
    # 一阶 Γ：点态 L 处的梯度 × 条件协方差；λ_i=δ²λ_i^0
    lam = SPEC_VAR0.copy(); lam[1:] *= delta ** 2
    gam_terms = []
    for k in range(len(uniq)):
        eb = ebar_g[k]; sL = sL_g[k]
        q = 0.0
        for i in range(r, SPEC_P):
            wi = SPEC_W[i]
            q += lam[i] * (spec_dmu1(sL) * wi * SPEC_SLOPE * wi / eb
                           + spec_dmu0(sL) * wi * SPEC_SLOPE * wi / (1 - eb))
        gam_terms.append(weights[k] * q)
    gam = float(np.sum(gam_terms))
    qB = float(np.sum(SPEC_VAR0[r:] * SPEC_W[r:] ** 2))
    # 类常数（在固定律的支撑上取上确界；所有秩共用同一支撑）
    M_e = SPEC_SLOPE * float(np.linalg.norm(SPEC_W))
    M_mu = float(max(np.abs(spec_dmu1(s)).max(), np.abs(spec_dmu0(s)).max())
                 * np.linalg.norm(SPEC_W))
    M2 = float(max(np.abs(0.6 + 0.3 * s).max(), np.abs(0.2 + 0.12 * s).max())
               * np.sum(SPEC_W ** 2))
    mu4 = float(np.sum(SPEC_VAR0[r:]) ** 2)
    eps = float(0.5 - SPEC_SLOPE * np.abs(s).max())
    lam_next = float(SPEC_VAR0[r])
    inv_eps = 1.0 / eps + 1.0 / (1.0 - eps)
    gam_bar = M_mu * M_e * lam_next * inv_eps
    c1 = 2.0 * M2 * M_e * mu4 ** 0.75          # M_2' = 0：e 是线性的
    c_tilde = inv_eps * c1
    return dict(th0=th0, thL=thL, gam=gam, V=V, bias=thL - th0, qB=qB,
                lam_next=lam_next, eps=eps, M_mu=M_mu, M_e=M_e, M2=M2,
                mu4=mu4, gam_bar=gam_bar, C_tilde=c_tilde,
                sup_s=float(np.abs(s).max()))


def gen_spectral(n, rng, delta, r):
    """固定律 X：X_1=√0.75 S_1，X_j=δ√λ_j^0 S_j (j≥2)；L=X[:, :r]。"""
    S = rng.integers(0, 2, size=(n, SPEC_P)) * 2.0 - 1.0
    X = _spec_x(delta, S)
    Lc = X[:, :r]
    s = X @ SPEC_W
    D = (rng.uniform(size=n) < spec_e(s)).astype(float)
    Y = np.where(D == 1, spec_mu1(s), spec_mu0(s)) + rng.normal(size=n)
    return Lc, X, s, D, Y


# ── 2c. 异方差 DGP：B = σ(L) S，σ 随 L 变；检验一般谱恒等式 ────────────────
HET_BASE, HET_SLOPE = 0.8, 0.5


def het_sigma(L):
    return HET_BASE + HET_SLOPE * L


def gen_hetero(n, rng, delta):
    L = rng.uniform(-L_HALF, L_HALF, n)
    S = rng.choice([-1.0, 1.0], size=n)
    B = het_sigma(L) * S
    X = L + delta * B
    D = (rng.uniform(size=n) < e_fn(X)).astype(float)
    Y = np.where(D == 1, mu1(X), mu0(X)) + rng.normal(size=n)
    return L, X, D, Y


def hetero_truths(delta):
    """异方差设计的真值：θ0, θL, V, 一阶 Γ（一般式与错误因子化式）。"""
    sig = het_sigma(LQ)
    xp, xm = LQ + delta * sig, LQ - delta * sig
    ep, em = e_fn(xp), e_fn(xm)
    u1p, u1m = mu1(xp), mu1(xm)
    u0p, u0m = mu0(xp), mu0(xm)
    ebar = 0.5 * (ep + em)
    m1 = (ep * u1p + em * u1m) / (ep + em)
    m0 = ((1 - ep) * u0p + (1 - em) * u0m) / (2 - ep - em)
    thL = float(np.sum(WQ * (m1 - m0)))
    th0 = float(np.sum(WQ * 0.5 * (u1p + u1m - u0p - u0m)))
    var1 = (ep * (u1p - m1) ** 2 + em * (u1m - m1) ** 2) / (ep + em) + 1.0
    var0 = ((1 - ep) * (u0p - m0) ** 2 + (1 - em) * (u0m - m0) ** 2) / (2 - ep - em) + 1.0
    V = float(np.sum(WQ * ((m1 - m0 - thL) ** 2 + var1 / ebar + var0 / (1 - ebar))))
    q = dmu1(LQ) * A1 / ebar + dmu0(LQ) * A1 / (1 - ebar)
    gam = float(np.sum(WQ * delta ** 2 * sig ** 2 * q))
    gam_factor = float(np.sum(WQ * delta ** 2 * sig ** 2) * np.sum(WQ * q))
    return dict(th0=th0, thL=thL, V=V, gam=gam, gam_factor=gam_factor,
                bias=thL - th0)


def mc_cell_hetero(n, delta, R, seed):
    tr = hetero_truths(delta)
    streams = SeedSequence(seed).spawn(R)
    est, se = [], []
    for s in streams:
        rng = default_rng(s)
        L, X, D, Y = gen_hetero(n, rng, delta)
        tt = est_target(L, D, Y, rng, delta)
        est.append(tt["th"]); se.append(tt["se"])
    est, se = np.asarray(est), np.asarray(se)
    c0 = cov_stats(est, se, tr["th0"])
    cL = cov_stats(est, se, tr["thL"])
    return [dict(kind="hetero", n=n, delta=delta, dbar=delta, estimator="target",
                 R=len(est), theta0=tr["th0"], thetaL=tr["thL"], V=tr["V"],
                 gam=tr["gam"], gam_factor=tr["gam_factor"],
                 bias_L=cL["bias"], bias_L_se=cL["bias_se"],
                 bias_0=c0["bias"], bias_0_se=c0["bias_se"],
                 coverage_0=c0["coverage"], coverage_0_se=c0["coverage_se"],
                 coverage_L=cL["coverage"], coverage_L_se=cL["coverage_se"],
                 mean_se=float(se.mean()), mean_half_naive=float((Z975 * se).mean()))]


def exp_hetero(args):
    rows = []
    for n in (4000, 16000):
        seed = MASTER_SEED + 9000 + n
        rows += mc_cell_hetero(n, 0.4, args.R_spec, seed)
    return rows


def _td_basis(L, deg):
    """总次数 ≤ deg 的张量 Legendre 基（坐标按 SPEC_L_HALF 缩放）。"""
    n, r = L.shape
    x = L / SPEC_L_HALF
    Pm = [np.polynomial.legendre.legvander(x[:, j], deg) for j in range(r)]
    cols = []
    for idx in itertools.product(range(deg + 1), repeat=r):
        if sum(idx) <= deg:
            col = np.ones(n)
            for j, k in enumerate(idx):
                col = col * Pm[j][:, k]
            cols.append(col)
    return np.column_stack(cols)


def _fit_nd(xtr, ytr, xte, deg):
    coef, *_ = np.linalg.lstsq(_td_basis(xtr, deg), ytr, rcond=None)
    return _td_basis(xte, deg) @ coef


def aipw_nd(L, D, Y, rng, J=5, deg=4):
    """r 维 L 上的 cross-fitted AIPW：ĉ(L) 线性、m̂_d(L) 总次数 ≤ deg。"""
    n = len(D)
    folds = np.array_split(rng.permutation(n), J)
    m1h, m0h, eh = np.empty(n), np.empty(n), np.empty(n)
    for k in range(J):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(J) if j != k])
        A = np.column_stack([np.ones(len(tr)), L[tr]])
        coef, *_ = np.linalg.lstsq(A, D[tr], rcond=None)
        eh[te] = np.clip(coef[0] + L[te] @ coef[1:], 0.02, 0.98)
        for arm, out in ((1, m1h), (0, m0h)):
            ids = tr[D[tr] == arm]
            out[te] = _fit_nd(L[ids], Y[ids], L[te], deg)
    psi = (m1h - m0h + D * (Y - m1h) / eh - (1 - D) * (Y - m0h) / (1 - eh))
    th = float(psi.mean())
    return th, math.sqrt(max(float(np.mean((psi - th) ** 2)), 1e-12) / n)


def est_target_fixed(L, D, Y, rng, dbar, deg=6):
    """固定阶 sieve 的稳健性检查（A7(iv) 的类条件）。"""
    th, se, Vhat = aipw_estimate(L, D, Y, rng, K=5, deg=deg)
    half_naive = Z975 * se
    return dict(th=th, se=se, Vhat=Vhat, half_naive=half_naive,
                half_rob=half_naive + dbar**2 * gamma_bar_eff(dbar))


def mc_cell_fixed(n, delta, R, seed, deg=6):
    tr = truths_main(delta)
    streams = SeedSequence(seed).spawn(R)
    est, se = [], []
    for s in streams:
        rng = default_rng(s)
        L, X, D, Y = gen_main(n, rng, delta)
        tt = est_target_fixed(L, D, Y, rng, delta, deg)
        est.append(tt["th"]); se.append(tt["se"])
    est, se = np.asarray(est), np.asarray(se)
    c0 = cov_stats(est, se, tr["th0"])
    cL = cov_stats(est, se, tr["thL"])
    return [dict(kind="fixed_deg", n=n, delta=delta, dbar=delta, estimator="target",
                 R=len(est), theta0=tr["th0"], thetaL=tr["thL"], V=tr["V"],
                 bias_L=cL["bias"], bias_L_se=cL["bias_se"],
                 bias_0=c0["bias"], bias_0_se=c0["bias_se"],
                 coverage_0=c0["coverage"], coverage_0_se=c0["coverage_se"],
                 coverage_L=cL["coverage"], coverage_L_se=cL["coverage_se"],
                 mean_se=float(se.mean()), mean_half_naive=float((Z975 * se).mean())),
            ]


def est_target_spectral(L, D, Y, rng, delta, tr):
    th, se = aipw_nd(L, D, Y, rng, J=5, deg=4)
    lead = delta ** 2 * tr["gam_bar"]
    rem = delta ** 3 * tr["C_tilde"]
    half_naive = Z975 * se
    return dict(th=th, se=se, half_naive=half_naive,
                radius_lead=lead, radius_rem=rem,
                half_rob=half_naive + lead + rem)


# ── 3. 单格 Monte Carlo ───────────────────────────────────────────────────
def cov_stats(est, se, truth):
    est = np.asarray(est)
    se = np.asarray(se)
    covered = np.abs(est - truth) <= Z975 * se
    p = float(covered.mean())
    return dict(bias=float(est.mean() - truth),
                bias_se=float(est.std(ddof=1) / math.sqrt(len(est))),
                rmse=float(math.sqrt(np.mean((est - truth) ** 2))),
                coverage=p,
                coverage_se=float(math.sqrt(max(p * (1 - p), 0.0) / len(est))))


def q95_se(vals, seed=12345, B=200):
    vals = np.asarray(vals)
    rng = default_rng(seed)
    boots = [np.quantile(rng.choice(vals, size=len(vals), replace=True), 0.95)
             for _ in range(B)]
    return float(np.std(boots, ddof=1))


def mc_cell(kind, n, delta, R, seed, dbar=None, cD=0.2, cY=1.0,
            oracle=True, deg=None):
    """跑一个 (DGP, n, δ) 格；返回各估计量的轨迹与真值。"""
    if dbar is None:
        dbar = delta
    if kind == "off":
        tr = truths_off(delta)
        gen = gen_off
    elif kind == "asym":
        tr = truths_asym(delta)
        gen = gen_asym
    else:
        tr = truths_main(delta)
        gen = gen_main if kind == "main" else (lambda n_, r, d: gen_a1(n_, r, d, cD, cY))
    streams = SeedSequence(seed).spawn(R)
    out = {k: {"est": [], "se": [], "rob": []}
           for k in ("target", "oracle", "naive", "naive_lm")}
    for s in streams:
        rng = default_rng(s)
        Lc, X, D, Y = gen(n, rng, delta)
        tt = est_target(Lc, D, Y, rng, dbar)
        out["target"]["est"].append(tt["th"])
        out["target"]["se"].append(tt["se"])
        out["target"]["rob"].append(tt["half_rob"])
        dm = est_naive_dm(D, Y)
        out["naive"]["est"].append(dm["th"])
        out["naive"]["se"].append(dm["se"])
        out["naive"]["rob"].append(dm["half_naive"])
        nl = est_naive_lm(Lc, D, Y, rng)
        out["naive_lm"]["est"].append(nl["th"])
        out["naive_lm"]["se"].append(nl["se"])
        out["naive_lm"]["rob"].append(nl["half_naive"])
        if oracle:
            oo = est_oracle(X, D, Y, rng)
            out["oracle"]["est"].append(oo["th"])
            out["oracle"]["se"].append(oo["se"])
            out["oracle"]["rob"].append(oo["half_naive"])
    return tr, out


def summarize_cell(tr, out, n, delta, dbar, kind):
    rows = []
    for name in ("target", "oracle", "naive", "naive_lm"):
        if not out[name]["est"]:
            continue
        a = np.asarray(out[name]["est"])
        se = np.asarray(out[name]["se"])
        rob = np.asarray(out[name]["rob"])
        c0 = cov_stats(a, se, tr["th0"])
        cL = cov_stats(a, se, tr["thL"])
        covered_rob = np.abs(a - tr["th0"]) <= rob
        p_rob = float(covered_rob.mean())
        qL = float(np.quantile(np.abs(a - tr["thL"]), 0.95))
        q0 = float(np.quantile(np.abs(a - tr["th0"]), 0.95))
        e0 = a - tr["th0"]
        rmse0 = float(np.sqrt(np.mean(e0**2)))
        rmse0_se = (float(np.std(e0**2, ddof=1) / np.sqrt(len(e0)) / (2 * rmse0))
                    if rmse0 > 0 else float("nan"))
        rows.append(dict(
            kind=kind, n=n, delta=delta, dbar=dbar, estimator=name, R=len(a),
            theta0=tr["th0"], thetaL=tr["thL"], V=tr["V"], gam=tr["gam"],
            bias_L=cL["bias"], bias_L_se=cL["bias_se"],
            bias_0=c0["bias"], bias_0_se=c0["bias_se"],
            rmse_0=c0["rmse"], rmse_0_se=rmse0_se, coverage_0=c0["coverage"],
            coverage_0_se=c0["coverage_se"],
            coverage_L=cL["coverage"], coverage_L_se=cL["coverage_se"],
            coverage_rob=p_rob,
            coverage_rob_se=float(math.sqrt(max(p_rob * (1 - p_rob), 0.0) / len(a))),
            q95_L=qL, q95_L_se=q95_se(a - tr["thL"]),
            q95_0=q0, q95_0_se=q95_se(a - tr["th0"]),
            mean_se=float(se.mean()),
            mean_half_naive=float((Z975 * se).mean()),
            mean_half_rob=float(rob.mean()),
            radius_abs=dbar**2 * gamma_bar_eff(dbar),
        ))
    return rows


# ── 4. 五个实验 ───────────────────────────────────────────────────────────
def exp_main(args):
    """D1（收敛速率/触底）+ D3 的固定 δ 对照 + oracle/naive 谱。"""
    rows = []
    n_grid = [250, 500, 1000, 2000, 4000, 8000, 16000, 32000]
    if args.quick:
        n_grid = [250, 1000, 4000]
    for delta in (0.0, 0.2, 0.3, 0.4):
        for n in n_grid:
            seed = MASTER_SEED + 1000 * int(delta * 10) + n
            tr, out = mc_cell("main", n, delta, args.R_main, seed,
                              oracle=(delta in (0.0, 0.3)))
            rows += summarize_cell(tr, out, n, delta, delta, "main")
            print(f"[main] δ={delta:.1f} n={n:6d} done")
    return rows


def exp_bias_delta(args):
    """偏差对 δ 的走向：与 δ²Γ 对照（D1 的偏差侧）。"""
    rows = []
    n = 2000
    for delta in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        seed = MASTER_SEED + 2000 + int(delta * 100)
        tr, out = mc_cell("main", n, delta, args.R_main, seed, oracle=False)
        rows += summarize_cell(tr, out, n, delta, delta, "bias_delta")
        print(f"[bias_delta] δ={delta:.1f} done")
    return rows


def exp_critical(args):
    """D2（分布）+ D3（临界序列上的覆盖）：δ_n = sqrt(κ) n^{-1/4}。"""
    rows = []
    n_grid = [500, 1000, 2000, 4000, 8000]
    if args.quick:
        n_grid = [500, 2000]
    for kappa in (0.0, 1.0, 4.0, 9.0):
        for n in n_grid:
            delta = math.sqrt(kappa) * n ** -0.25
            seed = MASTER_SEED + 3000 + int(kappa * 10) + n
            tr, out = mc_cell("main", n, delta, args.R_crit, seed, oracle=False)
            a = np.asarray(out["target"]["est"])
            z = math.sqrt(n) * (a - tr["th0"]) / math.sqrt(tr["V"])
            shift = kappa * tr["gam"] / math.sqrt(tr["V"])
            zs = np.sort(z)
            ecdf = np.arange(1, len(zs) + 1) / len(zs)
            ks_shift = float(np.max(np.abs(ecdf - _Phi(zs - shift))))
            ks_0 = float(np.max(np.abs(ecdf - _Phi(zs))))
            rr = summarize_cell(tr, out, n, delta, delta, f"critical_k{kappa:g}")
            for r in rr:
                if r["estimator"] == "target":
                    r.update(kappa=kappa, shift_pred=shift, pred_c=_c_pred(shift),
                             mean_z=float(z.mean()),
                             sd_z=float(z.std(ddof=1)),
                             skew_z=float(np.mean(((z - z.mean()) / z.std()) ** 3)),
                             kurt_z=float(np.mean(((z - z.mean()) / z.std()) ** 4)),
                             ks_shift=ks_shift, ks_0=ks_0)
                else:
                    r.update(kappa=kappa, shift_pred=shift)
            rows += rr
            print(f"[critical] κ={kappa:g} n={n:5d} δ={delta:.3f} "
                  f"mean_z={z.mean():+.3f} shift={shift:+.3f} "
                  f"cov={[r for r in rr if r['estimator']=='target'][0]['coverage_0']:.3f}")
    return rows


def exp_phase(args):
    """D4 的相变：δ_n = c n^{-1/4}，看朴素覆盖的转折是否落在 n^{-1/4} 标度。"""
    rows = []
    cs = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    ns = [1000, 8000]
    if args.quick:
        ns = [1000]
    for n in ns:
        for c in cs:
            delta = c * n ** -0.25
            seed = MASTER_SEED + 4000 + n + int(c * 10)
            tr, out = mc_cell("main", n, delta, args.R_phase, seed, oracle=False)
            rr = summarize_cell(tr, out, n, delta, delta, "phase")
            for r in rr:
                r.update(c=c, shift_pred=c**2 * tr["gam"] / math.sqrt(tr["V"]),
                         pred_c=_c_pred(c**2 * tr["gam"] / math.sqrt(tr["V"])))
            rows += rr
            tgt = [r for r in rr if r["estimator"] == "target"][0]
            print(f"[phase] n={n} c={c:.1f} δ={delta:.3f} "
                  f"cov_naive={tgt['coverage_0']:.3f} pred={tgt['pred_c']:.3f}")
    return rows


def exp_off(args):
    """D5：机制关掉（B ⟂ D | L）——朴素覆盖回到名义水平；K_n→∞ 仍成立。"""
    rows = []
    n = 2000
    for delta in (0.0, 0.2, 0.4):
        seed = MASTER_SEED + 5000 + int(delta * 100)
        tr, out = mc_cell("off", n, delta, args.R_off, seed, oracle=False)
        rows += summarize_cell(tr, out, n, delta, delta, "off")
        print(f"[off] δ={delta:.1f} done")
    for n in (1000, 10000, 100000):
        delta = n ** -0.1
        seed = MASTER_SEED + 5100 + n
        tr, out = mc_cell("off", n, delta, args.R_off, seed, oracle=False)
        rr = summarize_cell(tr, out, n, delta, delta, "off_kn")
        for r in rr:
            r.update(kappa=math.sqrt(n) * delta**2)
        rows += rr
        print(f"[off/Kn→∞] n={n} δ={delta:.3f} K_n={math.sqrt(n)*delta**2:.2f} done")
    return rows


def exp_asym(args):
    """P2-7b：非对称残差，三次余项不再被对称性消掉。"""
    rows = []
    for n, R in ((4000, args.R_main), (16000, args.R_spec)):
        seed = MASTER_SEED + 7000 + n
        tr, out = mc_cell("asym", n, 0.3, R, seed, oracle=False)
        rr = summarize_cell(tr, out, n, 0.3, 0.3, "asym")
        rows += rr
        tgt = [r for r in rr if r["estimator"] == "target"][0]
        print(f"[asym] n={n}: bias={tgt['bias_0']:+.4f} "
              f"(delta^2 Gamma={0.09*tr['gam']:+.4f}), cov_naive={tgt['coverage_0']:.3f}")
    return rows


def exp_negative(args):
    """⑧ 负例：违反 A1（U 同时进 D 与 Y）。"""
    rows = []
    for n in (1000, 4000):
        seed = MASTER_SEED + 6000 + n
        tr, out = mc_cell("a1", n, 0.3, args.R_neg, seed, oracle=False, cY=2.0)
        rr = summarize_cell(tr, out, n, 0.3, 0.3, "neg_a1")
        rows += rr
        tgt = [r for r in rr if r["estimator"] == "target"][0]
        print(f"[neg-A1] n={n}: bias={tgt['bias_0']:+.3f} "
              f"cov_naive={tgt['coverage_0']:.3f} cov_rob={tgt['coverage_rob']:.3f} "
              f"radius={tgt['radius_abs']:.3f}")
    return rows


# ── 4b. 光谱实验：秩 r 真正承重（⑥ spectral，P0 修复）────────────────────
def mc_cell_spectral(r, n, delta, R, seed):
    tr = spec_truths(r, delta)
    streams = SeedSequence(seed).spawn(R)
    est, se, lead, rem, robs = [], [], [], [], []
    for s in streams:
        rng = default_rng(s)
        Lc, X, _s, D, Y = gen_spectral(n, rng, delta, r)
        tt = est_target_spectral(Lc, D, Y, rng, delta, tr)
        est.append(tt["th"]); se.append(tt["se"])
        lead.append(tt["radius_lead"]); rem.append(tt["radius_rem"])
        robs.append(tt["half_rob"])
    est = np.asarray(est); se = np.asarray(se)
    rob = np.asarray(robs)
    c0 = cov_stats(est, se, tr["th0"])
    cL = cov_stats(est, se, tr["thL"])
    p_rob = float(np.mean(np.abs(est - tr["th0"]) <= rob))
    e0 = est - tr["th0"]
    rmse0 = float(np.sqrt(np.mean(e0 ** 2)))
    rmse0_se = (float(np.std(e0 ** 2, ddof=1) / np.sqrt(len(e0)) / (2 * rmse0))
                if rmse0 > 0 else float("nan"))
    return [dict(
        kind="spectral", n=n, delta=delta, dbar=delta, estimator="target",
        R=len(est), theta0=tr["th0"], thetaL=tr["thL"], V=tr["V"], gam=tr["gam"],
        bias_L=cL["bias"], bias_L_se=cL["bias_se"],
        bias_0=c0["bias"], bias_0_se=c0["bias_se"],
        rmse_0=rmse0, rmse_0_se=rmse0_se,
        coverage_0=c0["coverage"], coverage_0_se=c0["coverage_se"],
        coverage_L=cL["coverage"], coverage_L_se=cL["coverage_se"],
        coverage_rob=p_rob,
        coverage_rob_se=float(math.sqrt(max(p_rob * (1 - p_rob), 0.0) / len(est))),
        mean_se=float(se.mean()),
        mean_half_naive=float((Z975 * se).mean()), mean_half_rob=float(rob.mean()),
        radius_abs=float(np.mean(np.asarray(lead) + np.asarray(rem))),
        r_ret=r, lam_next=tr["lam_next"], qB=tr["qB"],
        radius_lead=float(np.mean(lead)), radius_rem=float(np.mean(rem)),
        eps_r=tr["eps"], gam_bar_r=tr["gam_bar"], C_tilde_r=tr["C_tilde"],
    )]


def exp_spectral(args):
    """H1'：偏差 = Σ_{j>r}ν_j a_j；诚实半径 ∝ δ²λ_{r+1}；秩规则。"""
    rows = []
    ns = (4000, 16000)
    if args.quick:
        ns = (4000,)
    for r in (1, 2, 3, 4):
        for delta in (0.4, 0.5):
            for n in ns:
                seed = MASTER_SEED + 7000 + 100 * r + 10 * int(delta * 10) + n
                rr = mc_cell_spectral(r, n, delta, args.R_spec, seed)
                rows += rr
                t = rr[0]
                print(f"[spectral] r={r} δ={delta:.1f} n={n:5d}: "
                      f"bias={t['bias_0']:+.4f} (δ²Γ={delta**2*t['gam']:+.4f}) "
                      f"cov_naive={t['coverage_0']:.3f} "
                      f"cov_rob={t['coverage_rob']:.3f} "
                      f"radius={t['radius_abs']:.3f} "
                      f"(lead {t['radius_lead']:.3f}+rem {t['radius_rem']:.3f}) "
                      f"λ_next={t['lam_next']:.3f}")
    return rows


def spectral_rank_rule(rows):
    """临界秩：r*(n) = min{r : δ²λ_{r+1} ≤ n^{-1/2}}（保守的充分条件）。"""
    out = {}
    for r in rows:
        if r["kind"] != "spectral":
            continue
        n, delta = r["n"], r["delta"]
        if n not in out:
            cand = []
            for rr in rows:
                if rr["kind"] == "spectral" and rr["n"] == n and abs(rr["delta"] - delta) < 1e-12:
                    cand.append((rr["r_ret"], delta ** 2 * rr["lam_next"]))
            thr = n ** -0.5
            out[n] = (delta, thr, min([x for x, v in cand if v <= thr], default=None))
    return out


# ── 5. P9 两点构造（修复版；D4 的下界侧，代数核对）────────────────────────
def p9_check(delta=0.3, n=4000, lam=1.0, M_mu=None, M_e=None):
    """P9（修复版）：两潜结构共享同一观测律、θ0 相差 Δ。

    结构 B（混淆）：B=σ_B S u，e_B=½+δM_eσ_B S，μ_d^B=μ_d^0+δM_μσ_B S。
    结构 A（约化式）：B≡0，e_A≡½，μ_d^A(L)=m_d^B(L)=μ_d^0±2δ²M_μM_eσ_B²，
    且 Y(d)|L 取与 B 完全相同的两点混合（权重 ½±δM_eσ_B）。
    于是 P(L,D,Y) 相同、θ0^A=θ_L 精确成立、θ0^B=θ_L−Δ，Δ=4δ²M_μM_eσ_B²。
    """
    M_mu = M_MU if M_mu is None else M_mu
    M_e = M_E if M_e is None else M_e
    sigma2_B = lam
    Delta = 4 * delta ** 2 * M_mu * M_e * sigma2_B
    m1_obs = 1.0 + 2 * delta ** 2 * M_mu * M_e * sigma2_B
    m0_obs = 0.0 - 2 * delta ** 2 * M_mu * M_e * sigma2_B
    thL = m1_obs - m0_obs
    th0_B = 1.0 - 0.0
    th0_A = thL
    sig2 = delta ** 2 * M_mu ** 2 * sigma2_B * (1 - 4 * delta ** 2 * M_e ** 2 * sigma2_B)
    c0 = 1.0  # fixed outcome-noise variance, so inf_P V >= 4 c0 > 0
    V_p9 = 4 * (sig2 + c0)
    half = Z975 * math.sqrt(V_p9 / n) + delta ** 2 * gamma_bar_eff(delta)
    bound = (1 - 0.1) * Delta / 2.0
    return dict(delta=delta, n=n, Delta=Delta, thetaL=thL, theta0_A=th0_A,
                theta0_B=th0_B, gap=th0_A - th0_B, V_p9=V_p9,
                halfwidth_rob=half, lower_bound=bound, ratio=half / bound,
                lam_next=sigma2_B, m1_obs=m1_obs, m0_obs=m0_obs,
                shift_A=2 * delta ** 2 * M_mu * M_e * sigma2_B)


# ── 6. 输出：CSV · summary · 图 ───────────────────────────────────────────
def _Phi(z):
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(z) / math.sqrt(2.0)))


def _c_pred(shift):
    return float(_Phi(Z975 - shift) - _Phi(-Z975 - shift))


FIELDS = ["kind", "n", "delta", "dbar", "kappa", "c", "estimator", "R",
          "theta0", "thetaL", "V", "gam", "bias_L", "bias_L_se", "bias_0",
          "bias_0_se", "rmse_0", "rmse_0_se", "coverage_0", "coverage_0_se", "coverage_L",
          "coverage_L_se", "coverage_rob", "coverage_rob_se", "q95_L",
          "q95_L_se", "q95_0", "q95_0_se", "mean_se", "mean_half_naive",
          "mean_half_rob", "radius_abs", "shift_pred", "mean_z", "sd_z",
          "skew_z", "kurt_z", "ks_shift", "ks_0", "pred_c",
          "r_ret", "lam_next", "qB", "radius_lead", "radius_rem", "eps_r",
          "gam_bar_r", "C_tilde_r", "gam_factor"]


def load_results(path):
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            for k in FIELDS:
                if k in ("kind", "estimator"):
                    continue
                if r.get(k, "") != "":
                    try:
                        r[k] = float(r[k])
                    except ValueError:
                        pass
            if "n" in r and r["n"] != "":
                r["n"] = int(float(r["n"]))
            rows.append(r)
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"表：{path}")


def _pick(rows, kind, est="target", n=None, delta=None, kappa=None, c=None):
    for r in rows:
        if r["kind"] != kind or r["estimator"] != est:
            continue
        if n is not None and r["n"] != n:
            continue
        if delta is not None and abs(r["delta"] - delta) > 1e-12:
            continue
        if kappa is not None and abs(r.get("kappa", -1) - kappa) > 1e-12:
            continue
        if c is not None and abs(r.get("c", -1) - c) > 1e-12:
            continue
        return r
    return None


def _pick_n(rows, kind, est="target", n_pref=None, **kw):
    ns = sorted({r["n"] for r in rows if r["kind"] == kind and r["estimator"] == est})
    n = n_pref if (n_pref in ns) else (max(ns) if ns else None)
    return _pick(rows, kind, est=est, n=n, **kw) if n is not None else None


def _main_radius(dbar, sigma_bar=1.0):
    """主 DGP 的诚实半径：δ̄² M_μM_e Σ̄_B(1/ε+1/(1-ε)) + C̃δ̄³，ε 取类常数 0.15。"""
    inv = 1.0 / E_EPS + 1.0 / (1.0 - E_EPS)
    c1 = (2 * M_MU * M2P * MU4 ** 0.75 + 2 * M2 * M_E * MU4 ** 0.75
          + M2 * M2P * MU4)
    return dbar ** 2 * M_MU * M_E * sigma_bar * inv + dbar ** 3 * inv * c1


def constants_and_breakdown():
    """常数、R(δ̄,Σ̄_B) 灵敏度表与 breakdown 值（对应 §5.6）。

    半径用类常数 ε=0.15（为 δ_max=0.7 取的保守值）；同时报告该 δ̄ 处
    支撑上的**有效**重叠常数 ε_bind(δ̄)=0.5-0.15(1.5+δ̄)。
    """
    rows = []
    inv = 1.0 / E_EPS + 1.0 / (1.0 - E_EPS)
    c1 = (2 * M_MU * M2P * MU4 ** 0.75 + 2 * M2 * M_E * MU4 ** 0.75
          + M2 * M2P * MU4)
    for dbar in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        eps_bind = max(0.02, 0.5 - A1 * (L_HALF + dbar))
        for sig in (0.5, 1.0, 2.0):
            rows.append(dict(dbar=dbar, sigma_bar=sig, eps=E_EPS,
                             eps_bind=eps_bind, inv_eps=inv,
                             gamma_bar=M_MU * M_E * sig * inv,
                             C_tilde=inv * c1, radius=_main_radius(dbar, sig)))
    grid = np.linspace(0.0, 0.8, 8001)
    br = {}
    for tau in (0.05, 0.1, 0.2):
        ok = [_d for _d in grid if _main_radius(_d) <= tau]
        br[tau] = max(ok) if ok else 0.0
    return rows, br


def make_summary(rows, p9, path):
    lines = []
    # D1：伪参数的 n^{-1/2} 速率 + 目标的 δ² 触底
    ns = sorted({r["n"] for r in rows if r["kind"] == "main" and r["estimator"] == "target"
                 and abs(r["delta"] - 0.3) < 1e-12})
    qs = [_pick(rows, "main", n=n, delta=0.3)["q95_L"] for n in ns]
    slope = float(np.polyfit(np.log(ns), np.log(qs), 1)[0]) if len(ns) >= 2 else float("nan")
    qfloor = _pick(rows, "main", n=max(ns), delta=0.2)
    qclean = _pick(rows, "main", n=max(ns), delta=0.3)
    lines.append(f"D1：|θ̂_L−θ_L| 的 95% 分位 log-log 斜率 = {slope:+.3f}"
                 f"（n={ns[0]}–{ns[-1]}，δ=0.3）；"
                 f"δ=0.3 时 n={max(ns)} 的分位 = {qclean['q95_L']:.4f}±{qclean['q95_L_se']:.4f}；"
                 f"δ=0.2 时 |θ̂−θ0| 的分位在 n={max(ns)} 触底于 {qfloor['q95_0']:.4f}"
                 f"（偏差 δ²Γ={qfloor['thetaL']-qfloor['theta0']:.4f}）")
    # D2：临界序列的位移与正态性
    d2 = _pick_n(rows, "critical_k4", n_pref=4000)
    lines.append(f"D2：κ=4, n=4000：mean(z)={d2['mean_z']:+.3f}（预言位移 {d2['shift_pred']:+.3f}），"
                 f"sd(z)={d2['sd_z']:.3f}，KS(N(shift,1))={d2['ks_shift']:.3f}，"
                 f"KS(N(0,1))={d2['ks_0']:.3f}")
    # D3：覆盖
    d3 = _pick_n(rows, "critical_k4", n_pref=4000)
    lines.append(f"D3：κ=4, n=4000：朴素覆盖 = {d3['coverage_0']:.3f}±{d3['coverage_0_se']:.3f}"
                 f"（预言 c(κ)={d3['pred_c']:.3f}）；CI_rob 覆盖 = {d3['coverage_rob']:.3f}"
                 f"±{d3['coverage_rob_se']:.3f}（预言 ≥0.95）")
    # D4：相变 + P9
    ph1 = _pick(rows, "phase", n=1000, c=2.0)
    ph2 = _pick(rows, "phase", n=8000, c=2.0) or ph1
    ph1b = _pick(rows, "phase", n=1000, c=2.5)
    ph2b = _pick(rows, "phase", n=8000, c=2.5) or ph1b
    lines.append(f"D4：相变 c=2 时朴素覆盖 n=1000: {ph1['coverage_0']:.3f} "
                 f"(pred {ph1['pred_c']:.3f})，n=8000: {ph2['coverage_0']:.3f} "
                 f"(pred {ph2['pred_c']:.3f})；c=2.5: {ph1b['coverage_0']:.3f} vs "
                 f"{ph2b['coverage_0']:.3f}（pred {ph1b['pred_c']:.3f}/{ph2b['pred_c']:.3f}）；"
                 f"P9 Δ={p9['Delta']:.4f}，CI_rob 半宽={p9['halfwidth_rob']:.3f}"
                 f"≥下界 {p9['lower_bound']:.3f}")
    sp9 = _pick(rows, "spectral", n=4000, delta=0.4)
    if sp9 is not None:
        tr1 = spec_truths(1, 0.4)
        D_s = 4 * 0.16 * tr1["M_mu"] * tr1["M_e"] * tr1["lam_next"]
        lines.append(f"光谱 P9（r=1, δ=0.4）：Δ=4δ²MμMeλ_2={D_s:.4f}，"
                     f"CI_rob 半宽={sp9['mean_half_rob']:.3f} ≥ 下界 "
                     f"(1-2α)Δ/2={0.9 * D_s / 2:.3f}")
    # D5：机制关掉
    off4 = _pick(rows, "off", n=2000, delta=0.4) or _pick_n(rows, "off", delta=0.4)
    kn = _pick_n(rows, "off_kn", n_pref=100000)
    lines.append(f"D5：机制关掉 δ=0.4：目标朴素覆盖 = {off4['coverage_0']:.3f}"
                 f"±{off4['coverage_0_se']:.3f}（名义 0.95），CI_rob 覆盖 = "
                 f"{off4['coverage_rob']:.3f}；δ_n=n^(-1/10), n=1e5（K_n={kn['kappa']:.1f}）："
                 f"朴素覆盖 = {kn['coverage_0']:.3f}±{kn['coverage_0_se']:.3f}")
    # 负例
    asym = _pick_n(rows, "asym", n_pref=4000)
    if asym is not None:
        lines.append(f"非对称残差（P(B=1)=0.7）：bias_0={asym['bias_0']:+.4f}，"
                     f"delta^2 Gamma={0.09*asym['gam']:+.4f}，CI_rob 覆盖={asym['coverage_rob']:.3f}")
    neg = _pick_n(rows, "neg_a1", n_pref=4000)
    lines.append(f"负例：违反 A1（U 同时进 D 与 μ_D）——目标偏差 {neg['bias_0']:+.3f}"
                 f"（朝上/高估），诚实半径 {neg['radius_abs']:.3f} 被突破，"
                 f"CI_rob 覆盖 {neg['coverage_rob']:.3f}，且不随 n 缩小")
    # 光谱实验（⑥）：秩 r 的因果代价
    for n in (4000, 16000):
        rr = sorted([r for r in rows if r["kind"] == "spectral" and r["n"] == n
                     and abs(r["delta"] - 0.4) < 1e-12], key=lambda z: z["r_ret"])
        if rr:
            parts = "；".join(
                f"r={t['r_ret']}: bias={t['bias_0']:+.4f}±{t['bias_0_se']:.4f}"
                f"（δ²Γ={t['gam']:+.4f}），cov={t['coverage_0']:.3f}，"
                f"R={t['radius_abs']:.3f}（主项 {t['radius_lead']:.3f}+余项 {t['radius_rem']:.3f}），"
                f"λ_next={t['lam_next']:.3f}" for t in rr)
            lines.append(f"光谱（δ=0.4, n={n}）：{parts}")
    # 异方差设计：一般式 vs 错误因子化
    for h in [r for r in rows if r["kind"] == "hetero"]:
        lines.append(f"异方差（δ=0.4, n={h['n']}）：bias={h['bias_0']:+.4f}±{h['bias_0_se']:.4f}，"
                     f"一般式 δ²Γ={h['gam']:+.4f}，错误因子化={h['gam_factor']:+.4f}")
    # 固定阶 sieve 稳健性（A7(iv) 的类条件）
    fd = [r for r in rows if r["kind"] == "fixed_deg"]
    if fd:
        parts = "；".join(f"n={r['n']}: bias={r['bias_0']:+.4f}±{r['bias_0_se']:.4f}, "
                         f"cov={r['coverage_0']:.3f}" for r in fd)
        lines.append(f"固定阶稳健性（deg=6, δ=0.3）：{parts}")
    # 秩规则：保证秩（λ_{r+1}）与对齐加权秩（qB）
    rule = []
    for n in (4000, 16000):
        thr = n ** -0.5
        cand = sorted({(r["r_ret"], r["delta"], r["lam_next"], r["qB"]) for r in rows
                       if r["kind"] == "spectral" and r["n"] == n})
        for delta in sorted({c[1] for c in cand}):
            sub = [c for c in cand if abs(c[1] - delta) < 1e-12]
            ok_lam = [c[0] for c in sub if delta ** 2 * c[2] <= thr]
            ok_q = [c[0] for c in sub if delta ** 2 * c[3] <= thr]
            rule.append(f"n={n},δ={delta:.1f}: 保证 r*={min(ok_lam) if ok_lam else '>4'}，"
                        f"对齐加权 r*={min(ok_q) if ok_q else '>4'}")
    lines.append("秩规则（δ²λ_{r+1}≤n^{-1/2}；对齐加权用 δ²q_B）：" + "；".join(rule))
    # 类常数、灵敏度与 breakdown
    lines.append(f"类常数：Mμ={M_MU} Me={M_E} Σ̄_B={SIGMA_B} μ4={MU4} "
                 f"M2={M2} M2'={M2P}；ε=0.15（为 δ_max=0.7 取），"
                 f"δ=0.3 处支撑上的有效 ε_bind=0.23")
    crows, br = constants_and_breakdown()
    lines.append("breakdown（ε=0.15）：τ=0.05→δ̄*={:.3f}，τ=0.10→{:.3f}，τ=0.20→{:.3f}"
                 .format(br[0.05], br[0.1], br[0.2]))
    sel = [c for c in crows if c["sigma_bar"] == 1.0]
    lines.append("R(δ̄,Σ̄_B=1，ε=0.15)：" + "，".join(
        f"δ̄={c['dbar']:.1f}→{c['radius']:.3f}（ε_bind={c['eps_bind']:.2f}）"
        for c in sel))
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    print("\n" + text)
    return text


def make_plots(rows, p9, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from scipy.stats import norm
        ppf = norm.ppf
    except Exception:  # scipy 不在时 QQ 图跳过
        ppf = None

    # 速率图
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for delta, marker in ((0.2, "o"), (0.3, "s"), (0.4, "^")):
        ns, qs = [], []
        for r in rows:
            if r["kind"] == "main" and r["estimator"] == "target" and abs(r["delta"] - delta) < 1e-12:
                ns.append(r["n"]); qs.append(r["q95_L"])
        order = np.argsort(ns); ns = np.asarray(ns)[order]; qs = np.asarray(qs)[order]
        ax.loglog(ns, qs, marker + "-", label=f"δ={delta}")
    nn = np.array([250, 32000])
    anchor = _pick(rows, "main", n=250, delta=0.3)
    if anchor is not None:
        ax.loglog(nn, anchor["q95_L"] * (nn / 250.0) ** -0.5, "k--", lw=1,
                  label=r"$n^{-1/2}$")
    ax.set_xlabel("n"); ax.set_ylabel(r"95% quantile of $|\hat\theta_L-\theta_L|$")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(outdir / "rate.png", dpi=130); plt.close(fig)

    # 偏差 vs δ
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    rr = [r for r in rows if r["kind"] == "bias_delta" and r["estimator"] == "target"]
    ds = [r["delta"] for r in rr]
    ax.plot(ds, [r["bias_0"] for r in rr], "o-", label="target bias(θ0)")
    ax.plot(ds, [r["thetaL"] - r["theta0"] for r in rr], "k--", label=r"$\delta^2\Gamma$")
    ax.set_xlabel(r"$\delta$"); ax.set_ylabel("bias")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(outdir / "bias.png", dpi=130)
    plt.close(fig)

    # 覆盖率 vs n
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for delta, marker in ((0.0, "o"), (0.2, "s"), (0.3, "^"), (0.4, "d")):
        ns, cov, se = [], [], []
        for r in rows:
            if r["kind"] == "main" and r["estimator"] == "target" and abs(r["delta"] - delta) < 1e-12:
                ns.append(r["n"]); cov.append(r["coverage_0"]); se.append(r["coverage_0_se"])
        order = np.argsort(ns); ns = np.asarray(ns)[order]
        ax.errorbar(ns, np.asarray(cov)[order], yerr=1.96 * np.asarray(se)[order],
                    fmt=marker + "-", capsize=2, label=f"target naive δ={delta}")
    for est, style, lab in (("oracle", "x-", "oracle (δ=0.3)"),
                            ("target", ":", "CI_rob (δ=0.3)")):
        ns, cov = [], []
        for r in rows:
            if r["kind"] == "main" and r["estimator"] == est and abs(r["delta"] - 0.3) < 1e-12:
                ns.append(r["n"])
                cov.append(r["coverage_0"] if est == "oracle" else r["coverage_rob"])
        order = np.argsort(ns)
        ax.plot(np.asarray(ns)[order], np.asarray(cov)[order], style, lw=1, label=lab)
    ax.axhline(0.95, color="grey", ls="--", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("n"); ax.set_ylabel("coverage of θ0")
    # P3-5a: 同一张图上给出 naive 半宽（右轴），让读者看到宽度代价
    ax2 = ax.twinx()
    ns_hw, hw = [], []
    for r in rows:
        if r["kind"] == "main" and r["estimator"] == "target" and abs(r["delta"] - 0.3) < 1e-12:
            ns_hw.append(r["n"]); hw.append(r["mean_half_naive"])
    order = np.argsort(ns_hw)
    ax2.plot(np.asarray(ns_hw)[order], np.asarray(hw)[order], "k--", lw=1,
             label="mean naive 95% half-width (δ=0.3)")
    ax2.set_ylabel("mean naive 95% half-width")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7)
    fig.tight_layout(); fig.savefig(outdir / "coverage.png", dpi=130); plt.close(fig)

    # 临界序列：mean(z) 与覆盖
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for kappa, marker in ((0.0, "o"), (1.0, "s"), (4.0, "^"), (9.0, "d")):
        rr = [r for r in rows if r["kind"] == f"critical_k{kappa:g}" and r["estimator"] == "target"]
        rr = sorted(rr, key=lambda r: r["n"])
        if not rr:
            continue
        axes[0].plot([r["n"] for r in rr], [r["mean_z"] for r in rr], marker + "-",
                     label=f"κ={kappa:g}")
        axes[1].plot([r["n"] for r in rr], [r["coverage_0"] for r in rr], marker + "-",
                     label=f"κ={kappa:g}")
    axes[0].set_xscale("log"); axes[0].set_xlabel("n"); axes[0].set_ylabel("mean(z)")
    axes[0].legend(fontsize=8)
    axes[1].axhline(0.95, color="grey", ls="--", lw=1)
    axes[1].set_xscale("log"); axes[1].set_xlabel("n"); axes[1].set_ylabel("coverage")
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(outdir / "critical.png", dpi=130); plt.close(fig)

    # 相变
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for n, marker in ((1000, "o"), (8000, "s")):
        rr = [r for r in rows if r["kind"] == "phase" and r["estimator"] == "target" and r["n"] == n]
        rr = sorted(rr, key=lambda r: r["c"])
        if rr:
            ax.plot([r["c"] for r in rr], [r["coverage_0"] for r in rr], marker + "-", label=f"n={n}")
    ax.axhline(0.95, color="grey", ls="--", lw=1)
    ax.set_xlabel(r"$c$  ($\delta=c\,n^{-1/4}$)"); ax.set_ylabel("coverage of θ0")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(outdir / "phase.png", dpi=130); plt.close(fig)

    # QQ（κ=4, n=4000）
    qq = _pick(rows, "critical_k4", n=4000)
    if qq is not None and ppf is not None:
        # 重跑一个轻量格子拿 z（为了不让主循环存大数组，这里用同种子重算 400 个复制）
        tr, out = mc_cell("main", 4000, math.sqrt(4.0) * 4000 ** -0.25, 1000,
                          MASTER_SEED + 3000 + 40 + 4000, oracle=False)
        a = np.asarray(out["target"]["est"])
        z = np.sqrt(4000) * (a - tr["th0"]) / math.sqrt(tr["V"])
        zs = np.sort(z)
        probs = (np.arange(1, len(zs) + 1) - 0.5) / len(zs)
        fig, ax = plt.subplots(figsize=(5.6, 4.6))
        ax.plot(ppf(probs), zs, "o", ms=3, label="empirical")
        ax.plot(ppf(probs), ppf(probs) + qq["shift_pred"], "k-", lw=1, label=r"$N(\kappa\Gamma/\sqrt{V},1)$")
        ax.plot(ppf(probs), ppf(probs), "r--", lw=1, label=r"$N(0,1)$")
        ax.set_xlabel("theoretical quantile"); ax.set_ylabel("z")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(outdir / "qq.png", dpi=130); plt.close(fig)

    # 负例
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    rr = [r for r in rows if r["kind"] == "neg_a1" and r["estimator"] == "target"]
    rr = sorted(rr, key=lambda r: r["n"])
    ax.plot([r["n"] for r in rr], [r["coverage_0"] for r in rr], "o-", label="naive CI")
    ax.plot([r["n"] for r in rr], [r["coverage_rob"] for r in rr], "s-", label="CI_rob")
    ax.axhline(0.95, color="grey", ls="--", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("n"); ax.set_ylabel("coverage of θ0")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(outdir / "negative.png", dpi=130); plt.close(fig)

    # 光谱：偏差/半径与覆盖随保留秩 r 的变化
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for n, marker in ((4000, "o"), (16000, "s")):
        rr = sorted([r for r in rows if r["kind"] == "spectral" and r["n"] == n
                     and abs(r["delta"] - 0.4) < 1e-12], key=lambda z: z["r_ret"])
        if not rr:
            continue
        axes[0].plot([t["r_ret"] for t in rr], [t["bias_0"] for t in rr],
                     marker + "-", label=f"bias, n={n}")
        axes[0].plot([t["r_ret"] for t in rr],
                     [t["radius_lead"] + t["radius_rem"] for t in rr],
                     marker + "--", label=f"honest radius, n={n}")
        axes[1].errorbar([t["r_ret"] for t in rr], [t["coverage_0"] for t in rr],
                         yerr=[1.96 * t["coverage_0_se"] for t in rr],
                         fmt=marker + "-", capsize=2, label=f"naive, n={n}")
        axes[1].plot([t["r_ret"] for t in rr], [t["coverage_rob"] for t in rr],
                     marker + ":", label=f"CI_rob, n={n}")
    axes[0].set_xlabel("retained rank r"); axes[0].set_ylabel("bias / radius")
    axes[0].legend(fontsize=7)
    axes[1].axhline(0.95, color="grey", ls="--", lw=1)
    axes[1].set_xlabel("retained rank r"); axes[1].set_ylabel("coverage of θ0")
    axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(outdir / "rank.png", dpi=130); plt.close(fig)
    print(f"图：{outdir}/*.png")


# ── 7. 主程序 ─────────────────────────────────────────────────────────────
def main():
    global MASTER_SEED
    ap = argparse.ArgumentParser(description="h001 Monte Carlo（econometric-simu）")
    ap.add_argument("--quick", action="store_true", help="快速冒烟（小 R / 小网格）")
    ap.add_argument("--seed", type=int, default=MASTER_SEED)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
    ap.add_argument("--R-main", type=int, default=1000)
    ap.add_argument("--R-crit", type=int, default=1000)
    ap.add_argument("--R-phase", type=int, default=1000)
    ap.add_argument("--R-off", type=int, default=1000)
    ap.add_argument("--R-neg", type=int, default=1000)
    ap.add_argument("--R-spec", type=int, default=500)
    ap.add_argument("--plot-only", action="store_true",
                    help="只从 results/results.csv 重画表与图，不重跑 Monte Carlo")
    args = ap.parse_args()
    if args.quick:
        args.R_main = args.R_crit = args.R_phase = args.R_off = args.R_neg = 60
        args.R_spec = 40

    MASTER_SEED = args.seed
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        rows = load_results(outdir / "results.csv")
        with (outdir / "results_p9.csv").open() as f:
            p9 = {k: float(v) for k, v in next(csv.DictReader(f)).items()}
        make_summary(rows, p9, outdir / "summary.txt")
        make_plots(rows, p9, outdir)
        return

    # 真值与常数自检
    t = truths_main(0.3)
    print(f"[truth] δ=0.3: θ0={t['th0']:.5f} θL={t['thL']:.5f} "
          f"bias={t['bias']:.5f} δ²Γ={0.3**2*t['gam']:.5f} V={t['V']:.4f}")
    print(f"[const] Mμ={M_MU} Me={M_E} ε={E_EPS} Γ̄={GAMMA_BAR:.4f} C̃={C_TILDE:.4f}")

    rows = []
    rows += exp_main(args)
    rows += exp_bias_delta(args)
    rows += exp_critical(args)
    rows += exp_phase(args)
    rows += exp_off(args)
    rows += exp_asym(args)
    rows += exp_negative(args)
    rows += exp_spectral(args)
    rows += exp_hetero(args)

    p9 = p9_check()
    write_csv(outdir / "results.csv", rows)
    with (outdir / "results_p9.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(p9.keys()))
        w.writeheader(); w.writerow(p9)
    print(f"表：{outdir / 'results_p9.csv'}")
    make_summary(rows, p9, outdir / "summary.txt")
    make_plots(rows, p9, outdir)


if __name__ == "__main__":
    main()
