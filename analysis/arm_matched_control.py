#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
arm_matched_control.py — 同半径对照：把"高误差悬臂"和【同一半径环上其余方位】的像素比。

为什么要做
----------
已知 χ² 随半径强烈变化（R<5 kpc ≈0.5，R>12 kpc ≈2-5）。
所以"这条臂误差大"可能只是因为它在外侧。要做半径匹配的对照才能分离出
【这条臂在该半径上是否真的特殊】。

注意：半径 R 只用于【分析配对】，绝不作为模型特征（模型只用观测量）。

输出:
  fig_arm_matched_control.png
     (a) 观测 qPAH vs R：区域 vs 同半径对照
     (b) pull vs R：区域 vs 对照
     (c) 特征 z（区域−对照）vs R
     (d) pull 分布（区域 vs 匹配对照）
"""
import os, argparse
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mplconfig')
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FEATS = ['H1', 'sfr', 'CO', 'H1_ew', 'Ldust']   # 定稿模型
RA0, DEC0, INC, PA = 10.6847, 41.2690, np.radians(77.7), np.radians(37.7)


def deproj_R(ra, dec):
    dx = (ra - RA0) * np.cos(np.radians(DEC0)); dy = dec - DEC0
    x = dx * np.sin(PA) + dy * np.cos(PA)
    y = (-dx * np.cos(PA) + dy * np.sin(PA)) / np.cos(INC)
    return np.sqrt(x ** 2 + y ** 2) * 3600.0 / (206265.0 / 744.0)


def boot_med(x, n=400, seed=0):
    if len(x) < 5:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    b = np.median(x[idx], axis=1)
    return float(np.median(x)), float(np.percentile(b, 16)), float(np.percentile(b, 84))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', default='data/M31_1d_array_nodust.h5')
    ap.add_argument('--mask-npz', required=True)
    ap.add_argument('--err-floor', type=float, default=0.2)
    ap.add_argument('--step', type=float, default=0.5, help='半径匹配的环宽 (kpc)')
    ap.add_argument('--outdir', default='results/armregion/arm_matched')
    args = ap.parse_args()

    with h5py.File(args.h5, 'r') as f:
        keys = ['ra', 'dec', 'qpah', 'qpah_err', 'CO_sigma'] + FEATS
        d = {k: f[k][:] for k in keys if k in f}
    sel = (d['qpah'] >= 0) & (d['qpah_err'] > 0)
    for k in FEATS:
        sel &= np.isfinite(d[k])
    sel &= (d['H1'] >= 0) & (d['H1_ew'] >= 0) & np.isfinite(d['CO_sigma']) & (d['CO_sigma'] > 0)
    ra, dec = d['ra'][sel].astype(float), d['dec'][sel].astype(float)
    X = np.maximum(np.column_stack([d[k][sel] for k in FEATS]).astype(float), 0.0)
    y = d['qpah'][sel].astype(float)
    sig = np.sqrt(d['qpah_err'][sel].astype(float) ** 2 + args.err_floor ** 2)

    zm = np.load(args.mask_npz)
    assert len(zm['region']) == len(ra), 'mask 长度不符'
    region = zm['region'].astype(bool)
    R = deproj_R(ra, dec)

    np.random.seed(0)
    ang = (np.arctan2(dec - 41.25, ra - 10.75) + 2 * np.pi) % (2 * np.pi)
    sec = np.digitize(ang, np.linspace(0, 2 * np.pi, 11)) - 1
    tr_s = np.random.choice(np.arange(10), size=6, replace=False)
    in_tr = np.isin(sec, tr_s)

    floors = np.array([max(1e-9, 0.01 * np.median(X[in_tr, j][X[in_tr, j] > 0]))
                       for j in range(X.shape[1])])
    Z = np.log10(X + floors[None, :]); mu, sd = Z[in_tr].mean(0), Z[in_tr].std(0) + 1e-12
    Zs = (Z - mu) / sd
    D = np.column_stack([np.ones(len(Z)), Zs])
    w = 1.0 / sig[in_tr] ** 2
    coef = np.linalg.solve(D[in_tr].T @ (w[:, None] * D[in_tr]), D[in_tr].T @ (w * y[in_tr]))
    pred = D @ coef
    pull = (pred - y) / sig
    resid = pred - y

    rlo, rhi = np.percentile(R[region], [2, 98])
    edges = np.arange(np.floor(rlo / args.step) * args.step, rhi + args.step, args.step)
    print('区域 %d 像素，R ∈ [%.1f, %.1f] kpc；按 %.1f kpc 环匹配对照'
          % (region.sum(), rlo, rhi, args.step))
    print()
    print('%9s %6s %6s | %13s %13s | %11s %11s | %10s' %
          ('R(kpc)', 'n_reg', 'n_ctl', 'pull_reg', 'pull_ctl',
           'qPAH_reg', 'qPAH_ctl', 'dchi2'))
    print('-' * 110)
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        mr = region & (R >= a) & (R < b)
        mc = (~region) & (R >= a) & (R < b)
        if mr.sum() < 10 or mc.sum() < 50:
            continue
        pr = pull[mr]; pc = pull[mc]
        qr = y[mr]; qc = y[mc]
        dz = {f: np.median(Zs[mr, j]) - np.median(Zs[mc, j]) for j, f in enumerate(FEATS)}
        rows.append(dict(R=0.5 * (a + b), nr=int(mr.sum()), nc=int(mc.sum()),
                         pr=np.median(pr), pc=np.median(pc),
                         qr=np.median(qr), qc=np.median(qc),
                         c2r=np.mean(pr ** 2), c2c=np.mean(pc ** 2), dz=dz))
        print('%9.1f %6d %6d | %+13.3f %+13.3f | %11.2f %11.2f | %+10.3f'
              % (0.5 * (a + b), mr.sum(), mc.sum(), np.median(pr), np.median(pc),
                 np.median(qr), np.median(qc), np.mean(pr ** 2) - np.mean(pc ** 2)))

    if not rows:
        raise SystemExit('没有可配对的半径环')

    # 汇总（按区域像素数加权）
    wr = np.array([r['nr'] for r in rows], float)
    d_pull = np.array([r['pr'] - r['pc'] for r in rows])
    d_qpah = np.array([r['qr'] - r['qc'] for r in rows])
    print()
    print('=== 半径匹配后的差异（按区域像素加权） ===')
    print('pull(区域−对照) 中位 = %+.3f   （>0 = 区域被【高估】；<0 = 被【低估】）'
          % np.average(d_pull, weights=wr))
    print('qPAH(区域−对照) 中位 = %+.3f %%' % np.average(d_qpah, weights=wr))
    for j, f in enumerate(FEATS):
        dz = np.array([r['dz'][f] for r in rows])
        print('  z(%s) 区域−对照 = %+.3f σ' % (f, np.average(dz, weights=wr)))

    # 全局：区域 vs 全体对照（不匹配半径，作对比）
    print()
    print('（未做半径匹配的朴素对比）pull 区域 %+.3f vs 其余 %+.3f'
          % (np.median(pull[region]), np.median(pull[~region])))

    # ---------------- 图 ----------------
    os.makedirs(args.outdir, exist_ok=True)
    Rr = [r['R'] for r in rows]
    fig, ax = plt.subplots(1, 4, figsize=(23, 5.2))

    ax[0].plot(Rr, [r['qr'] for r in rows], 'o-', color='crimson', lw=2, label='arm region')
    ax[0].plot(Rr, [r['qc'] for r in rows], 's--', color='gray', lw=2, label='matched control')
    ax[0].set_ylabel(r'observed $q_{\rm PAH}$ (%)'); ax[0].set_title('observed qPAH vs R')
    ax[0].legend(); ax[0].grid(alpha=.3)

    ax[1].plot(Rr, [r['pr'] for r in rows], 'o-', color='crimson', lw=2, label='arm region')
    ax[1].plot(Rr, [r['pc'] for r in rows], 's--', color='gray', lw=2, label='matched control')
    ax[1].axhline(0, color='k', ls='--', lw=1)
    ax[1].set_ylabel('pull median'); ax[1].set_title('pull vs R (>0 over-predicted)')
    ax[1].legend(); ax[1].grid(alpha=.3)

    for j, f in enumerate(FEATS):
        ax[2].plot(Rr, [r['dz'][f] for r in rows], 'o-', lw=1.8, ms=4, label=f)
    ax[2].axhline(0, color='k', ls='--', lw=1)
    ax[2].set_ylabel(r'median $z$ (region $-$ control)')
    ax[2].set_title('feature offsets at matched R'); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)

    mr_all = region & (R >= rlo) & (R <= rhi)
    mc_all = (~region) & (R >= rlo) & (R <= rhi)
    bins = np.linspace(-4, 4, 50)
    ax[3].hist(pull[mc_all], bins=bins, density=True, color='lightgray', label='control')
    ax[3].hist(pull[mr_all], bins=bins, density=True, histtype='step', color='crimson',
               lw=2, label='arm region')
    ax[3].axvline(np.median(pull[mc_all]), color='k', ls=':', lw=1)
    ax[3].axvline(np.median(pull[mr_all]), color='crimson', ls='--', lw=1)
    ax[3].set_title('pull distribution (same R range)'); ax[3].legend(fontsize=9)
    ax[3].set_xlabel('pull')
    for a in ax[:3]:
        a.set_xlabel('R (kpc, deprojected)')
    fig.tight_layout()
    f = os.path.join(args.outdir, 'fig_arm_matched_control.png')
    fig.savefig(f, dpi=160, bbox_inches='tight'); plt.close(fig)
    print('saved', f)


if __name__ == '__main__':
    main()
