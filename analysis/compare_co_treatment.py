#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compare_co_treatment.py — 并排对比 CO 的两种训练口径：
  上排：训练只用 CO>3σ 的像素，预测用原始 CO（负值截 0）
  下排：训练全样本，CO/σ<=3 的置 0（现行）
6 个面板：观测 / 预测 / 相对误差，各一行。
"""
import os, argparse
import numpy as np
import h5py
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def fit(ra, dec, X, y, sig, in_tr, fit_mask, spa):
    raw = np.column_stack([X] + spa)
    m = fit_mask.copy()
    for j in range(raw.shape[1]):
        m &= np.isfinite(raw[:, j])
    floors = np.array([max(1e-9, 0.01 * np.median(X[m, j][X[m, j] > 0])) for j in range(X.shape[1])])
    Z = np.column_stack([np.log10(X + floors[None, :])] + spa)
    mu, sd = Z[m].mean(0), Z[m].std(0) + 1e-12
    D = np.column_stack([np.ones(len(Z)), (Z - mu) / sd])
    w = 1.0 / sig[m] ** 2
    c = np.linalg.solve(D[m].T @ (w[:, None] * D[m]), D[m].T @ (w * y[m]))
    return D @ c, c, D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', default='/home/xzhang/0_Stanford/paper_0/results_final/M31_1d_array_final.h5')
    ap.add_argument('--mask', default='/home/xzhang/0_Stanford/paper_0/results_final/mask_arm_final.npz')
    ap.add_argument('--out', default='/home/xzhang/0_Stanford/paper_0/results_final/fig_co_compare.png')
    args = ap.parse_args()

    with h5py.File(args.h5) as f:
        d = {k: f[k][:] for k in ['ra', 'dec', 'qpah', 'qpah_err', 'CO_sigma',
                                  'H1', 'sfr', 'CO', 'H1_ew', 'Ldust']}
    y = d['qpah'].astype(float)
    sig = np.sqrt(d['qpah_err'].astype(float) ** 2 + 0.2 ** 2)
    ra, dec = d['ra'].astype(float), d['dec'].astype(float)
    RA0, DEC0, INC, PA = 10.6847, 41.2690, np.radians(77.7), np.radians(37.7)
    dx = (ra - RA0) * np.cos(np.radians(DEC0)); dy = dec - DEC0
    xm = (dx * np.sin(PA) + dy * np.cos(PA)) * 3600 / (206265 / 744)
    ym = ((-dx * np.cos(PA) + dy * np.sin(PA)) / np.cos(INC)) * 3600 / (206265 / 744)
    spa = [xm[:, None], ym[:, None]]
    X = np.maximum(np.column_stack([d[k].astype(float) for k in ['H1', 'sfr', 'CO', 'H1_ew', 'Ldust']]), 0.0)
    snr = X[:, 2] / np.maximum(d['CO_sigma'].astype(float), 1e-30)

    np.random.seed(0)
    ang = (np.arctan2(dec - 41.25, ra - 10.75) + 2 * np.pi) % (2 * np.pi)
    sec = np.digitize(ang, np.linspace(0, 2 * np.pi, 11)) - 1
    trs = np.random.choice(np.arange(10), size=6, replace=False)
    in_tr = np.isin(sec, trs)

    zm = np.load(args.mask); m = zm['region'].astype(bool)
    t = cKDTree(np.column_stack([ra, dec]))
    dd, ii = t.query(np.column_stack([zm['ra'][m], zm['dec'][m]]), k=1)
    ARM = np.zeros(len(y), bool); ARM[ii[dd < 1e-7]] = True

    COr = d['CO'].astype(float); SGc = d['CO_sigma'].astype(float)
    out = {}
    # (1) 软阈值：max(CO - 3sigma, 0)  —— 连续，无自由参数
    Xs = X.copy()
    Xs[:, 2] = np.maximum(COr - 3.0 * SGc, 0.0)
    out['soft'] = fit(ra, dec, Xs, y, sig, in_tr, in_tr.copy(), spa)
    # (2) 现行硬 cut：CO/sigma<=3 置 0
    Xz = X.copy(); Xz[snr <= 3, 2] = 0.0
    out['cur'] = fit(ra, dec, Xz, y, sig, in_tr, in_tr.copy(), spa)

    nrm_q = Normalize(0, 6.0)
    nrm_c = Normalize(-3, 3)
    fig, axes = plt.subplots(2, 3, figsize=(19, 12.6))
    for row, (tag, lab) in enumerate([('soft', 'SOFT THRESHOLD:  $\\mathrm{CO}_{\\rm eff}=\\max(\\mathrm{CO}-3\\sigma,\\,0)$'),
                                      ('cur', 'CURRENT HARD CUT:  $\\mathrm{CO}:=0$ where CO/$\\sigma$ $\\leq$ 3')]):
        pred, c, D = out[tag]
        chi = (pred - y) / sig
        order = np.argsort(np.abs(chi))[::-1]
        panels = [(r'Observation  $q_{\rm PAH}$', y, 'inferno', nrm_q, r'$q_{\rm PAH}$ (%)'),
                  (r'Prediction  $q_{\rm PAH}$', pred, 'inferno', nrm_q, r'$q_{\rm PAH}$ (%)'),
                  (r'$\chi=(\mathrm{pred}-\mathrm{obs})/\sigma_{\rm eff}$', chi, 'coolwarm', nrm_c, r'$\chi$')]
        for a, (tt, val, cm, nm, cl) in zip(axes[row], panels):
            sc = a.scatter(ra[order], dec[order], s=1.0, c=val[order], cmap=cm, norm=nm,
                           rasterized=True, linewidths=0)
            a.scatter(ra[ARM], dec[ARM], s=2.6, facecolors='none', edgecolors='lime',
                      linewidths=0.3, rasterized=True)
            a.set_title(tt, fontsize=13); a.set_aspect('equal'); a.invert_xaxis()
            a.set_xlabel('RA (J2000)'); a.grid(True, ls=':', alpha=.35)
            fig.colorbar(sc, ax=a, fraction=0.046, pad=0.02).set_label(cl, fontsize=11)
        axes[row][0].set_ylabel('Dec (J2000)')
        p = (pred - y) / sig
        te = ~in_tr
        axes[row][1].text(0.02, 0.02,
                          '$\\chi^2$: fit-set %.4f | train %.4f | test %.4f | all %.4f\n'
                          'arm: $\\chi^2$=%.3f  pull=%+.3f   (n=%d)'
                          % (np.mean(p[in_tr] ** 2),
                             np.mean(p[in_tr] ** 2), np.mean(p[te] ** 2), np.mean(p ** 2),
                             np.mean(p[ARM] ** 2), np.median(p[ARM]), ARM.sum()),
                          transform=axes[row][1].transAxes, fontsize=11,
                          bbox=dict(fc='white', alpha=.85, ec='none'))
        axes[row][0].text(0.02, 0.87, lab, transform=axes[row][0].transAxes, fontsize=12,
                          bbox=dict(fc='white', alpha=.9, ec='gray'))
    fig.suptitle('CO treatment: soft threshold vs hard cut  (same data, same features)', fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=140, bbox_inches='tight')
    print('saved', args.out)


if __name__ == '__main__':
    main()
