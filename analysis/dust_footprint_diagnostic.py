#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dust_footprint_diagnostic.py — 诊断"SCUBA-2 视场内 / 外"两片区域的模型表现差异。

背景
----
把数据集从"要求 dust 有限"放宽后，样本 72,469 -> 116,339 行，但测试 χ²_red
从 1.01 恶化到 1.29。分区域看：

    有 dust 覆盖（SCUBA-2 视场内, n=25,453）  χ²_red = 1.003
    无 dust 覆盖（视场外,          n=16,663）  χ²_red = 1.721

本脚本回答关键问题：视场外的恶化是【沿（去投影）径向单调变差】——那就是真实的
物理/金属度效应，本身可发表；还是【随机分布】——那更可能是标签或误差问题。

模型：log10 线性（4 特征，不含 dust），误差加权最小二乘，与 linear_baseline.py 同口径。

输出:
  fig_dustfootprint_spatial.png  覆盖掩膜 / χ 空间分布 / |χ|>2 掩膜
  fig_dustfootprint_radial.png   χ²_red、pull、qpah & qpah_err 随半径的变化
"""
import os, argparse
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mplconfig')
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

FEATS = ['H1', 'sfr', 'CO', 'H1_ew']
# M31 盘几何（Draine+2014 采用值）
RA0, DEC0 = 10.6847, 41.2690
INC, PA = np.radians(77.7), np.radians(37.7)
ARCSEC_PER_KPC = 206265.0 / 744.0          # 744 kpc 距离下，1 kpc = 277.2"


def deprojected_radius_kpc(ra, dec):
    """去投影（盘面）半径，单位 kpc。"""
    dx = (ra - RA0) * np.cos(np.radians(DEC0))
    dy = dec - DEC0
    x = dx * np.cos(PA) + dy * np.sin(PA)          # 长轴
    y = -dx * np.sin(PA) + dy * np.cos(PA)         # 短轴
    y /= np.cos(INC)
    return np.sqrt(x ** 2 + y ** 2) * 3600.0 / ARCSEC_PER_KPC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', default='data/M31_1d_array_nodust.h5')
    ap.add_argument('--err-floor', type=float, default=0.2)
    ap.add_argument('--nbins', type=int, default=14)
    ap.add_argument('--out', default='results/dustfootprint')
    args = ap.parse_args()

    with h5py.File(args.h5, 'r') as f:
        keys = ['ra', 'dec', 'qpah', 'qpah_err', 'CO_sigma', 'dust_density'] + FEATS
        d = {k: f[k][:] for k in keys if k in f}
    sel = (d['qpah'] >= 0) & (d['qpah_err'] > 0)
    for k in FEATS:
        sel &= np.isfinite(d[k])
    sel &= (d['H1'] >= 0) & (d['H1_ew'] >= 0) & np.isfinite(d['CO_sigma']) & (d['CO_sigma'] > 0)
    ra, dec = d['ra'][sel].astype(float), d['dec'][sel].astype(float)
    X = np.maximum(np.column_stack([d[k][sel] for k in FEATS]).astype(float), 0.0)
    y = d['qpah'][sel].astype(float)
    ye = d['qpah_err'][sel].astype(float)
    dust = d['dust_density'][sel].astype(float)
    has_dust = np.isfinite(dust) & (dust > 0)
    sig = np.sqrt(ye ** 2 + args.err_floor ** 2)
    R = deprojected_radius_kpc(ra, dec)

    np.random.seed(0)
    ang = (np.arctan2(dec - 41.25, ra - 10.75) + 2 * np.pi) % (2 * np.pi)
    sec = np.digitize(ang, np.linspace(0, 2 * np.pi, 11)) - 1
    tr_s = np.random.choice(np.arange(10), size=6, replace=False)
    in_tr = np.isin(sec, tr_s)

    floors = np.array([max(1e-9, 0.01 * np.median(X[in_tr, j][X[in_tr, j] > 0]))
                       for j in range(len(FEATS))])
    Z = np.log10(X + floors[None, :])
    mu, sd = Z[in_tr].mean(0), Z[in_tr].std(0) + 1e-12
    Ztr = np.column_stack([np.ones(int(in_tr.sum())), (Z[in_tr] - mu) / sd])
    Zte = np.column_stack([np.ones(int((~in_tr).sum())), (Z[~in_tr] - mu) / sd])
    w = 1.0 / sig[in_tr] ** 2
    coef = np.linalg.solve(Ztr.T @ (w[:, None] * Ztr), Ztr.T @ (w * y[in_tr]))
    pred = Zte @ coef
    pull = (pred - y[~in_tr]) / sig[~in_tr]

    Rte, hd = R[~in_tr], has_dust[~in_tr]
    print('测试像素 %d | 有 dust %d | 无 dust %d' % (len(pull), hd.sum(), (~hd).sum()))
    print('R(kpc): 全体 med=%.1f p84=%.1f | 有dust med=%.1f | 无dust med=%.1f'
          % (np.median(Rte), np.percentile(Rte, 84),
             np.median(Rte[hd]), np.median(Rte[~hd])))

    os.makedirs(args.out, exist_ok=True)

    # ================= 图 1：空间 =================
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.6))
    order = np.argsort(pull)                       # 让离群点画在上面
    ra_te, dec_te = ra[~in_tr], dec[~in_tr]
    ax[0].scatter(ra_te[hd], dec_te[hd], s=1.0, c='steelblue',
                  label='dust-covered (SCUBA-2)', rasterized=True)
    ax[0].scatter(ra_te[~hd], dec_te[~hd], s=1.0, c='crimson',
                  label='no dust', rasterized=True)
    ax[0].set_title('850um coverage (test pixels)')
    ax[0].set_xlabel('RA (deg)'); ax[0].set_ylabel('Dec (deg)')
    ax[0].legend(markerscale=8, fontsize=9)

    norm = Normalize(-3, 3)
    ax[1].scatter(ra_te[order], dec_te[order], s=1.2,
                  c=pull[order], cmap='coolwarm', norm=norm, rasterized=True)
    ax[1].set_title(r'$\chi=(\mathrm{pred}-\mathrm{obs})/\sigma_{\rm eff}$ (test)')
    ax[1].set_xlabel('RA (deg)')

    big = np.abs(pull) > 2
    ax[2].scatter(ra_te[hd], dec_te[hd], s=0.6, c='lightgray', rasterized=True)
    ax[2].scatter(ra_te[~hd], dec_te[~hd], s=0.6, c='lightblue', rasterized=True)
    ax[2].scatter(ra_te[big], dec_te[big], s=1.6, c='crimson', rasterized=True)
    ax[2].set_title(r'$|\chi|>2$ (red) over coverage mask (gray=dust, blue=none)')
    ax[2].set_xlabel('RA (deg)')
    for a in ax[1:]:
        a.invert_xaxis(); a.set_aspect('equal')
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='coolwarm'), ax=ax[1],
                 fraction=0.046, pad=0.02, label=r'$\chi$')
    fig.tight_layout()
    p1 = os.path.join(args.out, 'fig_dustfootprint_spatial.png')
    fig.savefig(p1, dpi=150, bbox_inches='tight'); plt.close(fig)
    print('saved', p1)

    # ================= 图 2：径向 =================
    rmax = np.percentile(Rte, 99.5)
    edges = np.linspace(0, rmax, args.nbins + 1)
    bc = 0.5 * (edges[:-1] + edges[1:])

    def binned(mask):
        c2, pm, ps, n, ep, qm = [], [], [], [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            m = mask & (Rte >= a) & (Rte < b)
            n.append(int(m.sum()))
            if m.sum() >= 30:
                c2.append(float(np.mean(pull[m] ** 2)))
                pm.append(float(np.median(pull[m]))); ps.append(float(np.std(pull[m])))
                ep.append(float(np.median(ye[~in_tr][m])))
                qm.append(float(np.median(y[~in_tr][m])))
            else:
                c2.append(np.nan); pm.append(np.nan); ps.append(np.nan)
                ep.append(np.nan); qm.append(np.nan)
        return map(np.array, (c2, pm, ps, n, ep, qm))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for nm, m, col in [('dust-covered', hd, 'steelblue'), ('no dust', ~hd, 'crimson')]:
        c2, pm, ps, n, ep, qm = binned(m)
        ax[0].plot(bc, c2, 'o-', color=col, lw=2, ms=5, label=nm)
        ax[1].plot(bc, pm, 'o-', color=col, lw=2, ms=5, label=nm + ' median')
        ax[1].plot(bc, ps, 's--', color=col, lw=1.6, ms=4, alpha=0.75, label=nm + ' std')
        ax[2].plot(bc, qm, 'o-', color=col, lw=2, ms=5, label=nm + ' qPAH')
        ax[2].plot(bc, ep, 's--', color=col, lw=1.6, ms=4, alpha=0.75, label=nm + ' err')
    ax[0].axhline(1.0, color='k', ls='--', lw=1)
    ax[0].set_ylabel(r'$\chi^2_{\rm red}$'); ax[0].set_title('goodness of fit vs radius')
    ax[0].legend(fontsize=9); ax[0].grid(alpha=.3)
    ax[1].axhline(0.0, color='k', ls='--', lw=1); ax[1].axhline(1.0, color='k', ls=':', lw=1)
    ax[1].set_ylabel('pull'); ax[1].set_title('pull median (ideal 0) and std (ideal 1)')
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    ax[2].set_ylabel('qPAH and its error (%)')
    ax[2].set_title('qPAH level and error vs radius')
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
    for a in ax:
        a.set_xlabel('R (kpc, deprojected)')
    fig.tight_layout()
    p2 = os.path.join(args.out, 'fig_dustfootprint_radial.png')
    fig.savefig(p2, dpi=150, bbox_inches='tight'); plt.close(fig)
    print('saved', p2)

    # ---- 文字摘要 ----
    print('\n%-6s %8s %8s %8s | %8s %8s %8s' %
          ('R(kpc)', 'n_has', 'chi2_has', 'std_has', 'n_no', 'chi2_no', 'std_no'))
    for a, b in zip(edges[:-1], edges[1:]):
        mh = hd & (Rte >= a) & (Rte < b)
        mn = (~hd) & (Rte >= a) & (Rte < b)
        f = lambda m: (np.mean(pull[m] ** 2), np.std(pull[m])) if m.sum() >= 30 else (np.nan, np.nan)
        ch, sh = f(mh); cn, sn = f(mn)
        print('%-6.1f %8d %8.3f %8.3f | %8d %8.3f %8.3f'
              % (0.5 * (a + b), mh.sum(), ch, sh, mn.sum(), cn, sn))


if __name__ == '__main__':
    main()
