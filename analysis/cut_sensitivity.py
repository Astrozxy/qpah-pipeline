#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cut_sensitivity.py — 当前模型几乎没做数据质量 cut；本脚本量化"边缘/低质量像素"贡献了多少 χ²。

背景
----
现有筛选只有：qpah>0、qpah_err>0、特征有限（⇒ coverage>0.5）、H1/H1_ew>=0、CO_sigma>0。
**没有**对 qpah_err 大小、coverage 阈值(仅 0.5)、到 footprint 边缘的距离做任何限制。

本脚本回答：
  1. |χ| 是否与 qpah_err 正相关？（坏点是不是"本来就不确定"的点）
  2. 提高 coverage 阈值 / 限制 qpah_err / 要求离边缘足够远，χ² 会怎么变？
  3. 那条悬臂（#35-51）在不同 cut 下还剩多少？

做法：**固定同一个模型**（在基础样本的训练扇区上拟合），只是在不同子集上评估 χ²，
这样能干净地把"χ² 里有多少来自低质量像素"归因出来。

输出:
  fig_cut_sensitivity.png
     (a) 中位|χ| vs qpah_err   (b) χ² vs coverage 阈值
     (c) qpah_err 空间分布      (d) |χ|>2 的像素（按 qpah_err 着色）
"""
import os, argparse
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mplconfig')
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from scipy import ndimage

FEATS = ['H1', 'sfr', 'CO', 'H1_ew']
COV_KEY = {'H1': 'HI', 'sfr': 'SFR', 'CO': 'CO', 'H1_ew': 'HI_ew', 'dust_density': 'dust'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', default='data/M31_1d_array_nodust.h5')
    ap.add_argument('--data', default='data')
    ap.add_argument('--cache', default='sanity_cache')
    ap.add_argument('--err-floor', type=float, default=0.2)
    ap.add_argument('--aux', default=None,
                    help='本地紧凑包 aux_sample.npz（含 cov_*/edge_* 每行值）；'
                         '给了就不读 sanity_cache，可完全离线运行')
    ap.add_argument('--mask-npz', default=None)
    ap.add_argument('--outdir', default='results/cut_sensitivity')
    args = ap.parse_args()

    with h5py.File(args.h5, 'r') as f:
        keys = ['ra', 'dec', 'qpah', 'qpah_err', 'CO_sigma'] + FEATS
        d = {k: f[k][:] for k in dict.fromkeys(keys) if k in f}
    sel = ((d['qpah'] > 0) & (d['qpah_err'] > 0) & np.isfinite(d['H1']) & (d['H1'] >= 0)
           & np.isfinite(d['H1_ew']) & (d['H1_ew'] >= 0) & np.isfinite(d['sfr'])
           & np.isfinite(d['CO']) & np.isfinite(d['CO_sigma']) & (d['CO_sigma'] > 0))
    ra, dec = d['ra'][sel].astype(float), d['dec'][sel].astype(float)
    X = np.maximum(np.column_stack([d[k][sel] for k in FEATS]).astype(float), 0.0)
    y = d['qpah'][sel].astype(float)
    ye = d['qpah_err'][sel].astype(float)
    sig = np.sqrt(ye ** 2 + args.err_floor ** 2)

    region = None
    if args.mask_npz:
        zm = np.load(args.mask_npz)
        if len(zm['region']) == len(ra):
            region = zm['region'].astype(bool)

    # ---- coverage / 边缘距 ----
    if args.aux:
        za = np.load(args.aux)
        if len(za['cov_H1']) != len(y):
            raise SystemExit('aux 长度 %d != 样本 %d（筛选口径不一致）'
                             % (len(za['cov_H1']), len(y)))
        cov_min = np.min(np.column_stack([za['cov_' + f] for f in FEATS]),
                         axis=1).astype(float)
        edge_min = np.min(np.column_stack([za['edge_' + f] for f in FEATS]),
                          axis=1).astype(float)
        print('（coverage/边缘距 来自 %s）' % args.aux)
    else:
        hq = fits.open(os.path.join(args.data, 'M31_S350_110_SSS_110_Model_All_qpah.fits'))
        w = WCS(hq[0].header).celestial
        shape = hq[0].data.shape
        hq.close()
        xx, yy = w.world_to_pixel_values(ra, dec)
        ix = np.clip(np.round(xx).astype(int), 0, shape[1] - 1)
        iy = np.clip(np.round(yy).astype(int), 0, shape[0] - 1)

        covs, edges = {}, {}
        for f_ in FEATS:
            k = COV_KEY[f_]
            p = os.path.join(args.cache, k + '_ex_cov.npy')
            m = os.path.join(args.cache, k + '_ex_mean.npy')
            if os.path.exists(p):
                cv = np.load(p)
                covs[f_] = cv[iy, ix]
                foot = (np.isfinite(np.load(m)) & (cv > 0.5)
                        if os.path.exists(m) else (cv > 0.5))
                edges[f_] = ndimage.distance_transform_edt(foot)[iy, ix]
        cov_min = np.min(np.column_stack(list(covs.values())), axis=1)
        edge_min = np.min(np.column_stack(list(edges.values())), axis=1)

    # ---- 固定模型（基础样本训练扇区）----
    np.random.seed(0)
    ang = (np.arctan2(dec - 41.25, ra - 10.75) + 2 * np.pi) % (2 * np.pi)
    sec = np.digitize(ang, np.linspace(0, 2 * np.pi, 11)) - 1
    tr_s = np.random.choice(np.arange(10), size=6, replace=False)
    in_tr = np.isin(sec, tr_s)
    floors = np.array([max(1e-9, 0.01 * np.median(X[in_tr, j][X[in_tr, j] > 0]))
                       for j in range(X.shape[1])])
    Z = np.log10(X + floors[None, :])
    mu, sd = Z[in_tr].mean(0), Z[in_tr].std(0) + 1e-12
    D = np.column_stack([np.ones(len(Z)), (Z - mu) / sd])
    wt = 1.0 / sig[in_tr] ** 2
    coef = np.linalg.solve(D[in_tr].T @ (wt[:, None] * D[in_tr]), D[in_tr].T @ (wt * y[in_tr]))
    chi = (D @ coef - y) / sig
    achi = np.abs(chi)

    print('基础样本 n=%d | χ²_all=%.4f | χ²_test=%.4f | 中位 qpah_err=%.3f'
          % (len(y), np.mean(chi ** 2), np.mean(chi[~in_tr] ** 2), np.median(ye)))
    print('coverage(min over 4 feats): 中位=%.3f, <0.9 占 %.1f%%, <0.7 占 %.1f%%'
          % (np.median(cov_min), 100 * np.mean(cov_min < 0.9), 100 * np.mean(cov_min < 0.7)))
    print('到最近 footprint 边缘的中位距离 = %.1f px' % np.median(edge_min))

    # ---- 1. |χ| vs qpah_err ----
    print()
    print('=== 中位 |χ| 随 qpah_err 的变化（回答"坏点是不是本来就不确定的点"）===')
    qs = np.percentile(ye, np.linspace(0, 100, 11)); qs[-1] += 1e-9
    bi = np.digitize(ye, qs[:-1]) - 1
    print('%14s %9s %10s %12s %12s' % ('qpah_err 区间', 'n', '中位|χ|', '|χ|>2 占比', '|χ|>3 占比'))
    for b in range(10):
        m = bi == b
        if m.sum() < 30:
            continue
        print('%6.2f-%6.2f %9d %10.3f %11.1f%% %11.1f%%'
              % (qs[b], qs[b + 1], m.sum(), np.median(achi[m]),
                 100 * np.mean(achi[m] > 2), 100 * np.mean(achi[m] > 3)))

    # ---- 2. cut 敏感性（同一模型，只换评估子集）----
    print()
    print('=== cut 敏感性：固定同一模型，只在子集上评估 ===')
    print('%-34s %8s %10s %10s %10s %9s' %
          ('cut', 'n', 'keep%', 'chi2_all', 'chi2_test', 'frac|χ|>3'))
    cuts = [('C0 现状（cov>0.5, 无误差 cut）', np.ones(len(y), bool))]
    for t in [0.7, 0.9, 0.99]:
        cuts.append(('C cov_min >= %.2f' % t, cov_min >= t))
    for t in [np.median(ye), np.percentile(ye, 25)]:
        cuts.append(('C qpah_err <= %.2f' % t, ye <= t))
    for t in [2, 5, 10]:
        cuts.append(('C 边缘距 >= %d px' % t, edge_min >= t))
    cuts.append(('C cov>=0.9 & qpah_err<=p50',
                 (cov_min >= 0.9) & (ye <= np.median(ye))))
    cuts.append(('C 以上全部 + 边缘距>=5px',
                 (cov_min >= 0.9) & (ye <= np.median(ye)) & (edge_min >= 5)))
    res = []
    for nm, m in cuts:
        if m.sum() < 100:
            continue
        mt = m & in_tr
        mv = m & (~in_tr)
        c_all = np.mean(chi[m] ** 2)
        c_te = np.mean(chi[mv] ** 2) if mv.sum() > 50 else np.nan
        f3 = 100 * np.mean(achi[m] > 3)
        res.append((nm, int(m.sum()), np.mean(chi[m] ** 2), c_te, f3, m))
        print('%-34s %8d %9.1f%% %10.4f %10.4f %8.1f%%'
              % (nm, m.sum(), 100 * m.mean(), c_all, c_te, f3))
    if region is not None:
        print()
        print('悬臂（#35-51, n=%d）在不同 cut 下：' % region.sum())
        for nm, m in cuts:
            mm = m & region
            if mm.sum() >= 20:
                print('  %-34s 保留 %5d  中位 pull=%+.3f  χ²=%.3f'
                      % (nm, mm.sum(), np.median(chi[mm]), np.mean(chi[mm] ** 2)))

    # ---------------- 图 ----------------
    os.makedirs(args.outdir, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(15.5, 12))

    bc, bm = [], []
    for b in range(10):
        m = bi == b
        if m.sum() >= 30:
            bc.append(np.median(ye[m])); bm.append(np.median(achi[m]))
    ax[0, 0].plot(bc, bm, 'o-', color='crimson', lw=2)
    ax[0, 0].axhline(1, color='k', ls='--', lw=1)
    ax[0, 0].set_xlabel(r'$q_{\rm PAH}$ err (%)'); ax[0, 0].set_ylabel(r'median $|\chi|$')
    ax[0, 0].set_title('bad pixels vs observational error'); ax[0, 0].grid(alpha=.3)

    cov_ts = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    c2s = [np.mean(chi[cov_min >= t] ** 2) for t in cov_ts]
    ax[0, 1].plot(cov_ts, c2s, 'o-', color='steelblue', lw=2)
    ax[0, 1].axhline(1, color='k', ls='--', lw=1)
    ax[0, 1].set_xlabel('coverage threshold'); ax[0, 1].set_ylabel(r'$\chi^2_{\rm red}$')
    ax[0, 1].set_title('chi2 vs coverage cut'); ax[0, 1].grid(alpha=.3)

    order = np.argsort(ye)
    s1 = ax[1, 0].scatter(ra[order], dec[order], s=1.0, c=ye[order], cmap='viridis',
                          vmin=np.percentile(ye, 2), vmax=np.percentile(ye, 98),
                          rasterized=True, linewidths=0)
    ax[1, 0].set_title(r'$q_{\rm PAH}$ err map'); ax[1, 0].set_xlabel('RA (deg)')
    ax[1, 0].set_ylabel('Dec (deg)'); ax[1, 0].invert_xaxis(); ax[1, 0].set_aspect('equal')
    fig.colorbar(s1, ax=ax[1, 0], fraction=0.046, pad=0.02, label=r'$q_{\rm PAH}$ err (%)')

    big = achi > 2
    s2 = ax[1, 1].scatter(ra[big], dec[big], s=2.0, c=ye[big], cmap='viridis',
                          vmin=np.percentile(ye, 2), vmax=np.percentile(ye, 98),
                          rasterized=True, linewidths=0)
    ax[1, 1].set_title(r'$|\chi|>2$ pixels, coloured by their $q_{\rm PAH}$ err')
    ax[1, 1].set_xlabel('RA (deg)'); ax[1, 1].invert_xaxis(); ax[1, 1].set_aspect('equal')
    fig.colorbar(s2, ax=ax[1, 1], fraction=0.046, pad=0.02, label=r'$q_{\rm PAH}$ err (%)')
    fig.tight_layout()
    f = os.path.join(args.outdir, 'fig_cut_sensitivity.png')
    fig.savefig(f, dpi=150, bbox_inches='tight'); plt.close(fig)
    print('\nsaved', f)


if __name__ == '__main__':
    main()
