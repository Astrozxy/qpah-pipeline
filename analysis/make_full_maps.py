#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_full_maps.py — 4 面板全样本图：观测 / 预测 / 相对误差 / 绝对误差。

模型：log10 线性 + 可选的【空间项】。
  - 观测量特征：x → max(x,0) → log10(x + floor)，floor 取训练集正值的 1% 中位
  - 空间项（若给出 --spatial）：直接线性进入设计矩阵（**不做 log**）
      x    : 去投影长轴坐标 (kpc, 有符号) —— 东北/西南
      y    : 去投影短轴坐标 (kpc, 有符号) —— 近/远侧
      R    : 去投影半径 (kpc)
      logR : log10(R + 0.1)
  两类都做 z-score 标准化（统计量取自训练扇区）。
  拟合：误差加权最小二乘，权重 = 1/σ_eff²，σ_eff=sqrt(qpah_err²+err_floor²)。

与前作一致：**不标测试扇区、不做扇区筛选**，覆盖数据集全部像素；
模型指标只在角标给出。

用法:
  python make_full_maps.py --h5 data/M31_1d_array_nodust.h5 \
      --features H1 sfr CO H1_ew --spatial x \
      --out results/fullmaps/fig_4panel_base_x.png
"""
import os, argparse
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mplconfig')
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

DEFAULT_FEATS = ['H1', 'sfr', 'CO', 'H1_ew']
RA0, DEC0, INC, PA = 10.6847, 41.2690, np.radians(77.7), np.radians(37.7)
ARCSEC_PER_KPC = 206265.0 / 744.0


def spatial_terms(ra, dec, which):
    """去投影空间项（与 spatial_terms.py / arm_* 脚本同一套几何）。"""
    dx = (ra - RA0) * np.cos(np.radians(DEC0)); dy = dec - DEC0
    x = (dx * np.cos(PA) + dy * np.sin(PA)) * 3600.0 / ARCSEC_PER_KPC
    y = ((-dx * np.sin(PA) + dy * np.cos(PA)) / np.cos(INC)) * 3600.0 / ARCSEC_PER_KPC
    R = np.sqrt(x ** 2 + y ** 2)
    pool = {'x': x, 'y': y, 'R': R, 'logR': np.log10(R + 0.1)}
    return [pool[w] for w in which]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', default='data/M31_1d_array_nodust.h5')
    ap.add_argument('--features', nargs='+', default=DEFAULT_FEATS)
    ap.add_argument('--spatial', nargs='*', default=[], choices=['x', 'y', 'R', 'logR'],
                    help='要加入的空间项（可多个，例如 --spatial x 或 --spatial R y x）')
    ap.add_argument('--err-floor', type=float, default=0.2)
    ap.add_argument('--out', default='results/fullmaps/fig_full_4panel.png')
    ap.add_argument('--qmax', type=float, default=6.0, help='qPAH 色标上限 (%)')
    ap.add_argument('--chi-vmax', type=float, default=3.0, help='相对误差色标范围 ±(sigma)')
    ap.add_argument('--abs-vmax', type=float, default=2.0, help='绝对误差色标范围 ±(%)')
    args = ap.parse_args()
    feats, spa = args.features, list(args.spatial)

    with h5py.File(args.h5, 'r') as f:
        keys = ['ra', 'dec', 'qpah', 'qpah_err', 'CO_sigma'] + feats
        d = {k: f[k][:] for k in dict.fromkeys(keys) if k in f}
    sel = (d['qpah'] > 0) & (d['qpah_err'] > 0)
    for k in feats:
        sel &= np.isfinite(d[k])
    if 'dust_density' in feats:
        sel &= d['dust_density'] > 0
    if 'H1' in feats:
        sel &= d['H1'] >= 0
    if 'H1_ew' in feats:
        sel &= d['H1_ew'] >= 0
    if 'CO' in feats:
        sel &= np.isfinite(d['CO_sigma']) & (d['CO_sigma'] > 0)
    ra, dec = d['ra'][sel].astype(float), d['dec'][sel].astype(float)
    X = np.maximum(np.column_stack([d[k][sel] for k in feats]).astype(float), 0.0)
    y = d['qpah'][sel].astype(float)
    ye = d['qpah_err'][sel].astype(float)
    sig = np.sqrt(ye ** 2 + args.err_floor ** 2)

    np.random.seed(0)
    ang = (np.arctan2(dec - 41.25, ra - 10.75) + 2 * np.pi) % (2 * np.pi)
    sec = np.digitize(ang, np.linspace(0, 2 * np.pi, 11)) - 1
    tr_s = np.random.choice(np.arange(10), size=6, replace=False)
    in_tr = np.isin(sec, tr_s)

    floors = np.array([max(1e-9, 0.01 * np.median(X[in_tr, j][X[in_tr, j] > 0]))
                       for j in range(X.shape[1])])
    parts = [np.log10(X + floors[None, :])]
    if spa:
        parts += [np.asarray(s)[:, None] for s in spatial_terms(ra, dec, spa)]
    Z = np.column_stack(parts)
    names = list(feats) + spa

    mu, sd = Z[in_tr].mean(0), Z[in_tr].std(0) + 1e-12
    Dall = np.column_stack([np.ones(len(Z)), (Z - mu) / sd])
    w = 1.0 / sig[in_tr] ** 2
    coef = np.linalg.solve(Dall[in_tr].T @ (w[:, None] * Dall[in_tr]),
                           Dall[in_tr].T @ (w * y[in_tr]))
    pred = Dall @ coef
    resid = pred - y
    chi = resid / sig

    def met(m):
        p = chi[m]
        return float(np.mean(p ** 2)), float(np.median(p)), float(np.std(p))

    c2_tr, _, _ = met(in_tr)
    c2_te, pm_te, ps_te = met(~in_tr)
    c2_all, pm_all, ps_all = met(np.ones(len(y), bool))
    print('n=%d (train %d / test %d) | 观测量 %s | 空间项 %s'
          % (len(y), in_tr.sum(), (~in_tr).sum(), ' '.join(feats),
             ' '.join(spa) if spa else '(无)'))
    print('chi2  train=%.4f  test=%.4f  all=%.4f' % (c2_tr, c2_te, c2_all))
    print('pull(all) 中位=%+.4f  std=%.4f  [p16=%+.3f p84=%+.3f]'
          % (pm_all, ps_all, np.percentile(chi, 16), np.percentile(chi, 84)))
    print('绝对残差 med=%+.3f p16=%+.3f p84=%+.3f rms=%.3f'
          % (np.median(resid), np.percentile(resid, 16), np.percentile(resid, 84),
             np.sqrt(np.mean(resid ** 2))))
    print('标准化系数: 截距=%+.4f | ' % coef[0] +
          '  '.join('%s=%+.4f' % (n, c) for n, c in zip(names, coef[1:])))

    # ---------------- 4 面板 ----------------
    order = np.argsort(np.abs(chi))[::-1]        # 大残差画在上面
    nrm_q = Normalize(0, args.qmax)
    nrm_c = Normalize(-args.chi_vmax, args.chi_vmax)     # 固定 ±3 sigma
    nrm_a = Normalize(-args.abs_vmax, args.abs_vmax)     # 固定 ±2 %
    print('色标: qPAH 0–%.1f%% | 相对误差 ±%.1fσ | 绝对误差 ±%.1f%%（固定范围）'
          % (args.qmax, args.chi_vmax, args.abs_vmax))
    print('超出色标范围（会被裁到端点色）: |χ|>%.1fσ 占 %.2f%% ； |resid|>%.1f%% 占 %.2f%%'
          % (args.chi_vmax, 100 * np.mean(np.abs(chi) > args.chi_vmax),
             args.abs_vmax, 100 * np.mean(np.abs(resid) > args.abs_vmax)))

    fig, ax = plt.subplots(1, 4, figsize=(26, 6.6))
    panels = [
        (r'Observation  $q_{\rm PAH}$', y, 'inferno', nrm_q, r'$q_{\rm PAH}$ (%)'),
        (r'Prediction  $q_{\rm PAH}$', pred, 'inferno', nrm_q, r'$q_{\rm PAH}$ (%)'),
        (r'Relative error  $\chi=(\mathrm{pred}-\mathrm{obs})/\sigma_{\rm eff}$',
         chi, 'coolwarm', nrm_c, r'$\chi$   ($\sigma_{\rm eff}$)'),
        (r'Absolute error  $\mathrm{pred}-\mathrm{obs}$',
         resid, 'coolwarm', nrm_a, r'$\mathrm{pred}-\mathrm{obs}$   (%)'),
    ]
    for a, (title, val, cmap, nrm, clab) in zip(ax, panels):
        sc = a.scatter(ra[order], dec[order], s=1.0, c=val[order], cmap=cmap,
                       norm=nrm, rasterized=True, linewidths=0)
        a.set_title(title, fontsize=15)
        a.set_xlabel('RA (J2000)')
        a.set_aspect('equal'); a.invert_xaxis(); a.grid(True, ls=':', alpha=.35)
        a.tick_params(direction='in')
        cb = fig.colorbar(sc, ax=a, fraction=0.046, pad=0.02)
        cb.set_label(clab, fontsize=12); cb.ax.tick_params(direction='in')
    ax[0].set_ylabel('Dec (J2000)')

    txt = ('n = %d  (train %d / test %d)\n'
           'test: $\\chi^2_{\\rm red}$=%.3f, pull %+.3f$\\pm$%.3f\n'
           'all : $\\chi^2_{\\rm red}$=%.3f, pull %+.3f$\\pm$%.3f')
    fig.text(0.5, -0.02, txt % (len(y), int(in_tr.sum()), int((~in_tr).sum()),
                                c2_te, pm_te, ps_te, c2_all, pm_all, ps_all),
             ha='center', fontsize=12)
    fig.suptitle('log10-linear: %d observables (%s)%s  —  all pixels, no sector selection'
                 % (len(feats), ', '.join(feats),
                    ('  +  spatial: ' + ', '.join(spa)) if spa else ''), fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.savefig(args.out, dpi=170, bbox_inches='tight'); plt.close(fig)
    print('saved', args.out)


if __name__ == '__main__':
    main()
