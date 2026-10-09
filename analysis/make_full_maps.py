#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_full_maps.py — 4 面板全样本图：观测 / 预测 / 相对误差 / 绝对误差。

与前作 plot_galaxy_map.py 的区别：
  - 4 个面板（加了单独的绝对误差面板），不是 3 个；
  - **不标测试扇区、不做任何扇区筛选** —— 覆盖数据集里的全部像素；
  - 模型的测试指标只在角标里给出（不靠视觉分区）。

模型：log10 线性（默认 4 特征，不含 dust），误差加权最小二乘，在 6 个训练扇区上拟合，
      对全部像素做预测。空间测试扇区用于报告 χ²，但不影响这张图的覆盖范围。

用法:
  python qpah_pipeline/analysis/make_full_maps.py \
      --h5 data/M31_1d_array_nodust.h5 --features H1 sfr CO H1_ew \
      --out results/fullmaps/fig_full_4panel_nodust.png
"""
import os, argparse
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mplconfig')
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

DEFAULT_FEATS = ['H1', 'sfr', 'CO', 'H1_ew']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', default='data/M31_1d_array_nodust.h5')
    ap.add_argument('--features', nargs='+', default=DEFAULT_FEATS)
    ap.add_argument('--err-floor', type=float, default=0.2)
    ap.add_argument('--out', default='results/fullmaps/fig_full_4panel.png')
    ap.add_argument('--qmax', type=float, default=6.0, help='qPAH 色标上限 (%)')
    ap.add_argument('--chi-vmax', type=float, default=3.0)
    args = ap.parse_args()

    with h5py.File(args.h5, 'r') as f:
        keys = ['ra', 'dec', 'qpah', 'qpah_err', 'CO_sigma'] + args.features
        d = {k: f[k][:] for k in keys if k in f}
    sel = (d['qpah'] > 0) & (d['qpah_err'] > 0)
    for k in args.features:
        sel &= np.isfinite(d[k])
    if 'dust_density' in args.features:
        sel &= d['dust_density'] > 0          # 0 = 平滑时的零填充/缺数据（与 load_h5 一致）
    if 'H1' in args.features:
        sel &= d['H1'] >= 0
    if 'H1_ew' in args.features:
        sel &= d['H1_ew'] >= 0
    if 'CO' in args.features:
        sel &= np.isfinite(d['CO_sigma']) & (d['CO_sigma'] > 0)
    ra, dec = d['ra'][sel].astype(float), d['dec'][sel].astype(float)
    X = np.maximum(np.column_stack([d[k][sel] for k in args.features]).astype(float), 0.0)
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
    Z = np.log10(X + floors[None, :])
    mu, sd = Z[in_tr].mean(0), Z[in_tr].std(0) + 1e-12
    Zall = np.column_stack([np.ones(len(Z)), (Z - mu) / sd])
    w = 1.0 / sig[in_tr] ** 2
    coef = np.linalg.solve(Zall[in_tr].T @ (w[:, None] * Zall[in_tr]),
                           Zall[in_tr].T @ (w * y[in_tr]))
    pred = Zall @ coef
    resid = pred - y
    chi = resid / sig

    def met(m):
        p = chi[m]
        return float(np.mean(p ** 2)), float(np.median(p)), float(np.std(p))

    c2_te, pm_te, ps_te = met(~in_tr)
    c2_all, pm_all, ps_all = met(np.ones(len(y), bool))
    print('n=%d (train %d / test %d) | 特征 %s' % (len(y), in_tr.sum(), (~in_tr).sum(),
                                                  ' '.join(args.features)))
    print('test : chi2=%.4f pull med=%+.4f std=%.4f' % (c2_te, pm_te, ps_te))
    print('all  : chi2=%.4f pull med=%+.4f std=%.4f' % (c2_all, pm_all, ps_all))
    print('abs resid: med=%+.3f p16=%+.3f p84=%+.3f rms=%.3f'
          % (np.median(resid), np.percentile(resid, 16), np.percentile(resid, 84),
             np.sqrt(np.mean(resid ** 2))))

    # ==================== 4 面板 ====================
    n_obs = y
    order = np.argsort(np.abs(chi))[::-1]        # 大残差画在上面
    nrm_q = Normalize(0, args.qmax)
    nrm_c = Normalize(-args.chi_vmax, args.chi_vmax)
    vmax_a = float(np.percentile(np.abs(resid), 99))
    nrm_a = Normalize(-vmax_a, vmax_a)

    fig, ax = plt.subplots(1, 4, figsize=(26, 6.6))
    panels = [
        ('Observation  $q_{\\rm PAH}$', n_obs, 'inferno', nrm_q, r'$q_{\rm PAH}$ (%)', None),
        ('Prediction  $q_{\\rm PAH}$', pred, 'inferno', nrm_q, r'$q_{\rm PAH}$ (%)', None),
        (r'Relative error  $\chi=(\mathrm{pred}-\mathrm{obs})/\sigma_{\rm eff}$',
         chi, 'coolwarm', nrm_c, r'$\chi$   ($\sigma_{\rm eff}$)', None),
        (r'Absolute error  $\mathrm{pred}-\mathrm{obs}$',
         resid, 'coolwarm', nrm_a, r'$\mathrm{pred}-\mathrm{obs}$   (%)', None),
    ]
    for a, (title, val, cmap, nrm, clab, _) in zip(ax, panels):
        sc = a.scatter(ra[order], dec[order], s=1.0, c=val[order], cmap=cmap,
                       norm=nrm, rasterized=True, linewidths=0)
        a.set_title(title, fontsize=15)
        a.set_xlabel('RA (J2000)')
        a.set_aspect('equal')
        a.invert_xaxis()
        a.grid(True, ls=':', alpha=.35)
        a.tick_params(direction='in')
        cb = fig.colorbar(sc, ax=a, fraction=0.046, pad=0.02)
        cb.set_label(clab, fontsize=12)
        cb.ax.tick_params(direction='in')
    ax[0].set_ylabel('Dec (J2000)')

    txt = ('n = %d  (train sectors %d / test sectors %d)\n'
           'test: $\\chi^2_{\\rm red}$=%.3f, pull %+.3f$\\pm$%.3f\n'
           'all : $\\chi^2_{\\rm red}$=%.3f, pull %+.3f$\\pm$%.3f')
    fig.text(0.5, -0.02, txt % (len(y), int(in_tr.sum()), int((~in_tr).sum()),
                                c2_te, pm_te, ps_te, c2_all, pm_all, ps_all),
             ha='center', fontsize=12)
    fig.suptitle('log10-linear model (%d features: %s)  —  all pixels, no sector selection'
                 % (len(args.features), ', '.join(args.features)), fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.savefig(args.out, dpi=170, bbox_inches='tight')
    plt.close(fig)
    print('saved', args.out)


if __name__ == '__main__':
    main()
