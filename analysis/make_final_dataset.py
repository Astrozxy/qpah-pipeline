#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_final_dataset.py — 构建【最终口径】的数据集，并把 Draine 官网的列合并进来。

最终筛选链（2026-10-09 定）：
  1. 基础：qpah 有限、qpah_err 有限、4 特征有限（⇒ coverage>0.5）
  2. qpah >= 0                （保留 DL07 下界堆积的真实 0 值；填充值已被 qpah_err>0 排除）
  3. 0 < qpah_err <= 2        （目标质量 cut：±2% 以上的不确定度等于没测）
  4. H1 >= 0、H1_ew >= 0、CO_sigma 有限且 > 0
  5. R <= 18 kpc              （Draine+2014 明确说模型参数在大半径不可靠；
                                尘埃面密度到 ~18 kpc 才逐像素可信）

同时合并 Draine 官网 S350 分辨率图的每行值（同网格，无需 reproject）：
  Ldust（尘埃 SED 全红外积分 ≈ 直接测量）、Mdust、U_min、U_bar

用法:
  python make_final_dataset.py --h5 data/M31_1d_array_nodust.h5 \
      --draine cache/draine_rows.npz --rmax 18 --outdir results/final
"""
import os, argparse
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mplconfig')
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FEATS = ['H1', 'sfr', 'CO', 'H1_ew']
RA0, DEC0, INC, PA = 10.6847, 41.2690, np.radians(77.7), np.radians(37.7)
DRAINE = ['Ldust', 'Mdust', 'U_min', 'U_bar']


def deproj(ra, dec):
    dx = (ra - RA0) * np.cos(np.radians(DEC0)); dy = dec - DEC0
    x = dx * np.sin(PA) + dy * np.cos(PA)
    y = (-dx * np.cos(PA) + dy * np.sin(PA)) / np.cos(INC)
    k = 3600.0 / (206265.0 / 744.0)
    return np.sqrt(x ** 2 + y ** 2) * k, x * k, y * k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', required=True)
    ap.add_argument('--draine', default=None, help='extract_draine_maps.py 的输出（与 h5 行对齐）')
    ap.add_argument('--qpah-err-max', type=float, default=2.0)
    ap.add_argument('--rmax', type=float, default=18.0)
    ap.add_argument('--edge-npz', default=None,
                    help='export_edge_dist.py 的输出（与输入 h5 行对齐），提供各图 footprint 边缘距')
    ap.add_argument('--qpah-edge-min', type=float, default=5.0,
                    help='要求像素在 qPAH 图【有效区内】且到边界至少该像素数。'
                         '边界 5px 内残差显著偏大且有系统性高估；qpah==0 的"下界堆积"'
                         '其实是有效区【外】1-6px 的一圈薄边，属拟合失败产物')
    ap.add_argument('--co-edge-min', type=float, default=20.0,
                    help='剔除到 CO footprint 边界小于该像素距离的像素（默认 20 px = 200"）。'
                         'CO 是覆盖最贴边的图，其边界处残差显著偏大（同半径内亦然）')
    ap.add_argument('--hi-min', type=float, default=110.0,
                    help='HI mom0 下限 [K km/s]；默认 110 对应 N_HI >= 2e20 cm^-2'
                         '（Sigma_HI >= 1.60 Msun/pc^2）—— 其下视为事实上没有 ISM')
    ap.add_argument('--outdir', default='results/final')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    with h5py.File(args.h5) as f:
        cols = list(f.keys())
        d = {k: f[k][:] for k in cols}
        attrs = dict(f.attrs)
    n0 = len(d['qpah'])
    R, _, _ = deproj(d['ra'].astype(float), d['dec'].astype(float))

    steps = []
    m = np.ones(n0, bool);                        steps.append(('h5 全部行', m.copy()))
    m &= np.isfinite(d['qpah']);                  steps.append(('qpah 有限', m.copy()))
    m &= np.isfinite(d['qpah_err']);              steps.append(('qpah_err 有限', m.copy()))
    for k in FEATS:
        m &= np.isfinite(d[k])
    steps.append(('4 特征有限（coverage>0.5）', m.copy()))
    m &= (d['qpah'] >= 0);                        steps.append(('qpah >= 0', m.copy()))
    m &= (d['qpah_err'] > 0);                     steps.append(('qpah_err > 0', m.copy()))
    m &= (d['H1'] >= 0);                          steps.append(('H1 >= 0', m.copy()))
    m &= (d['H1_ew'] >= 0);                       steps.append(('H1_ew >= 0', m.copy()))
    m &= np.isfinite(d['CO_sigma']) & (d['CO_sigma'] > 0)
    steps.append(('CO_sigma 有限且 >0', m.copy()))
    m &= (d['qpah_err'] <= args.qpah_err_max)
    steps.append(('qpah_err <= %.1f' % args.qpah_err_max, m.copy()))
    m &= (R <= args.rmax)
    steps.append(('R <= %.0f kpc' % args.rmax, m.copy()))
    m &= (d['H1'] >= args.hi_min)
    steps.append(('H1 >= %g K km/s (N_HI>=2e20)' % args.hi_min, m.copy()))
    if args.edge_npz:
        ze = np.load(args.edge_npz)
        key = 'edge_qpah_signed' if 'edge_qpah_signed' in ze.files else 'edge_qpah'
        m &= (ze[key] >= args.qpah_edge_min)
        steps.append(('qPAH 边缘距 >= %g px' % args.qpah_edge_min, m.copy()))
        if len(ze['edge_CO']) != n0:
            raise SystemExit('edge npz 行数 %d != h5 %d' % (len(ze['edge_CO']), n0))
        m &= (ze['edge_CO'] >= args.co_edge_min)
        steps.append(('CO 边缘距 >= %g px' % args.co_edge_min, m.copy()))
    sel = m

    print('=== 最终口径筛选漏斗 ===')
    prev = n0
    for nm, mm in steps:
        print('  %-32s %8d  (%6.2f%%)  %+7d' % (nm, mm.sum(), 100 * mm.sum() / n0, mm.sum() - prev))
        prev = mm.sum()
    print('  %-32s %8d' % ('→ 最终样本', sel.sum()))
    print('  qpah==0 的 %d (%.2f%%) | R 中位=%.1f kpc' % ((d['qpah'][sel] == 0).sum(),
          100 * np.mean(d['qpah'][sel] == 0), np.median(R[sel])))

    z = None
    if args.draine:
        z = np.load(args.draine)
        if len(z['Ldust']) != n0:
            raise SystemExit('Draine npz 行数 %d != h5 %d' % (len(z['Ldust']), n0))

    out = os.path.join(args.outdir, 'M31_1d_array_final.h5')
    with h5py.File(out, 'w') as f:
        for k in cols:
            f.create_dataset(k, data=d[k][sel])
        f.create_dataset('R_kpc', data=R[sel])
        if z is not None:
            for k in DRAINE:
                if k in z.files:
                    f.create_dataset(k, data=z[k][sel])
        for k, v in attrs.items():
            try: f.attrs[k] = v
            except Exception: pass
        f.attrs['final_selection'] = ('qpah>=0 & 0<qpah_err<=%g & H1_ew>=0 & CO_sigma>0 '
                                      '& R<=%g kpc & H1>=%g K km/s (N_HI>=2e20 cm^-2)'
                                      ' & CO_edge>=%g px'
                                      % (args.qpah_err_max, args.rmax, args.hi_min,
                                         args.co_edge_min))
    print('written %s (%.1f MB)' % (out, os.path.getsize(out) / 1e6))

    # ---- 诊断图 ----
    ra, dec = d['ra'][sel].astype(float), d['dec'][sel].astype(float)
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.8))
    dropped = int((~sel & (d['qpah'] >= 0) & (d['qpah_err'] > 0) & np.isfinite(d['H1'])).sum())
    ax[0].scatter(d['ra'], d['dec'], s=0.4, c='lightgray', rasterized=True, linewidths=0)
    ax[0].scatter(ra, dec, s=0.8, c='steelblue', rasterized=True, linewidths=0)
    ax[0].set_title('final sample (blue) = %d px  |  dropped %d' % (sel.sum(), dropped))
    ax[0].set_xlabel('RA (deg)'); ax[0].set_ylabel('Dec (deg)')
    ax[0].invert_xaxis(); ax[0].set_aspect('equal')

    ax[1].hist(R[sel], bins=70, range=(0, 25), color='steelblue', label='final (R<=%g)' % args.rmax)
    ax[1].hist(R[~sel], bins=70, range=(0, 25), histtype='step', color='crimson', lw=2, label='dropped')
    ax[1].axvline(args.rmax, color='k', ls='--')
    ax[1].set_xlabel('R (kpc)'); ax[1].set_ylabel('N'); ax[1].legend(); ax[1].grid(alpha=.3)
    ax[1].set_title('radial distribution')

    ax[2].hist(d['qpah'][sel], bins=70, range=(0, 8), color='steelblue')
    ax[2].axvline(0, color='crimson', ls='--', lw=1)
    ax[2].set_xlabel(r'$q_{\rm PAH}$ (%)'); ax[2].set_ylabel('N'); ax[2].grid(alpha=.3)
    ax[2].set_title(r'$q_{\rm PAH}$ (exact zeros: %.2f%%)' % (100 * np.mean(d['qpah'][sel] == 0)))
    fig.tight_layout()
    fp = os.path.join(args.outdir, 'fig_final_selection.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight'); plt.close(fig)
    print('saved', fp)


if __name__ == '__main__':
    main()
