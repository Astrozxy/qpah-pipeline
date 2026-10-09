#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
show_selection.py — 打印当前【完整筛选链】与【训练/测试集划分】，并画一张图。

口径（2026-10-09 定）：
  1. align_maps.py  把 5 张图对齐到 qPAH 的 10″ 网格（NaN 感知的面积加权块平均），
                    coverage <= 0.5 的格置 NaN   ← 这是"隐式"的覆盖 cut
  2. build_dataset.py  要求 qpah 有限、不在 M32（2.5′ 圆）、必需特征有限
  3. 分析脚本的选择：
       qpah > 0  &  qpah_err > 0
       各特征有限（⇒ 各自 coverage > 0.5）
       H1 >= 0  &  H1_ew >= 0
       CO_sigma 有限且 > 0
  4. 【新】qpah_err <= 2.0        ← 目标质量 cut（用户拍板）
  5. 训练/测试划分：绕中心 (10.75, 41.25) 按方位角切 10 个扇区，
     np.random.seed(0) 抽 6 个作训练，其余 4 个作测试。
     ★ 测试集【没有】单独的筛选 —— 它就是同类样本里训练扇区的补集，cut 完全一致。
     ★ 没有 val（已移除）。
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', required=True)
    ap.add_argument('--qpah-err-max', type=float, default=None,
                    help='显示"若再加此 cut"的效果；默认按 h5 自身内容为准')
    ap.add_argument('--outdir', default='results/selection')
    args = ap.parse_args()

    with h5py.File(args.h5, 'r') as f:
        d = {k: f[k][:] for k in f.keys()}
        attrs = dict(f.attrs)
    n0 = len(d['qpah'])
    print('数据集: %s' % args.h5)
    if attrs:
        print('数据集自带属性: %s' % attrs)
    print('h5 行数: %d' % n0)
    print()

    # ---- 筛选漏斗 ----
    cuts = []
    m = np.ones(n0, bool)
    cuts.append(('h5 全部行', m.copy()))
    m &= np.isfinite(d['qpah']);                 cuts.append(('qpah 有限', m.copy()))
    m &= np.isfinite(d['qpah_err']);             cuts.append(('qpah_err 有限', m.copy()))
    for k in FEATS:
        m &= np.isfinite(d[k])
    cuts.append(('4 特征全部有限（⇒ coverage>0.5）', m.copy()))
    m &= (d['qpah'] >= 0);                        cuts.append(('qpah > 0', m.copy()))
    m &= (d['qpah_err'] > 0);                    cuts.append(('qpah_err > 0', m.copy()))
    m &= (d['H1'] >= 0);                         cuts.append(('H1 >= 0', m.copy()))
    m &= (d['H1_ew'] >= 0);                      cuts.append(('H1_ew >= 0', m.copy()))
    m &= np.isfinite(d['CO_sigma']) & (d['CO_sigma'] > 0)
    cuts.append(('CO_sigma 有限且 > 0', m.copy()))
    if args.qpah_err_max is not None:
        m &= (d['qpah_err'] <= args.qpah_err_max)
        cuts.append(('qpah_err <= %.1f' % args.qpah_err_max, m.copy()))
    sel = m
    print('=== 筛选漏斗 ===')
    prev = n0
    for nm, mm in cuts:
        print('  %-38s %8d  (%6.2f%%)  %+7d' % (nm, mm.sum(), 100 * mm.sum() / n0, mm.sum() - prev))
        prev = mm.sum()
    print('  %-38s %8d' % ('→ 最终分析样本', sel.sum()))
    print()

    # ---- 划分 ----
    ra, dec = d['ra'][sel].astype(float), d['dec'][sel].astype(float)
    np.random.seed(0)
    ang = (np.arctan2(dec - 41.25, ra - 10.75) + 2 * np.pi) % (2 * np.pi)
    sec = np.digitize(ang, np.linspace(0, 2 * np.pi, 11)) - 1
    tr_s = np.random.choice(np.arange(10), size=6, replace=False)
    te_s = np.setdiff1d(np.arange(10), tr_s)
    in_tr = np.isin(sec, tr_s)
    print('=== 划分（绕 (10.75, 41.25) 按方位角切 10 个 36° 扇区）===')
    print('  种子 np.random.seed(0) → 训练扇区 %s | 测试扇区 %s' % (tr_s.tolist(), te_s.tolist()))
    print('  ★ 测试集无单独筛选：= 同一样本里训练扇区的补集，cut 完全一致')
    print('  训练 %d (%.1f%%) | 测试 %d (%.1f%%) | val: 无（已移除）'
          % (in_tr.sum(), 100 * in_tr.mean(), (~in_tr).sum(), 100 * (~in_tr).mean()))
    print()
    print('  %-8s %10s %10s %10s' % ('扇区', '全部', '训练', '测试'))
    for s in range(10):
        ms = sec == s
        role = '训练' if s in tr_s else '测试'
        print('  #%-7d %10d %10d %10d   [%s]'
              % (s, ms.sum(), (ms & in_tr).sum(), (ms & ~in_tr).sum(), role))
    print()

    # ---- 径向构成 ----
    dx = (ra - RA0) * np.cos(np.radians(DEC0)); dy = dec - DEC0
    X = dx * np.sin(PA) + dy * np.cos(PA)
    Y = (-dx * np.cos(PA) + dy * np.sin(PA)) / np.cos(INC)
    R = np.sqrt(X ** 2 + Y ** 2) * 3600.0 / (206265.0 / 744.0)
    print('=== 径向构成（检查训练/测试是否均衡）===')
    print('  %-10s %10s %10s %8s | %10s %10s' %
          ('R (kpc)', '训练 n', '测试 n', '训练占比', '训练 χ²', '测试 χ²'))
    ye = d['qpah_err'][sel]
    sig = np.sqrt(ye ** 2 + 0.04)
    # 用一个基础模型算 chi 供参考
    Xm = np.maximum(np.column_stack([d[k][sel] for k in FEATS]).astype(float), 0.0)
    y = d['qpah'][sel].astype(float)
    fl = np.array([max(1e-9, 0.01 * np.median(Xm[in_tr, j][Xm[in_tr, j] > 0])) for j in range(4)])
    Z = np.log10(Xm + fl[None, :])
    mu, sd = Z[in_tr].mean(0), Z[in_tr].std(0) + 1e-12
    D = np.column_stack([np.ones(len(Z)), (Z - mu) / sd])
    w = 1.0 / sig[in_tr] ** 2
    c = np.linalg.solve(D[in_tr].T @ (w[:, None] * D[in_tr]), D[in_tr].T @ (w * y[in_tr]))
    chi = (D @ c - y) / sig
    edges = [0, 3, 6, 9, 12, 15, 18, 21, 30]
    for a, b in zip(edges[:-1], edges[1:]):
        mr = (R >= a) & (R < b) & in_tr
        mv = (R >= a) & (R < b) & (~in_tr)
        if mr.sum() + mv.sum() < 30:
            continue
        print('  %-10s %10d %10d %7.1f%% | %10s %10s'
              % ('%d-%d' % (a, b), mr.sum(), mv.sum(),
                 100 * mr.sum() / max(1, mr.sum() + mv.sum()),
                 '%.3f' % np.mean(chi[mr] ** 2) if mr.sum() > 20 else '-',
                 '%.3f' % np.mean(chi[mv] ** 2) if mv.sum() > 20 else '-'))

    # ---- 图 ----
    os.makedirs(args.outdir, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(19, 6))
    sc = ax[0].scatter(ra, dec, s=0.8, c=sec, cmap='tab10', vmin=0, vmax=9,
                       rasterized=True, linewidths=0)
    ax[0].set_title('azimuthal sectors (10 x 36 deg about 10.75, 41.25)')
    fig.colorbar(sc, ax=ax[0], fraction=0.046, pad=0.02, label='sector id')
    ax[1].scatter(ra[~in_tr], dec[~in_tr], s=0.8, c='crimson', rasterized=True,
                  linewidths=0, label='test (%d)' % (~in_tr).sum())
    ax[1].scatter(ra[in_tr], dec[in_tr], s=0.8, c='steelblue', rasterized=True,
                  linewidths=0, label='train (%d)' % in_tr.sum())
    ax[1].legend(markerscale=8, fontsize=9)
    ax[1].set_title('train / test split (no separate test filtering)')
    sc2 = ax[2].scatter(ra, dec, s=0.8, c=ye, cmap='viridis',
                        vmin=np.percentile(ye, 2), vmax=np.percentile(ye, 98),
                        rasterized=True, linewidths=0)
    ax[2].set_title(r'$q_{\rm PAH}$ err (the quality variable)')
    fig.colorbar(sc2, ax=ax[2], fraction=0.046, pad=0.02, label=r'$q_{\rm PAH}$ err (%)')
    for a in ax:
        a.set_xlabel('RA (deg)'); a.set_ylabel('Dec (deg)')
        a.invert_xaxis(); a.set_aspect('equal')
    fig.tight_layout()
    f_ = os.path.join(args.outdir, 'fig_selection.png')
    fig.savefig(f_, dpi=150, bbox_inches='tight'); plt.close(fig)
    print('\nsaved', f_)


if __name__ == '__main__':
    main()
