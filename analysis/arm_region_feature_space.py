#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
arm_region_feature_space.py — 圈出"误差异常的那条悬臂"，并在【参数空间】里检验它是否突出。

重要口径
--------
空间坐标（RA/Dec）**只用于圈定区域和画图**，绝不作为模型特征 ——
否则 reviewer 会指责我们把 map 位置信息偷偷塞进了模型。
模型始终只用观测量特征（默认 H1, sfr, CO, H1_ew）。

步骤
----
1. 拟合 log10 线性模型（训练扇区），算每像素 χ=(pred-obs)/σ_eff；
2. 把天区切成粗格子，算每格的【中位 |χ|】→ 得到"误差的空间图"；
3. 取超过阈值的格子，做连通域标记，挑最大的一块 = 那条悬臂；
4. 图 1：中位|χ|格图 / 选中区域叠加 / 逐像素 χ 图（画出区域轮廓）；
5. 图 2：参数空间检验 —— 各特征的 z 分布（区域 vs 其余）、
        k-NN 密度（以训练集为参照）、以及 |χ| 随 k-NN 距离的变化。

输出:
  fig_arm_region_map.png      空间：圈了哪块
  fig_arm_feature_space.png   参数空间：那块是否突出
"""
import os, json, argparse
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mplconfig')
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

DEFAULT_FEATS = ['H1', 'sfr', 'CO', 'H1_ew']


# ---------------------------------------------------------------- 连通域（自己写，免依赖）
def label_components(mask):
    """4-连通标记。返回 (labels, ncomp)。"""
    lab = np.zeros(mask.shape, dtype=int)
    n = 0
    ny, nx = mask.shape
    for i in range(ny):
        for j in range(nx):
            if mask[i, j] and lab[i, j] == 0:
                n += 1
                stack = [(i, j)]
                lab[i, j] = n
                while stack:
                    a, b = stack.pop()
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        p, q = a + da, b + db
                        if 0 <= p < ny and 0 <= q < nx and mask[p, q] and lab[p, q] == 0:
                            lab[p, q] = n
                            stack.append((p, q))
    return lab, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', default='data/M31_1d_array_nodust.h5')
    ap.add_argument('--features', nargs='+', default=DEFAULT_FEATS)
    ap.add_argument('--err-floor', type=float, default=0.2)
    ap.add_argument('--cell', type=float, default=0.02, help='粗格子边长（deg）')
    ap.add_argument('--min-per-cell', type=int, default=20)
    ap.add_argument('--pct', type=float, default=90.0,
                    help='格子中位|χ| 的阈值分位（越高越严格）')
    ap.add_argument('--knn', type=int, default=15)
    ap.add_argument('--mask-npz', default=None,
                    help='若给出（含 ra/dec/region，长度需与样本一致），直接用其中的 region，'
                         '跳过自动检测 —— 配合 select_blob.py 指定某一条悬臂')
    ap.add_argument('--outdir', default='results/armregion')
    args = ap.parse_args()

    # ---------------- 1. 数据与模型 ----------------
    with h5py.File(args.h5, 'r') as f:
        keys = ['ra', 'dec', 'qpah', 'qpah_err', 'CO_sigma'] + args.features
        d = {k: f[k][:] for k in keys if k in f}
    sel = (d['qpah'] > 0) & (d['qpah_err'] > 0)
    for k in args.features:
        sel &= np.isfinite(d[k])
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
    Zs = (Z - mu) / sd                                  # 标准化 = "参数空间"坐标
    Dall = np.column_stack([np.ones(len(Zs)), Zs])
    w = 1.0 / sig[in_tr] ** 2
    coef = np.linalg.solve(Dall[in_tr].T @ (w[:, None] * Dall[in_tr]),
                           Dall[in_tr].T @ (w * y[in_tr]))
    chi = (Dall @ coef - y) / sig
    print('n=%d | 特征 %s | 全体 chi2=%.4f | 测试扇区 chi2=%.4f'
          % (len(y), ' '.join(args.features), np.mean(chi ** 2), np.mean(chi[~in_tr] ** 2)))

    # ---------------- 2. 粗格子的中位 |chi| ----------------
    ra_lo, ra_hi = ra.min(), ra.max()
    de_lo, de_hi = dec.min(), dec.max()
    nx = int(np.ceil((ra_hi - ra_lo) / args.cell))
    ny = int(np.ceil((de_hi - de_lo) / args.cell))
    ix = np.clip(((ra - ra_lo) / args.cell).astype(int), 0, nx - 1)
    iy = np.clip(((dec - de_lo) / args.cell).astype(int), 0, ny - 1)
    cell_id = iy * nx + ix
    ncell = nx * ny
    med = np.full(ncell, np.nan)
    achi = np.abs(chi)
    order = np.argsort(cell_id, kind='stable')
    sid, sach = cell_id[order], achi[order]
    bounds = np.searchsorted(sid, np.arange(ncell + 1))
    for c in range(ncell):
        a, b = bounds[c], bounds[c + 1]
        if b - a >= args.min_per_cell:
            med[c] = np.median(sach[a:b])
    cnt = np.bincount(cell_id, minlength=ncell)
    ok_flat = cnt >= args.min_per_cell
    med = med.reshape(ny, nx)
    ok = ok_flat.reshape(ny, nx)
    print('格子 %dx%d，合格格（n>=%d）%d 个' % (nx, ny, args.min_per_cell, ok.sum()))

    thr = np.nanpercentile(med[ok], args.pct)
    hot = ok & (med > thr)
    region_cell = np.zeros_like(hot)
    if args.mask_npz:
        zm = np.load(args.mask_npz)
        if len(zm['region']) != len(ra):
            raise SystemExit('mask 长度 %d != 样本 %d（可能筛选口径不一致）'
                             % (len(zm['region']), len(ra)))
        region = zm['region'].astype(bool)
        region_cell[iy[region], ix[region]] = True
        print('用外部 mask: %d 像素（%.2f%%）| 该区域 chi2=%.3f，其余 chi2=%.3f'
              % (region.sum(), 100 * region.mean(),
                 np.mean(chi[region] ** 2), np.mean(chi[~region] ** 2)))
    else:
        print('阈值 = 中位|chi| 的 p%.0f = %.3f，热点格 %d 个' % (args.pct, thr, hot.sum()))
        lab, ncomp = label_components(hot)
        if ncomp == 0:
            raise SystemExit('没有找到超过阈值的连通区域，请调低 --pct')
        sizes = [(k, int((lab == k).sum())) for k in range(1, ncomp + 1)]
        sizes.sort(key=lambda t: -t[1])
        print('连通域 %d 个，最大的前 5 个（格数）: %s' % (ncomp, sizes[:5]))
        best = sizes[0][0]
        region_cell = (lab == best)
        region = region_cell[iy, ix]
        print('选中区域：%d 格 / %d 像素（占 %.2f%%）| 该区域 chi2=%.3f，其余 chi2=%.3f'
              % (region_cell.sum(), region.sum(), 100 * region.mean(),
                 np.mean(chi[region] ** 2), np.mean(chi[~region] ** 2)))

    # 区域几何摘要（只用于描述，不进模型）
    print('区域 RA ∈ [%.3f, %.3f], Dec ∈ [%.3f, %.3f], 中心 (%.3f, %.3f)'
          % (ra[region].min(), ra[region].max(), dec[region].min(), dec[region].max(),
             np.median(ra[region]), np.median(dec[region])))

    os.makedirs(args.outdir, exist_ok=True)

    # ---------------- 3. 图 1：空间 ----------------
    fig, ax = plt.subplots(1, 3, figsize=(19, 6.2))
    ext = [ra_lo, ra_hi, de_lo, de_hi]
    im0 = ax[0].imshow(med, origin='lower', extent=ext, cmap='inferno',
                       vmin=np.nanpercentile(med[ok], 5), vmax=np.nanpercentile(med[ok], 99),
                       aspect='equal')
    ax[0].set_title('median $|\\chi|$ per %.2f deg cell' % args.cell)
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.02, label=r'median $|\chi|$')

    ax[1].imshow(med, origin='lower', extent=ext, cmap='Greys',
                 vmin=np.nanpercentile(med[ok], 5), vmax=np.nanpercentile(med[ok], 99),
                 aspect='equal')
    o = np.argwhere(region_cell)
    for i, j in o:
        ax[1].add_patch(Rectangle((ra_lo + j * args.cell, de_lo + i * args.cell),
                                  args.cell, args.cell, facecolor='crimson',
                                  edgecolor='none', alpha=0.75))
    ax[1].set_title('selected region (largest coherent hot blob)')

    no = np.argsort(np.abs(chi))
    ax[2].scatter(ra[no], dec[no], s=1.0, c=chi[no], cmap='coolwarm',
                  norm=Normalize(-3, 3), rasterized=True, linewidths=0)
    for i, j in o:
        ax[2].add_patch(Rectangle((ra_lo + j * args.cell, de_lo + i * args.cell),
                                  args.cell, args.cell, facecolor='none',
                                  edgecolor='lime', linewidth=1.2))
    ax[2].set_title(r'per-pixel $\chi$ + region outline (green)')
    for a in ax:
        a.set_xlabel('RA (deg)'); a.invert_xaxis()
    ax[0].set_ylabel('Dec (deg)')
    fig.tight_layout()
    f1 = os.path.join(args.outdir, 'fig_arm_region_map.png')
    fig.savefig(f1, dpi=160, bbox_inches='tight'); plt.close(fig)
    print('saved', f1)

    # ---------------- 4. 参数空间 ----------------
    # (a) 各特征 z 分布：区域 vs 其余
    # (b) k-NN 距离（以【训练集】为参照）→ 是否落在训练集稀疏区
    def knn_dist(Zq, Zref, k):
        """到参照集第 k 个近邻的距离（有 sklearn 用 sklearn，否则纯 numpy 兜底）。"""
        try:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=k + 1).fit(Zref)
            d, _ = nn.kneighbors(Zq)
            return d[:, -1]
        except ImportError:
            rng = np.random.default_rng(0)
            if len(Zref) > 15000:
                Zref = Zref[rng.choice(len(Zref), 15000, replace=False)]
            out = np.empty(len(Zq))
            for s in range(0, len(Zq), 1000):
                q = Zq[s:s + 1000]
                d2 = ((q[:, None, :] - Zref[None, :, :]) ** 2).sum(-1)
                kk = min(k, Zref.shape[0] - 1)
                out[s:s + 1000] = np.sqrt(np.partition(d2, kk, axis=1)[:, kk])
            return out

    knn_d = knn_dist(Zs, Zs[in_tr], args.knn)
    knn_tr = knn_d[in_tr]

    # 每特征的"越界"统计
    print('\n--- 参数空间 ---')
    print('%-10s %10s %10s %10s | %10s %10s | %s' %
          ('feature', 'z_med(reg)', 'z_med(rest)', '|z|>2(reg)', '|z|p84(reg)', '|z|p84(rest)', 'knn'))
    for j, f in enumerate(args.features):
        zr, zo = Zs[region, j], Zs[~region, j]
        print('%-10s %+10.3f %+10.3f %9.1f%% %11.2f %11.2f' %
              (f, np.median(zr), np.median(zo), 100 * np.mean(np.abs(zr) > 2),
               np.percentile(np.abs(zr), 84), np.percentile(np.abs(zo), 84)))

    # |chi| 随 knn 距离的变化（分位分箱）
    qs = np.percentile(knn_d, np.linspace(0, 100, 11))
    qs[-1] += 1e-9
    bi = np.digitize(knn_d, qs[:-1]) - 1
    bc, bm = [], []
    for b in range(10):
        m = bi == b
        if m.sum() >= 50:
            bc.append(0.5 * (qs[b] + qs[b + 1])); bm.append(np.median(achi[m]))
    print('knn 距离 中位: 区域=%.2f，其余=%.2f，训练集=%.2f (越大=越孤立/外推)'
          % (np.median(knn_d[region]), np.median(knn_d[~region]), np.median(knn_tr)))

    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    ax = ax.ravel()
    for j, f in enumerate(args.features[:4]):
        bins = np.linspace(-4, 4, 60)
        ax[j].hist(Zs[~region, j], bins=bins, density=True, color='lightgray',
                   label='rest (%.0f%%)' % (100 * (~region).mean()))
        ax[j].hist(Zs[region, j], bins=bins, density=True, histtype='step',
                   color='crimson', lw=2, label='selected region')
        ax[j].axvline(np.median(Zs[region, j]), color='crimson', ls='--', lw=1)
        ax[j].axvline(np.median(Zs[~region, j]), color='k', ls=':', lw=1)
        ax[j].set_title('%s  (z-score)' % f, fontsize=12)
        ax[j].set_xlabel('standardized log feature'); ax[j].legend(fontsize=8)
    bins = np.linspace(0, np.percentile(knn_d, 99.5), 60)
    ax[4].hist(knn_d[~region], bins=bins, density=True, color='lightgray', label='rest')
    ax[4].hist(knn_d[region], bins=bins, density=True, histtype='step',
               color='crimson', lw=2, label='selected region')
    ax[4].hist(knn_tr, bins=bins, density=True, histtype='step', color='steelblue',
               lw=1.5, ls='--', label='train (reference)')
    ax[4].set_title('k-NN distance in parameter space (k=%d)' % args.knn, fontsize=12)
    ax[4].set_xlabel('distance to k-th neighbour (sigma units)'); ax[4].legend(fontsize=8)
    ax[5].plot(bc, bm, 'o-', color='crimson', lw=2)
    ax[5].set_title(r'median $|\chi|$ vs k-NN distance'); ax[5].grid(alpha=.3)
    ax[5].set_xlabel('k-NN distance (sigma units)'); ax[5].set_ylabel(r'median $|\chi|$')
    fig.tight_layout()
    f2 = os.path.join(args.outdir, 'fig_arm_feature_space.png')
    fig.savefig(f2, dpi=160, bbox_inches='tight'); plt.close(fig)
    print('saved', f2)

    # 存 mask 供复用
    np.savez(os.path.join(args.outdir, 'arm_region_mask.npz'),
             ra=ra, dec=dec, region=region, chi=chi, cell=args.cell,
             ra_lo=ra_lo, de_lo=de_lo, nx=nx, ny=ny, hot=hot, med=med,
             features=np.array(args.features))
    print('saved', os.path.join(args.outdir, 'arm_region_mask.npz'))


if __name__ == '__main__':
    main()
