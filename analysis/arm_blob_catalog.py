#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
arm_blob_catalog.py — 把误差图上【所有】显著的空间连贯团块列出来编号，供人工挑选。

动机：自动"取最大团块"会挑到最外侧那条；用户关心的是靠中心的那条。
本脚本输出编号目录 + 一张带编号的图，便于指认。

判据：天区切 0.02° 粗格（每格 >=20 像素），算每格【中位 |χ|】；
      超过全体合格格 p{pct} 的格为热点；对热点做 4-连通标记；
      列出格数 >= min_cells 的所有团块，给出：
        编号 / 格数 / 像素数 / 该块 χ²_red / 去投影半径中位 / RA,Dec 范围

输出:
  fig_arm_blob_catalog.png    中位|χ| 格图 + 编号方框
  arm_blob_labels.npz         每个像素的团块编号（供后续单独提取某一块）
"""
import os, argparse
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mplconfig')
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize

FEATS = ['H1', 'sfr', 'CO', 'H1_ew']
RA0, DEC0, INC, PA = 10.6847, 41.2690, np.radians(77.7), np.radians(37.7)


def deproj_R(ra, dec):
    dx = (ra - RA0) * np.cos(np.radians(DEC0)); dy = dec - DEC0
    x = dx * np.cos(PA) + dy * np.sin(PA)
    y = (-dx * np.sin(PA) + dy * np.cos(PA)) / np.cos(INC)
    return np.sqrt(x ** 2 + y ** 2) * 3600.0 / (206265.0 / 744.0)


def label_components(mask):
    lab = np.zeros(mask.shape, dtype=int); n = 0
    ny, nx = mask.shape
    for i in range(ny):
        for j in range(nx):
            if mask[i, j] and lab[i, j] == 0:
                n += 1; stack = [(i, j)]; lab[i, j] = n
                while stack:
                    a, b = stack.pop()
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        p, q = a + da, b + db
                        if 0 <= p < ny and 0 <= q < nx and mask[p, q] and lab[p, q] == 0:
                            lab[p, q] = n; stack.append((p, q))
    return lab, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', default='data/M31_1d_array_nodust.h5')
    ap.add_argument('--err-floor', type=float, default=0.2)
    ap.add_argument('--cell', type=float, default=0.02)
    ap.add_argument('--min-per-cell', type=int, default=20)
    ap.add_argument('--pct', type=float, default=88.0)
    ap.add_argument('--min-cells', type=int, default=4)
    ap.add_argument('--outdir', default='results/armregion')
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

    np.random.seed(0)
    ang = (np.arctan2(dec - 41.25, ra - 10.75) + 2 * np.pi) % (2 * np.pi)
    sec = np.digitize(ang, np.linspace(0, 2 * np.pi, 11)) - 1
    tr_s = np.random.choice(np.arange(10), size=6, replace=False)
    in_tr = np.isin(sec, tr_s)

    floors = np.array([max(1e-9, 0.01 * np.median(X[in_tr, j][X[in_tr, j] > 0]))
                       for j in range(X.shape[1])])
    Z = np.log10(X + floors[None, :]); mu, sd = Z[in_tr].mean(0), Z[in_tr].std(0) + 1e-12
    D = np.column_stack([np.ones(len(Z)), (Z - mu) / sd])
    w = 1.0 / sig[in_tr] ** 2
    coef = np.linalg.solve(D[in_tr].T @ (w[:, None] * D[in_tr]), D[in_tr].T @ (w * y[in_tr]))
    chi = (D @ coef - y) / sig
    R = deproj_R(ra, dec)

    ra_lo, de_lo = ra.min(), dec.min()
    nx = int(np.ceil((ra.max() - ra_lo) / args.cell))
    ny = int(np.ceil((dec.max() - de_lo) / args.cell))
    ix = np.clip(((ra - ra_lo) / args.cell).astype(int), 0, nx - 1)
    iy = np.clip(((dec - de_lo) / args.cell).astype(int), 0, ny - 1)
    cid = iy * nx + ix
    ncell = nx * ny
    achi = np.abs(chi)
    o = np.argsort(cid, kind='stable')
    sid, sa = cid[o], achi[o]
    bnd = np.searchsorted(sid, np.arange(ncell + 1))
    cnt = np.bincount(cid, minlength=ncell)
    med = np.full(ncell, np.nan)
    for c in range(ncell):
        a, b = bnd[c], bnd[c + 1]
        if b - a >= args.min_per_cell:
            med[c] = np.median(sa[a:b])
    med = med.reshape(ny, nx)
    ok = (cnt >= args.min_per_cell).reshape(ny, nx)
    thr = np.nanpercentile(med[ok], args.pct)
    hot = ok & (med > thr)
    lab_c, nc = label_components(hot)

    px_lab = lab_c[iy, ix]                     # 每个像素所属团块（0 = 不在任何块）
    rows = []
    for k in range(1, nc + 1):
        ncellk = int((lab_c == k).sum())
        if ncellk < args.min_cells:
            continue
        m = px_lab == k
        rows.append(dict(id=k, cells=ncellk, npx=int(m.sum()),
                         chi2=float(np.mean(chi[m] ** 2)),
                         Rmed=float(np.median(R[m])),
                         Rlo=float(np.percentile(R[m], 5)), Rhi=float(np.percentile(R[m], 95)),
                         ra_c=float(np.median(ra[m])), de_c=float(np.median(dec[m])),
                         ra_w=float(np.ptp(ra[m])), de_w=float(np.ptp(dec[m]))))
    rows.sort(key=lambda r: r['Rmed'])
    print('阈值 = 合格格中位|chi| 的 p%.0f = %.3f ；连通域 %d 个，>=%d 格的 %d 个'
          % (args.pct, thr, nc, args.min_cells, len(rows)))
    print()
    print('%3s %6s %7s %8s %9s %14s %16s %12s' %
          ('#', 'cells', 'npx', 'chi2', 'R_med', 'R_5-95', 'center(RA,Dec)', 'size(deg)'))
    for r in rows:
        print('%3d %6d %7d %8.3f %9.1f %6.1f-%5.1f  (%7.3f,%7.3f) %5.3fx%5.3f'
              % (r['id'], r['cells'], r['npx'], r['chi2'], r['Rmed'], r['Rlo'], r['Rhi'],
                 r['ra_c'], r['de_c'], r['ra_w'], r['de_w']))

    os.makedirs(args.outdir, exist_ok=True)
    np.savez(os.path.join(args.outdir, 'arm_blob_labels.npz'),
             ra=ra, dec=dec, chi=chi, R=R, px_lab=px_lab, lab_c=lab_c, med=med,
             ra_lo=ra_lo, de_lo=de_lo, cell=args.cell, nx=nx, ny=ny, hot=hot,
             ids=np.array([r['id'] for r in rows]))

    # ---------------- 图：编号 ----------------
    fig, ax = plt.subplots(1, 2, figsize=(19, 8.5))
    ext = [ra_lo, ra_lo + nx * args.cell, de_lo, de_lo + ny * args.cell]
    im = ax[0].imshow(med, origin='lower', extent=ext, cmap='inferno',
                      vmin=np.nanpercentile(med[ok], 5), vmax=np.nanpercentile(med[ok], 99.5),
                      aspect='equal')
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.02, label=r'median $|\chi|$')
    ax[0].set_title('median $|\\chi|$ per %.2f deg cell  (numbered blobs = p%.0f hot spots)'
                    % (args.cell, args.pct))

    ax[1].scatter(ra, dec, s=0.7, c=chi, cmap='coolwarm', norm=Normalize(-3, 3),
                  rasterized=True, linewidths=0)
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-3, 3), cmap='coolwarm'),
                 ax=ax[1], fraction=0.046, pad=0.02, label=r'$\chi$')
    ax[1].set_title('per-pixel $\\chi$ + numbered blobs')

    for r in rows:
        k = r['id']
        m = lab_c == k
        for i, j in np.argwhere(m):
            for a in ax:
                a.add_patch(Rectangle((ra_lo + j * args.cell, de_lo + i * args.cell),
                                      args.cell, args.cell, facecolor='none',
                                      edgecolor='lime', lw=1.0))
        a = ax[0]
        a.annotate(str(k), (r['ra_c'], r['de_c']), color='cyan', fontsize=13,
                   fontweight='bold', ha='center', va='center')
    for a in ax:
        a.set_xlabel('RA (deg)'); a.set_ylabel('Dec (deg)'); a.invert_xaxis()
    fig.tight_layout()
    f = os.path.join(args.outdir, 'fig_arm_blob_catalog.png')
    fig.savefig(f, dpi=150, bbox_inches='tight'); plt.close(fig)
    print('\nsaved', f)


if __name__ == '__main__':
    main()
