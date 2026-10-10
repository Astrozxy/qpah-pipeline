#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_ew_compare.py — 把 HI 等效宽度和 CO 等效宽度画在一起对比。

都放在 qPAH 网格上，叠加悬臂轮廓，并给出两者的散点图。
"""
import os, argparse
import numpy as np
import h5py
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def extent_in(arr_shape, wcs_in, wcs_out):
    ny, nx = arr_shape
    px = np.array([0, nx - 1, 0, nx - 1], float)
    py = np.array([0, 0, ny - 1, ny - 1], float)
    ra, dec = wcs_in.pixel_to_world_values(px, py)
    qx, qy = wcs_out.world_to_pixel_values(ra, dec)
    return [qx.min(), qx.max(), qy.min(), qy.max()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--cache', default='sanity_cache')
    ap.add_argument('--h5', default='data/M31_1d_array_nodust.h5')
    ap.add_argument('--ewnpz', default='coew_nodust.npz')
    ap.add_argument('--out', default='ew_compare.png')
    args = ap.parse_args()
    D, C = args.data, args.cache

    hq = fits.open(os.path.join(D, 'M31_S350_110_SSS_110_Model_All_qpah.fits'))
    wout = WCS(hq[0].header).celestial
    qpah = hq[0].data.astype(float)
    shape = qpah.shape
    hq.close()

    # 原生 HI EW
    hd = fits.open(os.path.join(D, 'm31_C+D+tp_hilores_120pc_strict_ew.fits'))
    hi_native = hd[0].data.astype(float)
    whi = WCS(hd[0].header).celestial
    hd.close()
    # 对齐后的 HI EW（管线已有）
    hi_ew = np.load(os.path.join(C, 'HI_ew_ex_mean.npy')).astype(float)
    # 对齐后的 CO EW
    co_ew = np.load(os.path.join(C, 'COew_ex_mean.npy')).astype(float)

    # 悬臂 mask
    with h5py.File(args.h5) as f:
        ra = f['ra'][:].astype(float); dec = f['dec'][:].astype(float)
    zm = np.load('mask_arm_final.npz') if os.path.exists('mask_arm_final.npz') else None
    arm_pts = None
    if zm is not None:
        m = zm['region'].astype(bool)
        arm_pts = (zm['ra'][m], zm['dec'][m])

    X0, X1, Y0, Y1 = 380, 900, 300, 950

    fig = plt.figure(figsize=(26, 13))
    gs = fig.add_gridspec(2, 4, hspace=0.22, wspace=0.16)

    def show(ax, arr, w, title, ext=None, cmap='inferno', lo=None, hi=None, log=False):
        e = ext if ext is not None else [0, shape[1], 0, shape[0]]
        v = np.isfinite(arr) & (arr > 0)
        if lo is None:
            g = arr[v]
            lo, hi = (np.nanpercentile(g, [2, 98]) if g.size else (0, 1))
        kw = dict(norm=LogNorm(vmin=max(lo, 1e-3), vmax=hi)) if log else dict(vmin=lo, vmax=hi)
        im = ax.imshow(arr, origin='lower', cmap=cmap, extent=e, interpolation='nearest', **kw)
        if arm_pts is not None:
            ax.plot(*wout.world_to_pixel_values(*arm_pts), '.', ms=0.7, color='lime', alpha=.55)
        ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('qPAH grid x (10")'); ax.set_ylabel('qPAH grid y (10")')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    ax1 = fig.add_subplot(gs[0, 0])
    show(ax1, np.where(qpah > 0, qpah, np.nan), wout, 'qPAH (target)  [green = arm]')
    ax2 = fig.add_subplot(gs[0, 1])
    show(ax2, np.where(hi_ew > 0, hi_ew, np.nan), wout, 'HI equivalent width, aligned (km/s)')
    ax3 = fig.add_subplot(gs[0, 2])
    show(ax3, np.where(co_ew > 0, co_ew, np.nan), wout, 'CO equivalent width, aligned (km/s)')
    ax4 = fig.add_subplot(gs[0, 3])
    # 原生 HI EW
    e = extent_in(hi_native.shape, whi, wout)
    g = hi_native[np.isfinite(hi_native) & (hi_native > 0)]
    im = ax4.imshow(hi_native, origin='lower', cmap='inferno', extent=e,
                    vmin=np.percentile(g, 2), vmax=np.percentile(g, 98), interpolation='nearest')
    ax4.plot(*wout.world_to_pixel_values(*arm_pts), '.', ms=0.7, color='lime', alpha=.55)
    ax4.set_xlim(X0, X1); ax4.set_ylim(Y0, Y1)
    ax4.set_title('native HI EW (32", %s)' % str(hi_native.shape), fontsize=12)
    ax4.set_xlabel('qPAH grid x (10")'); ax4.set_ylabel('qPAH grid y (10")')
    fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.02)

    # 散点
    with h5py.File(args.h5) as f:
        H1 = f['H1'][:].astype(float); H1E = f['H1_ew'][:].astype(float)
    zc = np.load(args.ewnpz)
    m = np.isfinite(zc['COew']) & (zc['COew'] > 0) & (H1E > 0)
    a = np.log10(H1E[m]); b = np.log10(zc['COew'][m])
    ax5 = fig.add_subplot(gs[1, 0])
    hb = ax5.hexbin(a, b, gridsize=55, bins='log', cmap='viridis')
    ax5.set_xlabel('log10 HI_ew (km/s)'); ax5.set_ylabel('log10 CO_ew (km/s)')
    ax5.set_title('HI_ew vs CO_ew   n=%d   Pearson %+.3f' % (m.sum(), np.corrcoef(a, b)[0, 1]))
    fig.colorbar(hb, ax=ax5, fraction=0.046, pad=0.02)

    # 一维分布
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.hist(np.log10(H1E[H1E > 0]), bins=70, alpha=.6, label='HI_ew (med %.1f)' % np.median(H1E[H1E > 0]), color='steelblue')
    ax6.hist(b, bins=70, alpha=.6, label='CO_ew (med %.1f)' % np.median(zc['COew'][m]), color='crimson')
    ax6.set_xlabel('log10 EW (km/s)'); ax6.set_ylabel('N'); ax6.legend(fontsize=10)
    ax6.set_title('distributions')

    # 臂 vs 对照：各自
    if arm_pts is not None:
        from scipy.spatial import cKDTree
        t = cKDTree(np.column_stack([ra, dec]))
        dd, ii = t.query(np.column_stack([zm['ra'][zm['region'].astype(bool)],
                                          zm['dec'][zm['region'].astype(bool)]]), k=1)
        armm = np.zeros(len(ra), bool); armm[ii[dd < 1e-7]] = True
        ax7 = fig.add_subplot(gs[1, 2])
        v1 = H1E[(H1E > 0) & armm]; v2 = H1E[(H1E > 0) & ~armm]
        ax7.hist(np.log10(v1), bins=50, alpha=.65, density=True, label='arm (n=%d, med %.1f)' % (len(v1), np.median(v1)), color='crimson')
        ax7.hist(np.log10(v2), bins=50, alpha=.55, density=True, label='rest (med %.1f)' % np.median(v2), color='gray')
        ax7.set_xlabel('log10 HI_ew'); ax7.legend(fontsize=10); ax7.set_title('HI_ew: arm vs rest')
        ax8 = fig.add_subplot(gs[1, 3])
        w1 = zc['COew'][(zc['COew'] > 0) & armm]; w2 = zc['COew'][(zc['COew'] > 0) & ~armm]
        ax8.hist(np.log10(w1), bins=40, alpha=.65, density=True, label='arm (n=%d, med %.1f)' % (len(w1), np.median(w1)), color='crimson')
        ax8.hist(np.log10(w2), bins=60, alpha=.55, density=True, label='rest (med %.1f)' % np.median(w2), color='gray')
        ax8.set_xlabel('log10 CO_ew'); ax8.legend(fontsize=10); ax8.set_title('CO_ew: arm vs rest')

    fig.savefig(args.out, dpi=120, bbox_inches='tight')
    print('saved', args.out)
    print('  HI_ew 对齐后有效 %.1f%% | CO_ew 对齐后有效 %.1f%%'
          % (100 * np.mean(np.isfinite(hi_ew) & (hi_ew > 0)),
             100 * np.mean(np.isfinite(co_ew) & (co_ew > 0))))


if __name__ == '__main__':
    main()
