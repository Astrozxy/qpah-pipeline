#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_dustmaps.py — 把尘埃图摊开看：原生 850um、对齐后的 dust_density、
DL07 的 Ldust / Mdust、以及 qPAH 目标，并给出散点图诊断。
"""
import os, argparse
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def show(ax, arr, title, cmap='inferno', log=False, pct=(2, 98)):
    a = arr.astype(float).copy()
    if log:
        a = np.where(a > 0, a, np.nan)
    g = a[np.isfinite(a)]
    if g.size == 0:
        ax.set_title(title); return
    lo, hi = np.percentile(g, pct)
    if log:
        from matplotlib.colors import LogNorm
        norm = LogNorm(vmin=max(lo, g[g > 0].min()), vmax=hi)
    else:
        norm = None
    im = ax.imshow(a, origin='lower', cmap=cmap, vmin=None if log else lo,
                   vmax=None if log else hi, norm=norm)
    ax.set_title(title, fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--cache', default='sanity_cache')
    ap.add_argument('--out', default='dustmaps.png')
    args = ap.parse_args()
    D, C = args.data, args.cache

    native = fits.open(os.path.join(D, 'M31_dust_NH2_sm24.fits'))[0].data.astype(float)
    raw850 = fits.open(os.path.join(D, 'm31_combined850_um.fits'))[0].data.astype(float)
    hq = fits.open(os.path.join(D, 'M31_S350_110_SSS_110_Model_All_qpah.fits'))[0]
    w = WCS(hq.header).celestial
    qpah = hq.data.astype(float)
    Ld = fits.open(os.path.join(D, 'M31_S350_110_SSS_110_Model_All_SurfBr_Ldust.fits'))[0].data.astype(float)
    Md = fits.open(os.path.join(D, 'M31_S350_110_SSS_110_Model_All_SurfBr_Mdust.fits'))[0].data.astype(float)

    aligned = np.load(os.path.join(C, 'dust_ex_mean.npy')).astype(float)
    cov = np.load(os.path.join(C, 'dust_ex_cov.npy')).astype(float)

    fig, ax = plt.subplots(2, 3, figsize=(21, 12))
    show(ax[0, 0], native, 'native 850um (M31_dust_NH2_sm24, mJy/beam)')
    show(ax[0, 1], raw850, 'native combined 850um (m31_combined850_um)')
    show(ax[0, 2], aligned, 'aligned dust_density on qPAH grid (10")')
    show(ax[1, 0], qpah, 'qPAH (target)')
    show(ax[1, 1], Ld, 'Ldust (DL07)')
    show(ax[1, 2], Md, 'Mdust (DL07)')

    # 覆盖
    print('native 850um : shape %s | finite %.1f%% >0 %.1f%% (of finite) | median>0 %.4g'
          % (native.shape, 100 * np.isfinite(native).mean(),
             100 * np.mean(native[np.isfinite(native)] > 0),
             np.median(native[native > 0])))
    print('aligned      : cov>0.5 占 %.1f%% | >0 中位 %.4g'
          % (100 * np.mean(cov > 0.5), np.median(aligned[aligned > 0])))
    print('Ldust        : >0 占 %.1f%% | 中位 %.4g' % (100 * np.mean(Ld > 0), np.median(Ld[Ld > 0])))
    print('qPAH         : >0 占 %.1f%%' % (100 * np.mean(qpah > 0)))

    # 散点诊断：只在对齐后有效的像素
    m = (cov > 0.5) & (aligned > 0) & (qpah > 0) & (Ld > 0)
    fig2, ax2 = plt.subplots(1, 3, figsize=(19, 5.6))
    x = np.log10(aligned[m]); yv = qpah[m]; yl = np.log10(Ld[m])
    ax2[0].hexbin(x, yv, gridsize=60, bins='log', cmap='viridis')
    ax2[0].set_xlabel('log10 dust_density (850um, aligned)'); ax2[0].set_ylabel('qPAH (%)')
    ax2[0].set_title('dust_density vs qPAH   (Pearson %+.3f)' % np.corrcoef(x, yv)[0, 1])
    ax2[1].hexbin(yl, yv, gridsize=60, bins='log', cmap='viridis')
    ax2[1].set_xlabel('log10 Ldust (DL07)'); ax2[1].set_ylabel('qPAH (%)')
    ax2[1].set_title('Ldust vs qPAH   (Pearson %+.3f)' % np.corrcoef(yl, yv)[0, 1])
    ax2[2].hexbin(yl, x, gridsize=60, bins='log', cmap='viridis')
    ax2[2].set_xlabel('log10 Ldust (DL07)'); ax2[2].set_ylabel('log10 dust_density (850um)')
    ax2[2].set_title('Ldust vs dust_density  (Pearson %+.3f)' % np.corrcoef(yl, x)[0, 1])
    for a in ax2: a.grid(alpha=.2)
    fig2.tight_layout(); fig2.savefig(args.out.replace('.png', '_scatter.png'), dpi=140, bbox_inches='tight')
    fig.tight_layout(); fig.savefig(args.out, dpi=130, bbox_inches='tight')
    print('\nsaved', args.out, 'and', args.out.replace('.png', '_scatter.png'))


if __name__ == '__main__':
    main()
