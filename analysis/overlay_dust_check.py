#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
overlay_dust_check.py — 用叠加等高线检验尘埃图与 qPAH 图是否真的对齐。
任何错位都会在叠加图上直接看出来。
"""
import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    D, C = 'data', 'sanity_cache'
    hq = fits.open(os.path.join(D, 'M31_S350_110_SSS_110_Model_All_qpah.fits'))[0]
    w = WCS(hq.header).celestial
    qpah = hq.data.astype(float)
    Ld = fits.open(os.path.join(D, 'M31_S350_110_SSS_110_Model_All_SurfBr_Ldust.fits'))[0].data.astype(float)
    old = np.load(os.path.join(C, 'dust_ex_mean.npy')).astype(float)
    new = np.load(os.path.join(C, 'dustc_ex_mean.npy')).astype(float)

    fig, ax = plt.subplots(1, 4, figsize=(26, 7))

    # 1) qPAH 灰度 + 两张尘埃图的等高线
    g = np.where(qpah > 0, qpah, np.nan)
    ax[0].imshow(g, origin='lower', cmap='gray_r', vmin=0, vmax=np.nanpercentile(g, 99))
    for a, c, lab in [(old, 'red', 'old dust (sm24)'), (new, 'lime', 'new dust (combined)')]:
        m = np.isfinite(a) & (a > 0)
        ax[0].contour(m.astype(float), levels=[0.5], colors=c, linewidths=1.6)
        ax[0].contour(np.where(m, a, np.nan), levels=4, colors=c, linewidths=0.5, alpha=.6)
    ax[0].set_title('qPAH (gray) + dust footprints\nred=old sm24, green=new combined', fontsize=11)

    # 2) Ldust 灰度 + 尘埃等高线
    ax[1].imshow(np.where(Ld > 0, Ld, np.nan), origin='lower', cmap='gray_r',
                 vmin=0, vmax=np.nanpercentile(Ld[Ld > 0], 99))
    ax[1].contour((np.isfinite(new) & (new > 0)).astype(float), levels=[0.5], colors='lime', linewidths=1.6)
    ax[1].set_title('Ldust (gray) + new dust footprint (green)', fontsize=11)

    # 3) 新尘埃图 + qPAH 等高线
    ax[2].imshow(np.where(new > 0, new, np.nan), origin='lower', cmap='inferno')
    ax[2].contour(np.where(qpah > 0, qpah, np.nan), levels=6, colors='cyan', linewidths=0.8)
    ax[2].set_title('new dust (color) + qPAH contours (cyan)', fontsize=11)

    # 4) 新 vs 旧 的比值
    both = (new > 0) & (old > 0)
    ratio = np.full_like(new, np.nan)
    ratio[both] = new[both] / old[both]
    im = ax[3].imshow(ratio, origin='lower', cmap='coolwarm', vmin=0, vmax=40)
    ax[3].set_title('new / old dust ratio (unit conversion + masking)', fontsize=11)
    plt.colorbar(im, ax=ax[3], fraction=0.046, pad=0.02)

    for a in ax:
        a.set_xlim(380, 900); a.set_ylim(300, 950)
    fig.tight_layout()
    fig.savefig('overlay_dust.png', dpi=130, bbox_inches='tight')
    print('saved overlay_dust.png')

    # 定量：把两张尘埃图在 qPAH 高值处归一化后比较
    m = both & (qpah > 0)
    r = new[m] / old[m]
    print('new/old 比值（公共像素）: 中位 %.2f p16 %.2f p84 %.2f' %
          (np.median(r), np.percentile(r, 16), np.percentile(r, 84)))
    print('预期单位换算: mJy/arcsec^2 -> mJy/beam，beam=24.9" '
          '=> 面积 = pi*(24.9/2)^2/ln2 = %.1f arcsec^2' % (np.pi * (24.9 / 2) ** 2 / np.log(2)))


if __name__ == '__main__':
    main()
