#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_dustmaps.py — 把尘埃图摊开看。

★ 关键：所有面板统一到【qPAH 网格的像素坐标】。每张原生图用它自己的 WCS 算出
   四个角在 qPAH 像素坐标下的位置，再用 imshow(extent=...) 放置。
   这样各图的【真实相对大小】才可比，不会出现"有的图大有的图小"。
"""
import os, argparse
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def qpah_extent(arr_shape, wcs_in, wcs_out):
    """把一张 (ny,nx) 图的范围表示成 qPAH 像素坐标下的 extent=[x0,x1,y0,y1]。"""
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
    ap.add_argument('--out', default='dustmaps.png')
    args = ap.parse_args()
    D, C = args.data, args.cache

    hq = fits.open(os.path.join(D, 'M31_S350_110_SSS_110_Model_All_qpah.fits'))[0]
    wout = WCS(hq.header).celestial
    qpah = hq.data.astype(float)
    shape = qpah.shape

    def load(fn):
        hd = fits.open(os.path.join(D, fn))
        a = hd[0].data.astype(float)
        w = WCS(hd[0].header).celestial
        hd.close()
        return a, w

    Ld, _ = load('M31_S350_110_SSS_110_Model_All_SurfBr_Ldust.fits')
    Md, _ = load('M31_S350_110_SSS_110_Model_All_SurfBr_Mdust.fits')
    n850, w850 = load('M31_dust_NH2_sm24.fits')
    comb, wcomb = load('m31_combined850_um.fits')
    old = np.load(os.path.join(C, 'dust_ex_mean.npy')).astype(float)
    new = np.load(os.path.join(C, 'dustc_ex_mean.npy')).astype(float)

    # 统一 xlim/ylim：覆盖全部内容
    X0, X1, Y0, Y1 = 330, 950, 280, 960

    panels = [
        ('qPAH (target, 10")', qpah, wout, 'inferno', (0, None)),
        ('Ldust (DL07, 10")', Ld, wout, 'inferno', (0, None)),
        ('Mdust (DL07, 10")', Md, wout, 'inferno', (0, None)),
        ('native 850um  sm24 (6")', n850, w850, 'inferno', (0, None)),
        ('native 850um  combined (4")', comb, wcomb, 'inferno', (0, None)),
        ('aligned dust_density (CORRUPT cols)', old, wout, 'inferno', (0, None)),
        ('aligned dust850 (fixed)', new, wout, 'inferno', (0, None)),
    ]
    fig, ax = plt.subplots(2, 4, figsize=(26, 13))
    ax = ax.ravel()
    for a, (lab, arr, w, cm, _) in zip(ax, panels):
        ext = qpah_extent(arr.shape, w, wout)
        g = arr[np.isfinite(arr) & (arr > 0)]
        lo, hi = (np.percentile(g, [2, 98]) if g.size else (0, 1))
        im = a.imshow(arr, origin='lower', cmap=cm, extent=ext,
                      vmin=lo, vmax=hi, interpolation='nearest')
        a.set_xlim(X0, X1); a.set_ylim(Y0, Y1)
        # 数据实际覆盖的范围（原生图=整幅；对齐图=只有尘埃覆盖的那块）
        v = np.isfinite(arr) & (arr > 0)
        yy, xx = np.where(v)
        psc = proj_plane_pixel_scales(w) * 3600.0   # 本图自己的像素尺度
        if v.sum() > 10:
            dx = (xx.max() - xx.min() + 1) * psc[0] / 3600
            dy = (yy.max() - yy.min() + 1) * psc[1] / 3600
        else:
            dx = dy = 0.0
        a.set_title('%s\ndata covers %.3f x %.3f deg' % (lab, dx, dy), fontsize=11)
        a.set_xlabel('qPAH grid x (10")'); a.set_ylabel('qPAH grid y (10")')
        plt.colorbar(im, ax=a, fraction=0.046, pad=0.02)

    # 第 8 格：三张图的 footprint 轮廓叠加
    a = ax[7]
    a.imshow(np.where(qpah > 0, qpah, np.nan), origin='lower', cmap='gray_r',
             extent=[0, shape[1], 0, shape[0]], vmin=0, vmax=np.nanpercentile(qpah[qpah > 0], 99))
    for arr, w, c, lab in [(n850, w850, 'red', 'sm24'), (comb, wcomb, 'lime', 'combined')]:
        e = qpah_extent(arr.shape, w, wout)
        a.add_patch(plt.Rectangle((e[0], e[2]), e[1] - e[0], e[3] - e[2],
                                  fill=False, ec=c, lw=2, label='%s footprint' % lab))
    a.legend(fontsize=9, loc='lower left')
    a.set_xlim(X0, X1); a.set_ylim(Y0, Y1)
    a.set_title('footprints on the qPAH grid')
    a.set_xlabel('qPAH grid x (10")'); a.set_ylabel('qPAH grid y (10")')

    fig.tight_layout()
    fig.savefig(args.out, dpi=120, bbox_inches='tight')
    print('saved', args.out)
    print()
    print('  %-38s %-14s %7s %20s %18s' % ('panel','native shape','px(arcsec)','data extent (deg)','array footprint (deg)'))
    for lab, arr, w, _, _ in panels:
        v = np.isfinite(arr) & (arr > 0)
        yy, xx = np.where(v)
        psc = proj_plane_pixel_scales(w) * 3600.0
        if v.sum() > 10:
            dx = (xx.max() - xx.min() + 1) * psc[0] / 3600
            dy = (yy.max() - yy.min() + 1) * psc[1] / 3600
        else:
            dx = dy = 0.0
        e = qpah_extent(arr.shape, w, wout)
        print('  %-38s %-14s %7.2f %20s %18s'
              % (lab, str(arr.shape), psc[0], '%.3f x %.3f' % (dx, dy),
                 '%.3f x %.3f' % ((e[1] - e[0]) * 10 / 3600, (e[3] - e[2]) * 10 / 3600)))


if __name__ == '__main__':
    main()
