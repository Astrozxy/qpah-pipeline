#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
smooth_co.py — 对对齐后的 CO 图做几种不同程度的高斯平滑，并抽出逐行值。

动机：CO 图噪声主导（中位 S/N 0.71）。它的波束是 23"，我们的像素是 10"，
      所以相邻像素本来就相关；轻度平滑可以压掉重建引入的像素级噪声。
"""
import os, argparse
import numpy as np
import h5py
from astropy.io import fits
from astropy.wcs import WCS
from scipy import ndimage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--cache', default='sanity_cache')
    ap.add_argument('--h5', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    hd = fits.open(os.path.join(args.data, 'M31_S350_110_SSS_110_Model_All_qpah.fits'))
    w = WCS(hd[0].header).celestial
    shape = hd[0].data.shape
    hd.close()

    co = np.load(os.path.join(args.cache, 'CO_ex_mean.npy')).astype(float)
    pos = co > 0
    print('CO 图: 有效 %d px (%.2f%%) | 中位(>0) %.4f' % (pos.sum(), 100 * pos.mean(),
                                                     np.median(co[pos])))
    # 缺失处用 0 填充再平滑（面积权重另算），避免 NaN 扩散
    out = {'CO_raw': np.maximum(co, 0)}
    for sg in [1.0, 1.5, 2.0, 3.0]:
        num = ndimage.gaussian_filter(np.where(pos, co, 0.0), sg, mode='constant')
        den = ndimage.gaussian_filter(pos.astype(float), sg, mode='constant')
        sm = np.where(den > 0.5, num / np.maximum(den, 1e-12), np.nan)
        out['CO_sm%g' % sg] = np.maximum(sm, 0)
        print('  平滑 sigma=%.1f px : 有效 %d | 中位 %.4f' %
              (sg, int(np.isfinite(sm).sum()), np.nanmedian(sm)))

    with h5py.File(args.h5) as f:
        ra = f['ra'][:].astype(np.float64); dec = f['dec'][:].astype(np.float64)
    x, y = w.world_to_pixel_values(ra, dec)
    ix = np.clip(np.round(x).astype(int), 0, shape[1] - 1)
    iy = np.clip(np.round(y).astype(int), 0, shape[0] - 1)
    res = {}
    for k, a in out.items():
        res[k] = a[iy, ix].astype(np.float32)
        print('  逐行 %-10s 有效(>=0) %.1f%% | 中位 %.4f' %
              (k, 100 * np.mean(np.isfinite(res[k])), np.nanmedian(res[k])))
    np.savez_compressed(args.out, **res)
    print('saved %s (%.2f MB)' % (args.out, os.path.getsize(args.out) / 1e6))


if __name__ == '__main__':
    main()
