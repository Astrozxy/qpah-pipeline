#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dust850_noise.py — 从 850um 图本身经验性地估噪声，看它是否在 footprint 边缘升高。

方法：噪声 ≈ 图 减去 平滑版 的局部散射（信号是平滑的，残差主要是噪声）。
输出：
    <cache>/dust850_noise.npy     (1250x1250, 对齐到 qPAH 网格的噪声估计)
    <out>  逐行 npz: dust850 / dust850_sigma / dust850_snr / edge
"""
import os, argparse
import numpy as np
import h5py
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_exact
from scipy import ndimage


def reproj(data, wcs_in, wout, shape):
    V = np.isfinite(data)
    num, _ = reproject_exact((np.where(V, data, 0.0), wcs_in), wout, shape_out=shape)
    den, _ = reproject_exact((V.astype(float), wcs_in), wout, shape_out=shape)
    out = np.full(shape, np.nan)
    ok = (den > 0.5) & np.isfinite(num) & (den > 0)
    out[ok] = num[ok] / den[ok]
    return out, den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--cache', default='sanity_cache')
    ap.add_argument('--src', default='m31_combined850_um.fits')
    ap.add_argument('--box', type=int, default=15, help='局部散射的方框边长（原生像素）')
    ap.add_argument('--h5', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    hq = fits.open(os.path.join(args.data, 'M31_S350_110_SSS_110_Model_All_qpah.fits'))
    wout = WCS(hq[0].header).celestial
    shape = hq[0].data.shape
    hq.close()

    hd = fits.open(os.path.join(args.data, args.src))
    m = hd[0].data.astype(np.float64)
    wm = WCS(hd[0].header).celestial
    hd.close()
    pos = np.isfinite(m)
    print('%s: %s | 有效 %.1f%% | 中位(>0) %.4g' %
          (args.src, m.shape, 100 * pos.mean(), np.median(m[pos & (m > 0)])))

    # 平滑（信号是平滑的）后取残差
    sm = ndimage.gaussian_filter(np.where(pos, m, 0.0), 3.0, mode='constant')
    wgt = ndimage.gaussian_filter(pos.astype(float), 3.0, mode='constant')
    sm = np.where(wgt > 0.5, sm / np.maximum(wgt, 1e-12), np.nan)
    res = m - sm
    # 局部 RMS
    b = args.box
    r2 = np.where(pos, res ** 2, 0.0)
    n = ndimage.uniform_filter(pos.astype(float), b, mode='constant') * b * b
    s2 = ndimage.uniform_filter(r2, b, mode='constant') * b * b
    sig = np.sqrt(np.where(n > 0.3 * b * b, s2 / np.maximum(n - 1, 1), np.nan))
    print('局部噪声(RMS): 中位 %.4g | p16 %.4g p84 %.4g' %
          (np.nanmedian(sig[pos]), np.nanpercentile(sig[pos], 16),
           np.nanpercentile(sig[pos], 84)))

    sn, _ = reproj(sig, wm, wout, shape)
    np.save(os.path.join(args.cache, 'dust850_noise.npy'), sn.astype(np.float32))
    dm, cov = reproj(np.where(pos, m, np.nan), wm, wout, shape)
    covm = np.load(os.path.join(args.cache, 'dustc_ex_cov.npy')).astype(float)
    foot = np.isfinite(dm) & (covm > 0.5)
    ed = ndimage.distance_transform_edt(foot)
    print('对齐后: 噪声中位 %.4g | 覆盖率 %.2f%%' %
          (np.nanmedian(sn[np.isfinite(sn)]), 100 * foot.mean()))

    # 噪声 vs 边缘距（在全网格上）
    print('\n=== 噪声 vs 到 footprint 边界的距离（qPAH 网格像素）===')
    print('%-14s %10s %14s %12s' % ('边缘距(px)', 'n', '噪声中位', '噪声/整体'))
    g0 = np.nanmedian(sn[foot])
    for lo, hi in [(0, 3), (3, 6), (6, 10), (10, 20), (20, 40), (40, 10 ** 9)]:
        k = foot & (ed >= lo) & (ed < hi) & np.isfinite(sn)
        if k.sum() < 50: continue
        print('%-14s %10d %14.4g %12.2f' % ('%d-%s' % (lo, 'inf' if hi > 1e8 else hi),
                                            k.sum(), np.nanmedian(sn[k]),
                                            np.nanmedian(sn[k]) / g0))

    with h5py.File(args.h5) as f:
        ra = f['ra'][:].astype(np.float64); dec = f['dec'][:].astype(np.float64)
    x, y = wout.world_to_pixel_values(ra, dec)
    ix = np.clip(np.round(x).astype(int), 0, shape[1] - 1)
    iy = np.clip(np.round(y).astype(int), 0, shape[0] - 1)
    dv = dm[iy, ix]; sv = sn[iy, ix]; ev = ed[iy, ix]
    with np.errstate(invalid='ignore', divide='ignore'):
        snr = np.where(sv > 0, np.abs(dv) / sv, np.nan)
    np.savez_compressed(args.out, dust850=dv, dust850_sigma=sv, dust850_snr=snr, edge=ev)
    ok = np.isfinite(sv) & (sv > 0)
    print('\n逐行: 噪声可用 %.1f%% | 中位 %.4g | S/N 中位 %.1f | S/N<3 占 %.1f%%'
          % (100 * ok.mean(), np.nanmedian(sv[ok]), np.nanmedian(snr[ok]),
             100 * np.mean(snr[ok] < 3)))
    print('saved %s (%.2f MB)' % (args.out, os.path.getsize(args.out) / 1e6))


if __name__ == '__main__':
    main()
