#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
align_dust_combined.py — 用 m31_combined850_um.fits（4"，正确 NaN 掩膜）
重新对齐尘埃图到 qPAH 网格，并与旧的 dust_density（来自零填充的 sm24 版）对比。

输出：
    <cache>/dustc_ex_mean.npy  / dustc_ex_cov.npy
    <out> 逐行 npz（对齐到 --h5 的行）
"""
import os, argparse, time
import numpy as np
import h5py
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_exact


def reproj(data, wcs_in, wout, shape):
    V = np.isfinite(data)
    D0 = np.where(V, data, 0.0).astype(np.float64)
    Vf = V.astype(np.float64)
    num, _ = reproject_exact((D0, wcs_in), wout, shape_out=shape)
    den, _ = reproject_exact((Vf, wcs_in), wout, shape_out=shape)
    mean = np.full(shape, np.nan)
    ok = (den > 0.5) & np.isfinite(num) & np.isfinite(den) & (den > 0)
    mean[ok] = num[ok] / den[ok]
    return mean, den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--cache', default='sanity_cache')
    ap.add_argument('--src', default='m31_combined850_um.fits')
    ap.add_argument('--tag', default='dustc')
    ap.add_argument('--h5', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    hq = fits.open(os.path.join(args.data, 'M31_S350_110_SSS_110_Model_All_qpah.fits'))[0]
    wout = WCS(hq.header).celestial
    shape = hq.data.shape
    qpah = hq.data.astype(float)

    hd = fits.open(os.path.join(args.data, args.src))
    dat = hd[0].data.astype(np.float64)
    wd = WCS(hd[0].header).celestial
    print('%s: shape %s | finite %.1f%% | >0/finite %.1f%% | BUNIT %s'
          % (args.src, dat.shape, 100 * np.isfinite(dat).mean(),
             100 * np.mean(dat[np.isfinite(dat)] > 0), hd[0].header.get('BUNIT', '-')))
    hd.close()

    cp = os.path.join(args.cache, args.tag + '_ex_mean.npy')
    cc = os.path.join(args.cache, args.tag + '_ex_cov.npy')
    if os.path.exists(cp) and os.path.exists(cc):
        mean, cov = np.load(cp).astype(float), np.load(cc).astype(float)
        print('复用已有对齐')
    else:
        t = time.time()
        mean, cov = reproj(dat, wd, wout, shape)
        print('对齐 %.0fs' % (time.time() - t))
        np.save(cp, mean.astype(np.float32)); np.save(cc, cov.astype(np.float32))
    print('对齐后: cov>0.5 占 %.2f%% | 有效值中位 %.4g'
          % (100 * np.mean(cov > 0.5), np.nanmedian(mean)))

    # 与旧的（零填充 sm24）对比
    old = os.path.join(args.cache, 'dust_ex_mean.npy')
    if os.path.exists(old):
        o = np.load(old).astype(float)
        both = (cov > 0.5) & np.isfinite(o)
        print('与旧 dust_density 在公共像素上: n=%d | 相关 %.3f | 中位比 %.3f'
              % (both.sum(), np.corrcoef(mean[both], o[both])[0, 1],
                 np.median(mean[both]) / np.median(o[both])))
        # 各自与 qPAH 的相关
        for nm, a in [('旧 sm24(零填充)', o), ('新 combined', mean)]:
            m = (cov > 0.5) & (a > 0) & (qpah > 0)
            print('   %s: n=%d | Pearson(log a, qPAH)=%+.3f'
                  % (nm, m.sum(), np.corrcoef(np.log10(a[m]), qpah[m])[0, 1]))

    with h5py.File(args.h5) as f:
        ra = f['ra'][:].astype(np.float64); dec = f['dec'][:].astype(np.float64)
        old_row = f['dust_density'][:].astype(np.float32) if 'dust_density' in f else None
    x, y = wout.world_to_pixel_values(ra, dec)
    ix = np.clip(np.round(x).astype(int), 0, shape[1] - 1)
    iy = np.clip(np.round(y).astype(int), 0, shape[0] - 1)
    row = mean[iy, ix].astype(np.float32)
    crow = cov[iy, ix].astype(np.float32)
    print('逐行: n=%d | 有效 %.1f%% | 中位 %.4g'
          % (len(row), 100 * np.isfinite(row).mean(), np.nanmedian(row)))
    out = {'dust_combined': row, 'dust_combined_cov': crow}
    if old_row is not None:
        out['dust_density_old'] = old_row
    np.savez_compressed(args.out, **out)
    print('saved %s (%.2f MB)' % (args.out, os.path.getsize(args.out) / 1e6))


if __name__ == '__main__':
    main()
