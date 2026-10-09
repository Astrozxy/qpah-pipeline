#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
export_umin_ubar.py — 在服务器上把 U_min / U_bar 图对齐到 qPAH 网格，
只导出【我们样本每行】的值（几 MB），避免把大 FITS 拉回本地。

自动在 data/ 里找 U_min / U_bar 的 FITS（按文件名含 u_min/umin/U_bar/ubar 等）。
对齐方式与 align_maps.py 完全一致：reproject_exact 面积加权平均，
coverage <= cov_min 的格置 NaN。

用法（服务器）:
  python export_umin_ubar.py --root . --h5 data/M31_1d_array_nodust.h5 \
      --out ~/umin_ubar_rows.npz
"""
import os, re, glob, argparse
import numpy as np
import h5py
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_exact

QPah = 'M31_S350_110_SSS_110_Model_All_qpah.fits'


def find_maps(datadir):
    out = {}
    for p in sorted(glob.glob(os.path.join(datadir, '*.fits'))):
        b = os.path.basename(p).lower()
        key = None
        if re.search(r'u[_-]?min', b):
            key = 'U_min'
        elif re.search(r'u[_-]?bar', b):
            key = 'U_bar'
        if key and key not in out:
            out[key] = p
    return out


def align_one(path, wout, shape, cov_min=0.5):
    hd = fits.open(path)
    hdu = hd[0]
    D0 = hdu.data.astype(np.float64)
    wcs_in = WCS(hdu.header).celestial
    finite = np.isfinite(D0)
    V = np.where(finite, D0, 0.0)
    num, _ = reproject_exact((V, wcs_in), wout, shape_out=shape)
    den, _ = reproject_exact((finite.astype(np.float64), wcs_in), wout, shape_out=shape)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = num / den
    mean[den <= cov_min] = np.nan
    hd.close()
    return mean, den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--h5', default='data/M31_1d_array_nodust.h5')
    ap.add_argument('--out', default='umin_ubar_rows.npz')
    ap.add_argument('--cov-min', type=float, default=0.5)
    args = ap.parse_args()
    D = os.path.join(args.root, 'data')

    print('=== data/ 里的 FITS 清单 ===')
    for p in sorted(glob.glob(os.path.join(D, '*.fits'))):
        try:
            h = fits.open(p)[0].header
            print('  %-58s %s  %s  %s' % (os.path.basename(p),
                  tuple(h.get('NAXIS1', 0) for _ in [0]) + (h.get('NAXIS2', 0),),
                  h.get('BUNIT', '-'), h.get('TELESCOP', '-')))
        except Exception as e:
            print('  %-58s <读取失败 %s>' % (os.path.basename(p), e))

    maps = find_maps(D)
    print()
    print('=== 自动识别的 U 图 ===')
    if not maps:
        print('  ❌ 没找到 U_min / U_bar'); return
    for k, v in maps.items():
        print('  %-6s %s' % (k, os.path.basename(v)))

    hp = os.path.join(D, QPah)
    hq = fits.open(hp)
    wout = WCS(hq[0].header).celestial
    shape = hq[0].data.shape
    hq.close()
    print('  目标网格: %s' % (shape,))

    with h5py.File(args.h5, 'r') as f:
        ra = f['ra'][:].astype(np.float64)
        dec = f['dec'][:].astype(np.float64)
    xx, yy = wout.world_to_pixel_values(ra, dec)
    ix = np.clip(np.round(xx).astype(int), 0, shape[1] - 1)
    iy = np.clip(np.round(yy).astype(int), 0, shape[0] - 1)
    print('  样本行数 %d（注意：这是 h5 的全部行，未做分析筛选）' % len(ra))

    out = {}
    for k, p in maps.items():
        mean, den = align_one(p, wout, shape, args.cov_min)
        vals = mean[iy, ix]
        cov = den[iy, ix]
        out[k] = vals
        out[k + '_cov'] = cov
        print('  %-6s: 有效 %d (%.1f%%) | 中位=%.4g | p16=%.4g p84=%.4g | cov 中位=%.3f'
              % (k, np.isfinite(vals).sum(), 100 * np.isfinite(vals).mean(),
                 np.nanmedian(vals), np.nanpercentile(vals, 16),
                 np.nanpercentile(vals, 84), np.median(cov)))
    np.savez_compressed(args.out, **out)
    print('saved %s (%.2f MB)' % (args.out, os.path.getsize(args.out) / 1e6))


if __name__ == '__main__':
    main()
