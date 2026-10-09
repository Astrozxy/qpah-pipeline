#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
extract_draine_maps.py — 从 Draine 网站下载的 S350 分辨率图上，取出我们样本每行的值。

关键：这些图与我们用的 qPAH 图是【同一个网格】（CRPIX/CRVAL/CD 完全相同，
world_to_pixel 差为 0），所以**不需要 reproject**，直接按像素取即可。

处理：
  - 0 值视为缺失（与原图 qPAH 的填充约定一致），转成 NaN
  - 同时报告覆盖率

用法:
  python extract_draine_maps.py --h5 .../M31_1d_array_nodust.h5 \
      --maps-dir .../local_data/data \
      --out .../local_data/cache/draine_rows.npz
"""
import os, glob, argparse
import numpy as np
import h5py
from astropy.io import fits
from astropy.wcs import WCS

WANT = {
    'U_min': 'U_min', 'U_bar': 'U_bar',
    'SurfBr_Mdust': 'Mdust', 'SurfBr_Ldust': 'Ldust',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', required=True)
    ap.add_argument('--maps-dir', required=True)
    ap.add_argument('--ref', default='M31_S350_110_SSS_110_Model_All_qpah.fits')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    with h5py.File(args.h5) as f:
        ra = f['ra'][:].astype(np.float64)
        dec = f['dec'][:].astype(np.float64)
    hr = fits.open(os.path.join(args.maps_dir, args.ref))[0]
    w = WCS(hr.header).celestial
    shape = hr.data.shape
    x, y = w.world_to_pixel_values(ra, dec)
    ix = np.clip(np.round(x).astype(int), 0, shape[1] - 1)
    iy = np.clip(np.round(y).astype(int), 0, shape[0] - 1)
    print('样本行数 %d | 目标网格 %s' % (len(ra), shape))

    out = {}
    for p in sorted(glob.glob(os.path.join(args.maps_dir, 'M31_S350_110_SSS_110_Model_All_*.fits'))):
        b = os.path.basename(p)
        stem = b.replace('M31_S350_110_SSS_110_Model_All_', '').replace('.fits', '')
        key = WANT.get(stem)
        if key is None or key in out:
            continue
        hd = fits.open(p)[0]
        wm = WCS(hd.header).celestial
        # 网格一致性自检
        xm, ym = wm.world_to_pixel_values(ra[:200], dec[:200])
        ok = np.allclose(xm, x[:200]) and np.allclose(ym, y[:200])
        d = hd.data.astype(np.float64)
        v = d[iy, ix].copy()
        nz = (v != 0)
        v[~nz] = np.nan
        out[key] = v.astype(np.float32)
        print('  %-8s (%-24s) 网格一致=%s | 有效 %6d (%.1f%%) | 中位=%.4g p5=%.4g p95=%.4g'
              % (key, stem, ok, np.isfinite(v).sum(), 100 * np.isfinite(v).mean(),
                 np.nanmedian(v), np.nanpercentile(v, 5), np.nanpercentile(v, 95)))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **out)
    print('saved %s (%.2f MB)' % (args.out, os.path.getsize(args.out) / 1e6))


if __name__ == '__main__':
    main()
