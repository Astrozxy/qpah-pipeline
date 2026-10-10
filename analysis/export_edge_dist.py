#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
export_edge_dist.py — 为最终样本的每一行导出"到各观测图 footprint 边界的距离"。

这样就能检验：残差是否在靠近某张图边缘的像素上系统性变大（即"某图在边缘失效"）。
输出紧凑 npz（每行 4-5 个 float32）。
"""
import os, argparse
import numpy as np
import h5py
from astropy.io import fits
from astropy.wcs import WCS
from scipy import ndimage

MAPS = [('CO', 'CO'), ('HI', 'H1'), ('HI_ew', 'H1_ew'), ('SFR', 'sfr'), ('dust', 'dust_density')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', required=True)
    ap.add_argument('--data', default='data')
    ap.add_argument('--cache', default='sanity_cache')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    with h5py.File(args.h5) as f:
        ra = f['ra'][:].astype(np.float64)
        dec = f['dec'][:].astype(np.float64)
    hdul = fits.open(os.path.join(args.data, 'M31_S350_110_SSS_110_Model_All_qpah.fits'))
    w = WCS(hdul[0].header).celestial
    shape = hdul[0].data.shape
    qph = hdul[0].data.astype(float)
    hdul.close()
    x, y = w.world_to_pixel_values(ra, dec)
    ix = np.clip(np.round(x).astype(int), 0, shape[1] - 1)
    iy = np.clip(np.round(y).astype(int), 0, shape[0] - 1)

    # qPAH 图自身：用【有符号】距离
    #   >0 = 在 qPAH 有效区内，值为到边界的像素距离
    #   <0 = 在有效区【外】，值为到最近有效像素的距离
    # （注意：distance_transform_edt(mask) 对掩膜外像素返回 0，不能直接用）
    inside = qph > 0
    d_in = ndimage.distance_transform_edt(inside)
    d_out = ndimage.distance_transform_edt(~inside)
    qsd = np.where(inside, d_in, -d_out)
    out = {'edge_qpah_signed': qsd[iy, ix].astype(np.float32),
           'edge_qpah': d_in[iy, ix].astype(np.float32),
           'cov_qpah': inside[iy, ix].astype(np.float32)}
    v = out['edge_qpah_signed']
    print('  %-12s 有符号边缘距: 中位=%+6.1f | 在有效区外(<=0) 占 %5.2f%% | 0-5px 占 %5.2f%%'
          % ('qPAH', np.median(v), 100 * np.mean(v <= 0), 100 * np.mean((v > 0) & (v <= 5))))
    print('样本 %d 行' % len(ra))
    for k, lab in MAPS:
        cp = os.path.join(args.cache, k + '_ex_cov.npy')
        mp = os.path.join(args.cache, k + '_ex_mean.npy')
        if not (os.path.exists(cp) and os.path.exists(mp)):
            continue
        cov = np.load(cp); mn = np.load(mp)
        foot = np.isfinite(mn) & (cov > 0.5)
        ed = ndimage.distance_transform_edt(foot)
        out['edge_' + lab] = ed[iy, ix].astype(np.float32)
        out['cov_' + lab] = cov[iy, ix].astype(np.float32)
        v = out['edge_' + lab]
        print('  %-12s 边缘距: 中位=%6.1f p16=%6.1f | 落在 <20px 的占 %.1f%%'
              % (lab, np.median(v), np.percentile(v, 16), 100 * np.mean(v < 20)))
    np.savez_compressed(args.out, **out)
    print('saved %s (%.2f MB)' % (args.out, os.path.getsize(args.out) / 1e6))


if __name__ == '__main__':
    main()
