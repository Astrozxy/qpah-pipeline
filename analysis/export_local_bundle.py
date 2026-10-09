#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
export_local_bundle.py — 把本地复现所需的最小数据打包（在服务器上运行）。

为什么要它：服务器过于不稳定；但数据其实很小。
  - sanity_cache 的 14 个 1250×1250 float64 阵列共 168 MB，但**我们只用样本行的值**，
    压成紧凑 npz 后只有几 MB。
  - 其余（h5、qPAH FITS、悬臂 mask）本来就小。

产出 <out>/ 目录：
  data/M31_1d_array_nodust.h5, M31_1d_array_full.h5        (19 MB)
  data/M31_S350_110_..._qpah.fits + _unc.fits              (12 MB)
  masks/mask_arm_35_38_40_46_51.npz, arm_blob_labels.npz   (7 MB)
  cache/{CO,HI,HI_ew,SFR,dust}_mean_f32.npy                (31 MB, float32 供格点预测)
  cache/aux_sample.npz                                     (每行 cov/edge/空间项, ~5 MB)

用法（服务器）:
  python qpah_pipeline/analysis/export_local_bundle.py --out ~/qpah_bundle
"""
import os, shutil, argparse
import numpy as np
import h5py
from astropy.io import fits
from astropy.wcs import WCS
from scipy import ndimage

FEATS = ['H1', 'sfr', 'CO', 'H1_ew']
COV_KEY = {'H1': 'HI', 'sfr': 'SFR', 'CO': 'CO', 'H1_ew': 'HI_ew'}
CACHE_KEYS = ['CO', 'HI', 'HI_ew', 'SFR', 'dust']
RA0, DEC0, INC, PA = 10.6847, 41.2690, np.radians(77.7), np.radians(37.7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--out', default='qpah_bundle')
    args = ap.parse_args()
    R = args.root
    DATA = os.path.join(R, 'data')
    CACHE = os.path.join(R, 'sanity_cache')
    RES = os.path.join(R, 'qpah_pipeline', 'results', 'armregion')
    out = os.path.abspath(args.out)
    for sub in ['data', 'masks', 'cache']:
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    # ---- 1. 直接拷贝的小文件 ----
    for fn in ['M31_1d_array_nodust.h5', 'M31_1d_array_full.h5',
               'M31_S350_110_SSS_110_Model_All_qpah.fits',
               'M31_S350_110_SSS_110_Model_All_qpah_unc.fits']:
        src = os.path.join(DATA, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out, 'data', fn))
            print('copied data/%s (%.1f MB)' % (fn, os.path.getsize(src) / 1e6))
    for fn in ['mask_arm_35_38_40_46_51.npz', 'arm_blob_labels.npz']:
        src = os.path.join(RES, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out, 'masks', fn))
            print('copied masks/%s (%.1f MB)' % (fn, os.path.getsize(src) / 1e6))

    # ---- 2. sanity_cache 的 mean 转 float32（供格点预测）----
    for k in CACHE_KEYS:
        src = os.path.join(CACHE, k + '_ex_mean.npy')
        if os.path.exists(src):
            a = np.load(src).astype(np.float32)
            dst = os.path.join(out, 'cache', k + '_mean_f32.npy')
            np.save(dst, a)
            print('cache/%s_mean_f32.npy  shape=%s  %.1f MB'
                  % (k, a.shape, os.path.getsize(dst) / 1e6))

    # ---- 3. 紧凑 aux：每行的 coverage / 边缘距 / 空间项 ----
    with h5py.File(os.path.join(DATA, 'M31_1d_array_nodust.h5'), 'r') as f:
        d = {k: f[k][:] for k in ['ra', 'dec', 'qpah', 'qpah_err', 'CO_sigma'] + FEATS}
    sel = ((d['qpah'] >= 0) & (d['qpah_err'] > 0) & np.isfinite(d['H1']) & (d['H1'] >= 0)
           & np.isfinite(d['H1_ew']) & (d['H1_ew'] >= 0) & np.isfinite(d['sfr'])
           & np.isfinite(d['CO']) & np.isfinite(d['CO_sigma']) & (d['CO_sigma'] > 0))
    ra, dec = d['ra'][sel].astype(np.float64), d['dec'][sel].astype(np.float64)
    hq = fits.open(os.path.join(DATA, 'M31_S350_110_SSS_110_Model_All_qpah.fits'))
    w = WCS(hq[0].header).celestial
    hq.close()
    xx, yy = w.world_to_pixel_values(ra, dec)
    ix = np.round(xx).astype(int)
    iy = np.round(yy).astype(int)
    aux = {}
    for f_ in FEATS:
        k = COV_KEY[f_]
        cv = np.load(os.path.join(CACHE, k + '_ex_cov.npy'))
        mn = np.load(os.path.join(CACHE, k + '_ex_mean.npy'))
        foot = np.isfinite(mn) & (cv > 0.5)
        aux['cov_' + f_] = cv[iy, ix].astype(np.float32)
        aux['edge_' + f_] = ndimage.distance_transform_edt(foot)[iy, ix].astype(np.float32)
    dx = (ra - RA0) * np.cos(np.radians(DEC0)); dy = dec - DEC0
    X = dx * np.cos(PA) + dy * np.sin(PA)
    Y = (-dx * np.sin(PA) + dy * np.cos(PA)) / np.cos(INC)
    kk = 3600.0 / (206265.0 / 744.0)
    aux['R_kpc'] = (np.sqrt(X ** 2 + Y ** 2) * kk).astype(np.float32)
    aux['x_major_kpc'] = (X * kk).astype(np.float32)
    aux['y_minor_kpc'] = (Y * kk).astype(np.float32)
    aux['mask_arm'] = np.load(os.path.join(RES, 'mask_arm_35_38_40_46_51.npz'))['region']
    dst = os.path.join(out, 'cache', 'aux_sample.npz')
    np.savez_compressed(dst, **aux)
    print('cache/aux_sample.npz  n=%d  %.1f MB' % (len(ra), os.path.getsize(dst) / 1e6))
    print('总大小:')
    tot = 0
    for dp, _, fns in os.walk(out):
        for fn in fns:
            tot += os.path.getsize(os.path.join(dp, fn))
    print('  %.1f MB  ->  %s' % (tot / 1e6, out))


if __name__ == '__main__':
    main()
