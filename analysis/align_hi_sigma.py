#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
align_hi_sigma.py — 把 HI mom0 的误差图 (emom0) 对齐到 qPAH 网格，
并抽出管线样本每一行的 sigma_H1。

对齐方式与管线其它图完全一致：
    reproject_exact 面积加权平均，cov>0.5 之外置 NaN。

输出：
    <cache>/HI_sigma_ex_mean.npy     1250x1250 float32 对齐后的 sigma_H1
    <cache>/HI_sigma_ex_cov.npy      1250x1250 float32 覆盖率
    <out>  npz: {'H1_sigma': 每行 sigma_H1 (对应 --h5 的行), 'H1': 每行 mom0}
"""
import os, argparse, time
import numpy as np
import h5py
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_exact


def reproj(data, wcs_in, wout, shape):
    """面积加权平均 + 覆盖率；与 sanity_blockavg_v4.py 的做法一致。"""
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
    ap.add_argument('--h5', required=True, help='管线 h5（用它的 ra/dec 抽逐行值）')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    t0 = time.time()

    hq = fits.open(os.path.join(args.data, 'M31_S350_110_SSS_110_Model_All_qpah.fits'))
    wout = WCS(hq[0].header).celestial
    shape = hq[0].data.shape
    hq.close()

    hp = os.path.join(args.data, 'm31_C+D+tp_hilores_120pc_strict_emom0.fits')
    he = fits.open(hp)
    esig = he[0].data.astype(np.float64)
    wes = WCS(he[0].header).celestial
    he.close()
    print('emom0: shape %s | 有限 %.1f%% | 中位 %.3f K km/s'
          % (esig.shape, 100 * np.isfinite(esig).mean(), np.nanmedian(esig)))

    cp = os.path.join(args.cache, 'HI_sigma_ex_mean.npy')
    cc = os.path.join(args.cache, 'HI_sigma_ex_cov.npy')
    if os.path.exists(cp) and os.path.exists(cc):
        mean = np.load(cp); cov = np.load(cc)
        print('复用已有对齐结果')
    else:
        mean, cov = reproj(esig, wes, wout, shape)
        print('对齐完成 %.0fs' % (time.time() - t0))
        np.save(cp, mean.astype(np.float32))
        np.save(cc, cov.astype(np.float32))
    ok = np.isfinite(mean)
    print('对齐后: 覆盖 %.1f%% | sigma 中位 %.3f (p1 %.3f p99 %.3f)'
          % (100 * ok.mean(), np.nanmedian(mean), np.nanpercentile(mean[ok], 1),
             np.nanpercentile(mean[ok], 99)))

    # 抽逐行值
    with h5py.File(args.h5) as f:
        ra = f['ra'][:].astype(np.float64)
        dec = f['dec'][:].astype(np.float64)
        h1 = f['H1'][:].astype(np.float32) if 'H1' in f else np.full(len(ra), np.nan, np.float32)
    x, y = wout.world_to_pixel_values(ra, dec)
    ix = np.clip(np.round(x).astype(int), 0, shape[1] - 1)
    iy = np.clip(np.round(y).astype(int), 0, shape[0] - 1)
    sig_row = mean[iy, ix].astype(np.float32)
    cov_row = cov[iy, ix].astype(np.float32)
    print('逐行: n=%d | sigma 有限 %.1f%% | 中位 %.3f | H1/sigma 中位 %.1f'
          % (len(sig_row), 100 * np.isfinite(sig_row).mean(), np.nanmedian(sig_row),
             np.nanmedian(h1 / sig_row)))
    np.savez_compressed(args.out, H1_sigma=sig_row, H1_sigma_cov=cov_row, H1=h1)
    print('saved %s (%.2f MB)' % (args.out, os.path.getsize(args.out) / 1e6))


if __name__ == '__main__':
    main()
