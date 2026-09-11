#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_dataset.py — 构建 qPAH 训练数据集（像素中心坐标，特征=exact block 平均）。

用法:
    python build_dataset.py --selection full     # 全盘：coverage 齐全的像素（推荐主数据集）
    python build_dataset.py --selection agb      # AGB>=3 且剔除 M32 的子样本
    可选 --rebuild 强制重建（默认文件存在则跳过）

输出:
    full -> data/M31_1d_array_full.h5
    agb  -> data/M31_1d_array_agb.h5
键（两种选择相同，最小化公开接口）:
    ra dec qpah qpah_err sfr CO H1 H1_ew dust_density
说明:
    qpah/qpah_err 已 *100 转百分数；五特征只在 coverage>0.5 时视为有效；
    行按像素中心坐标；具体质量筛选(qpah>0 等)在训练脚本里统一做。
"""
import os, time, argparse
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area
import h5py

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
CACHE = os.path.join(ROOT, 'sanity_cache')
AGB = os.path.join(ROOT, 'AGB')

T0 = time.time()
def log(*a):
    print('[%.0fs]' % (time.time() - T0), *a, flush=True)


def load_aligned():
    """exact block 平均 + coverage（由 align_maps.py 生成）"""
    feats = {'CO': 'CO', 'HI': 'H1', 'HI_ew': 'H1_ew', 'SFR': 'sfr', 'dust': 'dust_density'}
    vals, covs = {}, {}
    for key in feats:
        vals[key] = np.load(os.path.join(CACHE, key + '_ex_mean.npy'))
        covs[key] = np.load(os.path.join(CACHE, key + '_ex_cov.npy'))
    return feats, vals, covs


def agb_mask(shape, wcs_cel):
    """AGB>=3/像素（用缓存的计数图；缺失时调用 make_agb_map.py 生成一次）"""
    fp = os.path.join(CACHE, 'AGB_count_map.npy')
    if not os.path.exists(fp):
        from make_agb_map import compute_agb_count_map
        compute_agb_count_map()
    counts = np.load(fp)
    if counts.shape != tuple(shape):
        raise RuntimeError('AGB count map shape %s != qPAH shape %s' % (counts.shape, shape))
    return counts > 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selection', choices=['full', 'agb'], default='full')
    ap.add_argument('--rebuild', action='store_true')
    args = ap.parse_args()

    out_name = 'M31_1d_array_full.h5' if args.selection == 'full' else 'M31_1d_array_agb.h5'
    out_path = os.path.join(DATA, out_name)
    if os.path.exists(out_path) and not args.rebuild:
        log('已存在，跳过（--rebuild 可重建）:', out_path)
        return

    hq = fits.open(os.path.join(DATA, 'M31_S350_110_SSS_110_Model_All_qpah.fits'))
    wcel = WCS(hq[0].header).celestial
    shape = hq[0].data.shape
    qpah = hq[0].data.astype(np.float64) * 100.0
    hq.close()
    hqe = fits.open(os.path.join(DATA, 'M31_S350_110_SSS_110_Model_All_qpah_unc.fits'))
    qpah_err = hqe[0].data.astype(np.float64) * 100.0
    hqe.close()
    ny, nx = shape

    feats, vals, covs = load_aligned()

    # CO 附属量（align_maps.py 生成；缺失时跳过）
    extra = {}
    for nm in ['CO_sigma', 'CO_det']:
        fp = os.path.join(CACHE, nm + '_ex_mean.npy')
        if os.path.exists(fp):
            extra[nm] = np.load(fp)
            log('loaded extra aligned column:', nm)

    yy, xx = np.mgrid[0:ny, 0:nx]
    ra_g, dec_g = wcel.pixel_to_world_values(xx.astype(np.float64), yy.astype(np.float64))
    idx_M32 = ((ra_g - 10.67) ** 2 + (dec_g - 40.86) ** 2) < (2.5 / 60.) ** 2

    ok = (~idx_M32) & np.isfinite(qpah) & np.isfinite(qpah_err)
    for key in feats:
        ok &= np.isfinite(vals[key])
    if args.selection == 'agb':
        ok &= agb_mask(shape, wcel)
    ry, rx = np.where(ok)

    out = dict(ra=ra_g[ry, rx], dec=dec_g[ry, rx],
               qpah=qpah[ry, rx], qpah_err=qpah_err[ry, rx])
    for key, col in feats.items():
        out[col] = vals[key][ry, rx]
    for nm in extra:
        out[nm] = extra[nm][ry, rx]

    with h5py.File(out_path, 'w') as f:
        for k, v in out.items():
            f[k] = v
        f.attrs['selection'] = args.selection
        f.attrs['note'] = ('pixel-center; features = reproject_exact block avg (cov>0.5); '
                           'qpah/qpah_err in percent; CO_sigma/CO_det from make_co_maps+align '
                           '(native S/N>3 探测占比)')
    log('saved', out_path, 'rows:', len(out['ra']))
    log('ALL DONE')


if __name__ == '__main__':
    main()
