#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_agb_map.py — 从 PHAT/PHAT-S 星表生成 qPAH 网格上的 AGB 计数图并缓存。

输出：sanity_cache/AGB_count_map.npy （(1250,1250) float；像素值 = qPAH 像素内 AGB 星数）
AGB 定义与 M31_plot.ipynb 一致（CMD 选区 + F814W 限幅）。
幂等：缓存存在则跳过（--rebuild 强制重算）。
"""
import os, time, argparse
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

REPO = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(REPO)
DATA = os.path.join(ROOT, 'data')
CACHE = os.path.join(ROOT, 'sanity_cache')
AGB = os.path.join(ROOT, 'AGB')
DEFAULT_OUT = os.path.join(CACHE, 'AGB_count_map.npy')
T0 = time.time()


def log(*a):
    print('[%.0fs]' % (time.time() - T0), *a, flush=True)


def compute_agb_count_map(qpah_fits=None, out=None, save=True):
    """与 M31_plot.ipynb cell 2-18 相同的 AGB 选区 -> qPAH 网格计数图"""
    from astropy.table import Table
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    qpah_fits = qpah_fits or os.path.join(DATA, 'M31_S350_110_SSS_110_Model_All_qpah.fits')
    hd = fits.open(qpah_fits)
    header = hd[0].header
    shape = hd[0].data.shape
    hd.close()
    ny, nx = shape

    log('reading PHAT / PHAT-South catalogs...')
    t_phast = Table.read(os.path.join(AGB, 'hlsp_phast_hst_acs-wfc3_m31-south-all_multi_v1.0_cat.fits'))
    t_phat = Table.read(os.path.join(AGB, 'hlsp_phat_hst_wfc3-uvis-acs-wfc-wfc3-ir_f275w-f336w-f475w-f814w-f110w-f160w_v3_phot.fits'))

    i1 = (t_phat['F814W_ST_FLAG'] == 1) & (t_phat['F475W_ST_FLAG'] == 1)
    f475p = np.array(t_phat['F475W_VEGA'][i1]); f814p = np.array(t_phat['F814W_VEGA'][i1])
    rap = np.array(t_phat['RA'][i1]); decp = np.array(t_phat['DEC'][i1])
    i2 = t_phast['f475w_st'] & t_phast['f814w_st'] & (~t_phast['in_phat'])
    f475s = np.array(t_phast['f475w_vega'][i2]); f814s = np.array(t_phast['f814w_vega'][i2])
    ras = np.array(t_phast['ra'][i2]); decs = np.array(t_phast['dec'][i2])

    f475 = np.hstack([f475p, f475s]); f814 = np.hstack([f814p, f814s])
    ra = np.hstack([rap, ras]); dec = np.hstack([decp, decs])
    color = f475 - f814
    slope = (22.6 - 20.4) / (7.0 - 2.7)
    intercept = 22.6 - slope * 7.0
    idx_AGB = ((color > 2.7) & (f814 < 22.6) & (f814 < slope * color + intercept) & (f814 > 18))
    log('AGB stars:', int(idx_AGB.sum()))

    sc = SkyCoord(ra=ra[idx_AGB] * u.deg, dec=dec[idx_AGB] * u.deg, frame='icrs')
    wcs2d = WCS(header)
    x, y = wcs2d.world_to_pixel(sc)
    mask = (x >= 0) & (x < nx) & (y >= 0) & (y < ny) & np.isfinite(color[idx_AGB])
    H, _, _ = np.histogram2d(x[mask], y[mask], bins=[nx, ny], range=[[0, nx], [0, ny]],
                             weights=np.ones(int(mask.sum())))
    counts = H.T.astype(np.float64)
    log('AGB count map: shape=%s, px>=3: %d, total stars binned: %d'
        % (counts.shape, int((counts > 3).sum()), int(counts.sum())))
    if save:
        out = out or DEFAULT_OUT
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.save(out, counts)
        log('saved', out)
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--rebuild', action='store_true')
    args = ap.parse_args()
    if os.path.exists(args.out) and not args.rebuild:
        log('AGB map 已存在，跳过（--rebuild 强制重算）:', args.out)
        return
    compute_agb_count_map(out=args.out)
    log('ALL DONE')


if __name__ == '__main__':
    main()
