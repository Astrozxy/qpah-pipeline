#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
render_pred_maps.py — 把 predict_grid.py 输出的全网格预测/残差 FITS 渲染成 PNG。

用法:
    python render_pred_maps.py --tag coalt_full
读取:
    data/qpah_pred_map_<tag>.fits, data/qpah_pred_resid_map_<tag>.fits
输出:
    results/<tag>/fig_pred_map_<tag>.png
    results/<tag>/fig_resid_map_<tag>.png
"""
import os, argparse
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mplconfig')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS

REPO = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(REPO)
DATA = os.path.join(ROOT, 'data')
RESULTS = os.path.join(REPO, 'results')


def render(fits_path, out_png, title, cmap, vmin, vmax, cbar_label='qPAH (%)'):
    hdul = fits.open(fits_path)
    data = hdul[0].data.astype(np.float64)
    hdr = hdul[0].header.copy()
    hdul.close()
    w = WCS(hdr)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection=w)
    im = ax.imshow(data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    lon, lat = ax.coords[0], ax.coords[1]
    lon.set_axislabel('RA (deg)')
    lat.set_axislabel('Dec (deg)')
    lon.set_major_formatter('d.dd')
    lat.set_major_formatter('d.dd')
    lon.set_ticks(number=6)
    lat.set_ticks(number=6)
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label)
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('saved', out_png, '| finite=%d | min=%.3f max=%.3f'
          % (np.isfinite(data).sum(), np.nanmin(data), np.nanmax(data)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='coalt_full')
    ap.add_argument('--pred-vmin', type=float, default=0.0)
    ap.add_argument('--pred-vmax', type=float, default=6.0)
    ap.add_argument('--resid-vmin', type=float, default=-5.0)
    ap.add_argument('--resid-vmax', type=float, default=5.0)
    args = ap.parse_args()

    outdir = os.path.join(RESULTS, args.tag)
    os.makedirs(outdir, exist_ok=True)
    pred_fits = os.path.join(DATA, 'qpah_pred_map_%s.fits' % args.tag)
    resid_fits = os.path.join(DATA, 'qpah_pred_resid_map_%s.fits' % args.tag)
    for f in (pred_fits, resid_fits):
        if not os.path.exists(f):
            raise SystemExit('缺文件: %s（先跑 predict_grid.py --tag %s）' % (f, args.tag))

    render(pred_fits, os.path.join(outdir, 'fig_pred_map_%s.png' % args.tag),
           'Predicted qPAH (%%)  [%s, full grid]' % args.tag, 'viridis',
           args.pred_vmin, args.pred_vmax)
    render(resid_fits, os.path.join(outdir, 'fig_resid_map_%s.png' % args.tag),
           'Residual (pred - obs)  [%s]' % args.tag, 'RdBu_r',
           args.resid_vmin, args.resid_vmax, cbar_label='pred - obs (%)')
    print('ALL DONE')


if __name__ == '__main__':
    main()
