#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
render_pred_maps.py — 把 predict_grid.py 输出的全网格预测 / 残差 / χ FITS 渲染成 PNG。

用法:
    python render_pred_maps.py --tag coalt_full [--resid-vmax 3]
读取:
    data/qpah_pred_map_<tag>.fits, data/qpah_pred_resid_map_<tag>.fits,
    data/qpah_chi_map_<tag>.fits
输出:
    results/<tag>/fig_pred_map_<tag>.png
    results/<tag>/fig_resid_map_<tag>.png     (绝对残差 pred−obs，qPAH 百分点)
    results/<tag>/fig_chi_map_<tag>.png        (χ = (pred−obs)/σ_eff)

注：χ = (pred − obs)/σ_eff（σ_eff=sqrt(qpah_err²+floor²)），用于判断拟合优度
（观测误差逐像素异方差）；绝对残差 pred−obs 物理尺度直观、能看出"哪里错得最多"，
但幅度被高 σ_eff 像素主导，不能当拟合判据。两张图互补，都出。
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
    ap.add_argument('--chi-vmin', type=float, default=-3.0)
    ap.add_argument('--chi-vmax', type=float, default=3.0)
    ap.add_argument('--resid-vmax', type=float, default=3.0,
                    help='绝对残差图对称色标上限（±vmax，qPAH 百分点）')
    ap.add_argument('--skip-resid', action='store_true',
                    help='不渲染绝对残差图（默认渲染）')
    args = ap.parse_args()

    outdir = os.path.join(RESULTS, args.tag)
    os.makedirs(outdir, exist_ok=True)
    pred_fits = os.path.join(DATA, 'qpah_pred_map_%s.fits' % args.tag)
    chi_fits = os.path.join(DATA, 'qpah_chi_map_%s.fits' % args.tag)
    for f in (pred_fits, chi_fits):
        if not os.path.exists(f):
            raise SystemExit('缺文件: %s（先跑 predict_grid.py --tag %s）' % (f, args.tag))

    render(pred_fits, os.path.join(outdir, 'fig_pred_map_%s.png' % args.tag),
           'Predicted qPAH (%%)  [%s, full grid]' % args.tag, 'viridis',
           args.pred_vmin, args.pred_vmax)
    render(chi_fits, os.path.join(outdir, 'fig_chi_map_%s.png' % args.tag),
           r'$\chi=(\mathrm{pred}-\mathrm{obs})/\sigma_{\rm eff}$  [%s]' % args.tag,
           'RdBu_r', args.chi_vmin, args.chi_vmax,
           cbar_label=r'$\chi$   ($\sigma_{\rm eff}$)')

    resid_fits = os.path.join(DATA, 'qpah_pred_resid_map_%s.fits' % args.tag)
    if args.skip_resid:
        print('skip resid map (--skip-resid)')
    elif not os.path.exists(resid_fits):
        print('skip resid map: 缺 %s（先跑 predict_grid.py --tag %s）'
              % (resid_fits, args.tag))
    else:
        render(resid_fits, os.path.join(outdir, 'fig_resid_map_%s.png' % args.tag),
               r'$\mathrm{pred}-\mathrm{obs}$  ($q_{\rm PAH}$ %%)  [%s]' % args.tag,
               'RdBu_r', -args.resid_vmax, args.resid_vmax,
               cbar_label=r'$\mathrm{pred}-\mathrm{obs}$   (%)')
    print('ALL DONE')


if __name__ == '__main__':
    main()
