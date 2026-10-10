#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
edge_diagnostic.py — 检查"左上角高残差区域"是否是某张观测图在边缘失效造成的。

对每张参与模型的图报告（在区域内的）：
  coverage 中位、coverage<1 的比例、到 footprint 边界的距离、值的中位，
  以及该区域周围更大范围的值中位（判断是否局部异常）。
并输出一张各图的放大对照图。
"""
import os, argparse
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MAPS = [('CO', 'CO'), ('HI', 'H1'), ('HI_ew', 'H1_ew'), ('SFR', 'sfr')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--cache', default='sanity_cache')
    ap.add_argument('--box', nargs=4, type=float,
                    default=[11.285, 11.561, 41.857, 42.035],
                    help='RA0 RA1 Dec0 Dec1')
    ap.add_argument('--out', default='edge_zoom.png')
    args = ap.parse_args()
    ra0, ra1, de0, de1 = args.box
    hdul = fits.open(os.path.join(args.data, 'M31_S350_110_SSS_110_Model_All_qpah.fits'))
    h = hdul[0]
    w = WCS(h.header).celestial
    shape = h.data.shape
    qph = h.data.astype(float)
    hdul.close()
    xs, ys = w.world_to_pixel_values([ra0, ra0, ra1, ra1], [de0, de1, de0, de1])
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    print('区域像素 x %d-%d  y %d-%d' % (x0, x1, y0, y1))

    qfoot = qph > 0
    qed = ndimage.distance_transform_edt(qfoot)
    sub = qed[y0:y1 + 1, x0:x1 + 1][qfoot[y0:y1 + 1, x0:x1 + 1]]
    print('qPAH 图自身: 区域内有效 %d/%d | 到 qPAH 有效边界距离 中位=%.1f px（全局 %.1f）'
          % (qfoot[y0:y1 + 1, x0:x1 + 1].sum(), (y1 - y0 + 1) * (x1 - x0 + 1),
             np.median(sub) if sub.size else np.nan, np.median(qed[qfoot])))
    print()
    print('%-7s %9s %8s %11s %13s %13s' %
          ('map', 'cov中位', 'cov<1', '边缘距中位', '值中位(区域)', '值中位(邻域)'))
    pad = 80
    for k, lab in MAPS:
        cp = os.path.join(args.cache, k + '_ex_cov.npy')
        mp = os.path.join(args.cache, k + '_ex_mean.npy')
        if not (os.path.exists(cp) and os.path.exists(mp)):
            print('%-7s  <缺文件>' % lab); continue
        cov = np.load(cp); mn = np.load(mp).astype(float)
        foot = np.isfinite(mn) & (cov > 0.5)
        ed = ndimage.distance_transform_edt(foot)
        m = qfoot[y0:y1 + 1, x0:x1 + 1]
        c = cov[y0:y1 + 1, x0:x1 + 1][m]; e = ed[y0:y1 + 1, x0:x1 + 1][m]
        v = mn[y0:y1 + 1, x0:x1 + 1][m]
        nb = mn[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
        print('%-7s %9.3f %7.1f%% %11.1f %13.4g %13.4g'
              % (lab, np.median(c), 100 * np.mean(c < 1), np.median(e),
                 np.nanmedian(v), np.nanmedian(nb)))

    padx = 90
    X0, X1 = max(0, x0 - padx), min(shape[1], x1 + padx)
    Y0, Y1 = max(0, y0 - padx), min(shape[0], y1 + padx)
    items = [('qPAH (target)', qph)]
    for k, lab in MAPS:
        p = os.path.join(args.cache, k + '_ex_mean.npy')
        if os.path.exists(p):
            items.append((lab, np.load(p).astype(float)))
    fig, ax = plt.subplots(1, len(items), figsize=(5 * len(items), 5.6))
    for a, (lab, arr) in zip(np.atleast_1d(ax), items):
        s = arr[Y0:Y1, X0:X1]
        g = s[np.isfinite(s)]
        lo, hi = (np.percentile(g, [2, 98]) if g.size else (0, 1))
        im = a.imshow(s, origin='lower', cmap='inferno', vmin=lo, vmax=hi)
        a.add_patch(plt.Rectangle((x0 - X0, y0 - Y0), x1 - x0, y1 - y0,
                                  fill=False, ec='lime', lw=1.6))
        a.set_title(lab, fontsize=12)
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
    fig.suptitle('zoom on the upper-left high-residual region (green box)', fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches='tight')
    print('\nsaved', args.out)


if __name__ == '__main__':
    main()
