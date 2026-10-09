#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
arm_region_coverage_check.py — 检查"高误差悬臂"是不是踩在某张输入图的【边缘】上。

动机
----
前一步圈出的高误差区域（48 格 / 1794 px / χ²=4.35）在参数空间里完全不突出，
说明不是外推问题。用户指出：它位于天区边缘，**有可能是某一张输入 map 的
边界伪影**（coverage 只有 0.5~0.8、面积加权平均在边界处被稀释/截断）。

本脚本回答：
  1. 该区域在每张输入图上的 coverage 是多少？和其余像素比是否明显偏低？
  2. 它离每张图的 footprint 边界有多近（到最近 cov<0.5 像素的距离）？
  3. 把 5 张对齐后的图放大到该区域，肉眼查有没有条纹/空洞等伪影。

输出:
  fig_arm_coverage.png     qPAH 网格定位 + 5 张图的 coverage（框出区域）
  fig_arm_zoom_maps.png    放大到区域：qPAH / CO / HI / HI_ew / SFR / dust 对齐图 + χ
"""
import os, argparse
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mplconfig')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from astropy.io import fits
from astropy.wcs import WCS

MAPS = ['CO', 'HI', 'HI_ew', 'SFR', 'dust']
TITLE = {'CO': 'CO (23")', 'HI': 'HI mom0 (32")', 'HI_ew': 'HI EW (32")',
         'SFR': 'SFR (1.5")', 'dust': '850um SCUBA-2 (25")'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mask-npz',
                    default='qpah_pipeline/results/armregion/arm_region_mask.npz')
    ap.add_argument('--data', default='data')
    ap.add_argument('--cache', default='sanity_cache')
    ap.add_argument('--outdir', default='qpah_pipeline/results/armregion')
    ap.add_argument('--pad', type=int, default=60, help='放大图的边距（像素）')
    args = ap.parse_args()

    z = np.load(args.mask_npz)
    ra, dec, region, chi = z['ra'], z['dec'], z['region'], z['chi']

    hq = fits.open(os.path.join(args.data, 'M31_S350_110_SSS_110_Model_All_qpah.fits'))
    w = WCS(hq[0].header).celestial
    qpah = hq[0].data.astype(float) * 100.0
    shape = qpah.shape
    hq.close()
    xx, yy = w.world_to_pixel_values(ra, dec)
    ix = np.clip(np.round(xx).astype(int), 0, shape[1] - 1)
    iy = np.clip(np.round(yy).astype(int), 0, shape[0] - 1)

    # 区域包围盒（放大用）
    x0, x1 = ix[region].min() - args.pad, ix[region].max() + args.pad
    y0, y1 = iy[region].min() - args.pad, iy[region].max() + args.pad
    x0, x1 = max(0, x0), min(shape[1], x1)
    y0, y1 = max(0, y0), min(shape[0], y1)

    covs, means = {}, {}
    for m in MAPS:
        p = os.path.join(args.cache, m + '_ex_cov.npy')
        q = os.path.join(args.cache, m + '_ex_mean.npy')
        if os.path.exists(p):
            covs[m] = np.load(p)
        if os.path.exists(q):
            means[m] = np.load(q)

    # ---------------- 统计 ----------------
    print('区域像素 %d / 全样本 %d' % (region.sum(), len(region)))
    print('%-6s %14s %14s %14s %14s %12s' %
          ('map', 'cov med(reg)', 'cov med(rest)', 'cov<0.9(reg)', '边缘距中位(reg)', '边缘距(rest)'))
    try:
        from scipy import ndimage
        have_scipy = True
    except ImportError:
        have_scipy = False

    edge_dist = {}
    for m, cv in covs.items():
        foot = np.isfinite(means.get(m, np.where(cv > 0.5, 1.0, np.nan))) & (cv > 0.5)
        if have_scipy:
            edge_dist[m] = ndimage.distance_transform_edt(foot)
        cr = cv[iy[region], ix[region]]
        co = cv[iy[~region], ix[~region]]
        if m in edge_dist:
            er = edge_dist[m][iy[region], ix[region]]
            eo = edge_dist[m][iy[~region], ix[~region]]
            es = '%12.1f %12.1f' % (np.median(er), np.median(eo))
        else:
            es = '%12s %12s' % ('-', '-')
        print('%-6s %14.3f %14.3f %13.1f%% %s'
              % (m, np.median(cr), np.median(co), 100 * np.mean(cr < 0.9), es))

    # ---------------- 图 1：coverage ----------------
    nmap = len(covs)
    fig, ax = plt.subplots(2, 3, figsize=(19, 12))
    ax = ax.ravel()
    ra_c, dec_c = np.median(ra[region]), np.median(dec[region])
    rw = (ra[region].max() - ra[region].min())
    rh = (dec[region].max() - dec[region].min())
    box = Rectangle((ra[region].min(), dec[region].min()), rw, rh,
                    facecolor='none', edgecolor='lime', lw=2)

    a = ax[0]
    a.imshow(qpah, origin='lower', cmap='inferno', vmin=0, vmax=6)
    a.set_title('qPAH map + region box'); a.set_xlabel('pixel x'); a.set_ylabel('pixel y')
    a.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor='none',
                          edgecolor='cyan', lw=2, ls='--'))
    a.scatter(ix[region], iy[region], s=0.5, c='lime')

    for a, m in zip(ax[1:], MAPS):
        if m not in covs:
            a.axis('off'); continue
        im = a.imshow(covs[m], origin='lower', cmap='viridis', vmin=0, vmax=1)
        a.scatter(ix[region], iy[region], s=0.3, c='red', alpha=0.35)
        a.set_title('%s  coverage (red = selected region)' % TITLE.get(m, m), fontsize=11)
        a.set_xlabel('pixel x')
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
    fig.tight_layout()
    f1 = os.path.join(args.outdir, 'fig_arm_coverage.png')
    fig.savefig(f1, dpi=150, bbox_inches='tight'); plt.close(fig)
    print('saved', f1)

    # ---------------- 图 2：放大看每张图 ----------------
    show = [('qPAH', qpah)] + [(TITLE.get(m, m), means[m]) for m in MAPS if m in means]
    fig, ax = plt.subplots(2, 3, figsize=(19, 12))
    ax = ax.ravel()
    for a, (nm, arr) in zip(ax, show):
        sub = np.array(arr[y0:y1, x0:x1], dtype=float)
        good = sub[np.isfinite(sub)]
        if good.size == 0:
            a.axis('off'); continue
        lo, hi = np.percentile(good, [2, 98])
        im = a.imshow(sub, origin='lower', cmap='inferno', vmin=lo, vmax=hi)
        # 区域轮廓（局部坐标）
        mm = np.zeros(sub.shape, bool)
        sel = region & (ix >= x0) & (ix < x1) & (iy >= y0) & (iy < y1)
        mm[iy[sel] - y0, ix[sel] - x0] = True
        if mm.any():
            a.contour(mm.astype(float), levels=[0.5], colors='lime', linewidths=1.5)
        a.set_title('%s   [zoom]' % nm, fontsize=12)
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
    for a in ax[len(show):]:
        a.axis('off')
    fig.suptitle('Zoom on the high-error region (green contour = selected pixels)', fontsize=13)
    fig.tight_layout()
    f2 = os.path.join(args.outdir, 'fig_arm_zoom_maps.png')
    fig.savefig(f2, dpi=150, bbox_inches='tight'); plt.close(fig)
    print('saved', f2)


if __name__ == '__main__':
    main()
