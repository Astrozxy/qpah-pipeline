#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_ldust.py — 把 Ldust 摊开看：图、与 qPAH/气体/尘埃/半径的关系。
"""
import os, argparse
import numpy as np
import h5py
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/home/xzhang/0_Stanford/paper_0/local_data/data')
    ap.add_argument('--h5', default='/home/xzhang/0_Stanford/paper_0/results_final/M31_1d_array_final.h5')
    ap.add_argument('--out', default='/home/xzhang/0_Stanford/paper_0/results_final/fig_ldust.png')
    args = ap.parse_args()

    hd = fits.open(os.path.join(args.data, 'M31_S350_110_SSS_110_Model_All_SurfBr_Ldust.fits'))
    w = WCS(hd[0].header).celestial
    Ld = hd[0].data.astype(float)
    shape = Ld.shape
    hd.close()
    hq = fits.open(os.path.join(args.data, 'M31_S350_110_SSS_110_Model_All_qpah.fits'))
    qp = hq[0].data.astype(float); hq.close()
    Md = fits.open(os.path.join(args.data, 'M31_S350_110_SSS_110_Model_All_SurfBr_Mdust.fits'))[0].data.astype(float)

    with h5py.File(args.h5) as f:
        d = {k: f[k][:] for k in ['ra', 'dec', 'qpah', 'qpah_err', 'H1', 'Ldust', 'Mdust', 'R_kpc',
                                  'sfr', 'CO', 'H1_ew', 'U_min', 'U_bar']}
    ra, dec = d['ra'].astype(float), d['dec'].astype(float)
    x, y = w.world_to_pixel_values(ra, dec)
    ix = np.clip(np.round(x).astype(int), 0, shape[1] - 1)
    iy = np.clip(np.round(y).astype(int), 0, shape[0] - 1)

    pos = Ld > 0
    print('Ldust 图: 有效 %d px (%.1f%%) | 中位 %.4g | p1 %.3g p99 %.3g | max %.3g'
          % (pos.sum(), 100 * pos.mean(), np.median(Ld[pos]),
             np.percentile(Ld[pos], 1), np.percentile(Ld[pos], 99), Ld.max()))
    L = d['Ldust'].astype(float)
    print('样本中 Ldust: 中位 %.4g | p1 %.3g p99 %.3g | min %.3g' %
          (np.median(L), np.percentile(L, 1), np.percentile(L, 99), L.min()))

    fig = plt.figure(figsize=(24, 13))
    gs = fig.add_gridspec(2, 3, hspace=0.22, wspace=0.22)

    # 1) 全网格 Ldust
    a = fig.add_subplot(gs[0, 0])
    g = Ld[pos]
    im = a.imshow(np.where(pos, Ld, np.nan), origin='lower', cmap='inferno',
                  norm=LogNorm(vmin=np.percentile(g, 2), vmax=np.percentile(g, 98)))
    a.plot(x, y, ',', color='cyan', alpha=.35)
    a.set_title('Ldust on the qPAH grid (log)\ncyan = sample footprint', fontsize=12)
    a.set_xlabel('grid x (10")'); a.set_ylabel('grid y (10")')
    plt.colorbar(im, ax=a, fraction=0.046, pad=0.02, label=r'$L_{\rm dust}$ ($L_\odot$/kpc$^2$)')

    # 2) 样本内的 Ldust（RA/Dec）
    a = fig.add_subplot(gs[0, 1])
    order = np.argsort(L)
    sc = a.scatter(ra[order], dec[order], s=1.0, c=L[order], cmap='inferno',
                   norm=LogNorm(vmin=np.percentile(L, 1), vmax=np.percentile(L, 99)),
                   rasterized=True, linewidths=0)
    a.set_title(r'Sample: $L_{\rm dust}$  (n=%d)' % len(L), fontsize=12)
    a.set_xlabel('RA (J2000)'); a.set_ylabel('Dec (J2000)')
    a.invert_xaxis(); a.set_aspect('equal')
    plt.colorbar(sc, ax=a, fraction=0.046, pad=0.02, label=r'$L_{\rm dust}$ ($L_\odot$/kpc$^2$)')

    # 3) Ldust vs qPAH
    a = fig.add_subplot(gs[0, 2])
    m = (L > 0) & (d['qpah'] > 0)
    hb = a.hexbin(np.log10(L[m]), d['qpah'][m], gridsize=60, bins='log', cmap='viridis')
    a.set_xlabel(r'$\log_{10} L_{\rm dust}$'); a.set_ylabel('qPAH (%)')
    a.set_title('Ldust vs qPAH   Pearson %+.3f' % np.corrcoef(np.log10(L[m]), d['qpah'][m])[0, 1])
    plt.colorbar(hb, ax=a, fraction=0.046, pad=0.02, label='N')

    # 4) Ldust vs H1
    a = fig.add_subplot(gs[1, 0])
    m = (L > 0) & (d['H1'] > 0)
    hb = a.hexbin(np.log10(d['H1'][m]), np.log10(L[m]), gridsize=60, bins='log', cmap='viridis')
    a.set_xlabel(r'$\log_{10}$ H I (K km/s)'); a.set_ylabel(r'$\log_{10} L_{\rm dust}$')
    a.set_title('Ldust vs H I (gas-to-dust)   Pearson %+.3f'
                % np.corrcoef(np.log10(d['H1'][m]), np.log10(L[m]))[0, 1])
    plt.colorbar(hb, ax=a, fraction=0.046, pad=0.02, label='N')

    # 5) Ldust vs Mdust
    a = fig.add_subplot(gs[1, 1])
    m = (L > 0) & (d['Mdust'] > 0)
    hb = a.hexbin(np.log10(d['Mdust'][m]), np.log10(L[m]), gridsize=60, bins='log', cmap='viridis')
    a.set_xlabel(r'$\log_{10} M_{\rm dust}$'); a.set_ylabel(r'$\log_{10} L_{\rm dust}$')
    a.set_title('Ldust vs Mdust (both DL07)   Pearson %+.3f'
                % np.corrcoef(np.log10(d['Mdust'][m]), np.log10(L[m]))[0, 1])
    plt.colorbar(hb, ax=a, fraction=0.046, pad=0.02, label='N')

    # 6) 径向 + Ldust/Mdust
    a = fig.add_subplot(gs[1, 2])
    R = d['R_kpc'].astype(float)
    bins = np.arange(0, 18.5, 1.0)
    idx = np.digitize(R, bins) - 1
    cen, med, q16, q84 = [], [], [], []
    for i in range(len(bins) - 1):
        mm = idx == i
        if mm.sum() < 30: continue
        cen.append(0.5 * (bins[i] + bins[i + 1])); med.append(np.median(L[mm]))
        q16.append(np.percentile(L[mm], 16)); q84.append(np.percentile(L[mm], 84))
    a.plot(cen, med, 'o-', color='crimson', label=r'$L_{\rm dust}$')
    a.fill_between(cen, q16, q84, color='crimson', alpha=.2)
    a.set_yscale('log'); a.set_xlabel('R (kpc)'); a.set_ylabel(r'$L_{\rm dust}$ ($L_\odot$/kpc$^2$)')
    a.set_title('radial profile of Ldust'); a.grid(alpha=.25)
    a2 = a.twinx()
    a2.plot(cen, np.array(med) / np.array([np.median(d['Mdust'][idx == i]) if (idx == i).sum() else np.nan
                                           for i in range(len(bins) - 1)]),
            's--', color='steelblue', alpha=.8, label=r'$L_{\rm dust}/M_{\rm dust}$')
    a2.set_ylabel(r'$L_{\rm dust}/M_{\rm dust}$  ($L_\odot/M_\odot$)', color='steelblue')
    a.legend(loc='upper right', fontsize=9)

    fig.suptitle(r'$L_{\rm dust}$  —  DL07 total dust luminosity surface density', fontsize=15)
    fig.savefig(args.out, dpi=120, bbox_inches='tight')
    print('saved', args.out)


if __name__ == '__main__':
    main()
