#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_co_update_map.py — CO 更新前/后 map 并排对比（CO_obs vs c_true）+ 差值 + 显著性。

读 results/<tag>/co_true_train.npz（由 train_co_alternate.py 生成）。
注意：按约定只有**训练扇区**像素做了 CO 反推（测试扇区不能用 qPAH 反推），
图中青色虚线标出测试扇区边界，那些区域没有更新值。

用法: python plot_co_update_map.py --tag coalt_full
输出: results/<tag>/fig_co_update_map.png (+ .pdf)
"""
import os, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(REPO, 'results')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='coalt_full')
    ap.add_argument('--xmin', type=float, default=10.01)
    ap.add_argument('--xmax', type=float, default=11.5)
    ap.add_argument('--ymin', type=float, default=40.6)
    ap.add_argument('--ymax', type=float, default=42.0)
    args = ap.parse_args()
    resdir = os.path.join(RESULTS, args.tag)
    d = np.load(os.path.join(resdir, 'co_true_train.npz'))

    ra, dec = d['ra'], d['dec']
    c_obs, c_upd = d['co_meas'], d['co_true']
    pull = d['co_pull']
    diff = c_upd - c_obs
    sig_ratio = d['sigma_ratio'] if 'sigma_ratio' in d else None

    # 与训练一致的扇区划分（test 扇区不做 CO 更新）
    ra_c, dec_c = 10.75, 41.25
    np.random.seed(0)
    tr = np.random.choice(np.arange(10), size=6, replace=False)
    te = np.setdiff1d(np.arange(10), tr)
    sector_bins = np.linspace(0, 2 * np.pi, 11)
    R = 1.0

    plt.style.use('default')
    plt.rcParams.update({'font.size': 13})
    vmax_co = float(np.nanpercentile(c_obs, 99))
    vmax_d = float(np.nanpercentile(np.abs(diff), 99))

    fig, axes = plt.subplots(1, 4, figsize=(23, 5.4))

    def style(ax, title):
        for s in te:
            for th in (sector_bins[s], sector_bins[s + 1]):
                ax.plot([ra_c, ra_c + R * np.cos(th)],
                        [dec_c, dec_c + R * np.sin(th)],
                        '--', color='cyan', lw=1.0, alpha=0.85)
        ax.set_xlim(args.xmin, args.xmax)
        ax.set_ylim(args.ymin, args.ymax)
        ax.invert_xaxis()
        ax.set_xlabel('RA (J2000)')
        ax.set_title(title, fontsize=13)
        ax.tick_params(direction='in')
        ax.grid(True, ls=':')

    ax = axes[0]
    sc = ax.scatter(ra, dec, c=c_obs, s=1.0, cmap='viridis',
                    vmin=0, vmax=vmax_co, rasterized=True)
    ax.set_ylabel('Dec (J2000)')
    style(ax, r'CO observed  $I_{\rm CO}$ (before)')
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02).set_label(
        r'$I_{\rm CO}$ (K km s$^{-1}$)')

    ax = axes[1]
    sc = ax.scatter(ra, dec, c=c_upd, s=1.0, cmap='viridis',
                    vmin=0, vmax=vmax_co, rasterized=True)
    style(ax, r'CO updated  $c^{\rm true}$ (after)')
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02).set_label(
        r'$I_{\rm CO}$ (K km s$^{-1}$)')

    ax = axes[2]
    sc = ax.scatter(ra, dec, c=diff, s=1.0, cmap='coolwarm',
                    vmin=-vmax_d, vmax=vmax_d, rasterized=True)
    style(ax, r'$\Delta I_{\rm CO}$ = after $-$ before')
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02).set_label(
        r'$\Delta I_{\rm CO}$ (K km s$^{-1}$)')

    ax = axes[3]
    sc = ax.scatter(ra, dec, c=pull, s=1.0, cmap='coolwarm',
                    vmin=-2, vmax=2, rasterized=True)
    style(ax, r'Significance  $(c^{\rm true}-I_{\rm CO})/\sigma_{\rm CO}$')
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02).set_label(r'$n\sigma$')

    extra = ''
    if sig_ratio is not None:
        extra = '   |   median $\\sigma_{\\rm post}/\\sigma_{\\rm CO}$ = %.2f' % np.median(sig_ratio)
    fig.suptitle('CO update driven by qPAH — training sectors only'
                 ' (dashed cyan = test sectors, not updated)' + extra, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    png = os.path.join(resdir, 'fig_co_update_map.png')
    fig.savefig(png, dpi=170)
    fig.savefig(os.path.join(resdir, 'fig_co_update_map.pdf'))
    plt.close(fig)
    print('saved', png)
    print('n=%d | dCO med=%.3f p16=%.3f p84=%.3f | pull std=%.3f'
          % (len(ra), np.median(diff), np.percentile(diff, 16),
             np.percentile(diff, 84), pull.std()))
    print('ALL DONE')


if __name__ == '__main__':
    main()
