#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_cut_dataset.py — 按目标质量 cut 派生一个新数据集（默认 qpah_err <= 2），
并同步裁剪本地紧凑 aux（coverage / 边缘距 / 空间项），使全部脚本无需改动即可统一口径。

理由（用户 2026-10-09 拍板）：
  qPAH 的动态范围是 0–10%，一个 ±2% 的不确定度等于"什么都不知道"；
  这些像素又集中在视场边缘，很可能是观测本身的问题 ⇒ 直接剔除。

同时出一张【cut 诊断图】记录被剔除的像素在哪、是什么样，便于论文交代。

用法:
  python make_cut_dataset.py --h5 .../M31_1d_array_nodust.h5 \
      --aux .../aux_sample.npz --qpah-err-max 2.0 --outdir /tmp/cut
"""
import os, argparse
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mplconfig')
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FEATS = ['H1', 'sfr', 'CO', 'H1_ew']
RA0, DEC0, INC, PA = 10.6847, 41.2690, np.radians(77.7), np.radians(37.7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', required=True)
    ap.add_argument('--aux', default=None, help='本地紧凑 aux_sample.npz（可选，会同步裁剪）')
    ap.add_argument('--mask-npz', default=None, help='悬臂 mask npz（可选，会同步裁剪）')
    ap.add_argument('--qpah-err-max', type=float, default=2.0)
    ap.add_argument('--outdir', default='.')
    ap.add_argument('--tag', default=None)
    args = ap.parse_args()
    tag = args.tag or ('qe%g' % args.qpah_err_max)
    os.makedirs(args.outdir, exist_ok=True)

    with h5py.File(args.h5, 'r') as f:
        cols = list(f.keys())
        d = {k: f[k][:] for k in cols}
    base = ((d['qpah'] > 0) & (d['qpah_err'] > 0) & np.isfinite(d['H1']) & (d['H1'] >= 0)
            & np.isfinite(d['H1_ew']) & (d['H1_ew'] >= 0) & np.isfinite(d['sfr'])
            & np.isfinite(d['CO']) & np.isfinite(d['CO_sigma']) & (d['CO_sigma'] > 0))
    keep = base & (d['qpah_err'] <= args.qpah_err_max)
    drop = base & ~keep
    print('基础样本 %d → 保留 %d (%.1f%%)，剔除 %d (%.1f%%)  [qpah_err > %.1f]'
          % (base.sum(), keep.sum(), 100 * keep.sum() / base.sum(),
             drop.sum(), 100 * drop.sum() / base.sum(), args.qpah_err_max))
    if drop.sum():
        print('  被剔除者: qpah_err 中位=%.2f 最大=%.2f | qpah 中位=%.2f'
              % (np.median(d['qpah_err'][drop]), d['qpah_err'][drop].max(),
                 np.median(d['qpah'][drop])))
        print('  保留者  : qpah_err 中位=%.2f | qpah 中位=%.2f'
              % (np.median(d['qpah_err'][keep]), np.median(d['qpah'][keep])))

    # ---- 写出新 h5 ----
    outp = os.path.join(args.outdir, os.path.basename(args.h5).replace('.h5', '_%s.h5' % tag))
    with h5py.File(outp, 'w') as f:
        for k in cols:
            arr = d[k]
            f.create_dataset(k, data=arr[keep])
        f.attrs['cut'] = 'qpah_err <= %g' % args.qpah_err_max
        f.attrs['n_base'] = int(base.sum())
    print('written', outp, '(%.1f MB)' % (os.path.getsize(outp) / 1e6))

    # ---- 同步裁剪 aux ----
    if args.aux:
        za = np.load(args.aux)
        n_aux = len(za['cov_H1'])
        if n_aux != int(base.sum()):
            raise SystemExit('aux 长度 %d != 基础样本 %d' % (n_aux, base.sum()))
        sub = {k: za[k][keep[base]] for k in za.files}
        ap_ = os.path.join(args.outdir, os.path.basename(args.aux).replace('.npz', '_%s.npz' % tag))
        np.savez_compressed(ap_, **sub)
        print('written', ap_, '(%.1f MB)' % (os.path.getsize(ap_) / 1e6))

    if args.mask_npz:
        zm = np.load(args.mask_npz)
        if len(zm['region']) != int(base.sum()):
            raise SystemExit('mask 长度 %d != 基础样本 %d' % (len(zm['region']), base.sum()))
        mp = os.path.join(args.outdir, os.path.basename(args.mask_npz).replace('.npz', '_%s.npz' % tag))
        np.savez_compressed(mp, ra=zm['ra'][keep[base]], dec=zm['dec'][keep[base]],
                            region=zm['region'][keep[base]], chi=zm['chi'][keep[base]])
        print('written', mp, ' (region 保留 %d / %d)'
              % (zm['region'][keep[base]].sum(), zm['region'].sum()))

    # ---- cut 诊断图 ----
    ra, dec = d['ra'][base].astype(float), d['dec'][base].astype(float)
    dx = (ra - RA0) * np.cos(np.radians(DEC0)); dy = dec - DEC0
    X = dx * np.cos(PA) + dy * np.sin(PA)
    Y = (-dx * np.sin(PA) + dy * np.cos(PA)) / np.cos(INC)
    kk = 3600.0 / (206265.0 / 744.0)
    R = np.sqrt(X ** 2 + Y ** 2) * kk
    kb, db = keep[base], drop[base]

    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6))
    ax[0].scatter(ra[kb], dec[kb], s=0.5, c='lightgray', rasterized=True, linewidths=0)
    ax[0].scatter(ra[db], dec[db], s=2.0, c='crimson', rasterized=True, linewidths=0)
    ax[0].set_title('cut map: kept (grey) vs removed $q_{\\rm PAH}$ err>$%.1f$ (red)' % args.qpah_err_max)
    ax[0].set_xlabel('RA (deg)'); ax[0].set_ylabel('Dec (deg)')
    ax[0].invert_xaxis(); ax[0].set_aspect('equal')

    ax[1].hist(R[kb], bins=60, range=(0, 25), density=True, color='lightgray',
               label='kept (%.0f%%)' % (100 * kb.mean()))
    ax[1].hist(R[db], bins=60, range=(0, 25), density=True, histtype='step',
               color='crimson', lw=2, label='removed')
    ax[1].set_xlabel('R (kpc, deprojected)'); ax[1].set_ylabel('density')
    ax[1].set_title('radial distribution'); ax[1].legend(); ax[1].grid(alpha=.3)
    print('  R 中位: 保留 %.1f kpc | 剔除 %.1f kpc' % (np.median(R[kb]), np.median(R[db])))

    ax[2].hist(d['qpah_err'][base], bins=80, range=(0, 4), color='steelblue')
    ax[2].axvline(args.qpah_err_max, color='crimson', ls='--', lw=2,
                  label='cut = %.1f' % args.qpah_err_max)
    ax[2].set_xlabel(r'$q_{\rm PAH}$ err (%)'); ax[2].set_ylabel('N')
    ax[2].set_title(r'$q_{\rm PAH}$ error distribution'); ax[2].legend(); ax[2].grid(alpha=.3)
    fig.tight_layout()
    fp = os.path.join(args.outdir, 'fig_cut_diagnostic_%s.png' % tag)
    fig.savefig(fp, dpi=150, bbox_inches='tight'); plt.close(fig)
    print('saved', fp)


if __name__ == '__main__':
    main()
