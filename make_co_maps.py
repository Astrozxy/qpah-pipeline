#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_co_maps.py — 从 m31_co10_native.fits 立方体生成两个图：
  1) M31_CO_intensity.fits   I_CO = Σ_channels T × Δv        [K km/s]
  2) M31_CO_noise.fits       1σ uncertainty of I_CO          [K km/s]

噪声原理（不直接取窗口内通道 std 的原因）：
  - moment0 是各通道噪声的线性叠加；若通道噪声独立同分布 σ_ch：
        σ_I = σ_ch × Δv × √N_used
  - σ_ch 只能从"无线通道"估计，窗口内 std 会被 CO 线本身拉大；
  - 这里用迭代 MAD（1.4826×MAD）做鲁棒 σ_ch，抗残留谱线/离群通道。

说明/分支：
  - 默认对全部 259 个通道积分（暂时忽略银河系前景的贡献）；
  - 提供 --vmin/--vmax 速度窗参数，未来启用"银河系速度分离"分支时使用。
"""
import os, time, argparse
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

REPO = os.path.dirname(os.path.abspath(__file__))   # qpah_pipeline/
ROOT = os.path.dirname(REPO)                        # data 所在根目录
T0 = time.time()


def log(*a):
    print('[%.0fs]' % (time.time() - T0), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cube', default=os.path.join(ROOT, 'data', 'm31_co10_native.fits'))
    ap.add_argument('--outdir', default=os.path.join(ROOT, 'data'))
    ap.add_argument('--vmin', type=float, default=None, help='积分速度下限 km/s (可选)')
    ap.add_argument('--vmax', type=float, default=None, help='积分速度上限 km/s (可选)')
    ap.add_argument('--min-good-frac', type=float, default=0.15,
                    help='有效通道占比低于该值的像素置 NaN (默认0.15≈39道；'
                         '该立方体覆盖很碎，0.5 会砍掉太多真实像素)')
    ap.add_argument('--clip-sigma', type=float, default=6.0,
                    help='迭代剔除通道的倍数(相对 MAD 噪声)')
    ap.add_argument('--clip-iters', type=int, default=2)
    args = ap.parse_args()

    hdul = fits.open(args.cube, memmap=True)
    cube = hdul[0].data
    hdr = hdul[0].header
    log('cube shape:', cube.shape)

    # ---- 速度轴 ----
    nv = hdr['NAXIS3']
    crval3 = hdr['CRVAL3']          # m/s
    cdelt3 = hdr['CDELT3']          # m/s per channel (负)
    crpix3 = hdr['CRPIX3']          # 1-based
    chan = np.arange(nv, dtype=float) + 1.0
    vel = (crval3 + (chan - crpix3) * cdelt3) / 1000.0   # km/s
    dvel = abs(cdelt3) / 1000.0                            # km/s per channel
    log('velocity coverage: %.1f .. %.1f km/s, dvel=%.3f' % (vel[-1], vel[0], dvel))

    sel = np.ones(nv, dtype=bool)
    if args.vmin is not None:
        sel &= vel >= args.vmin
    if args.vmax is not None:
        sel &= vel <= args.vmax
    log('selected channels: %d / %d (%.1f..%.1f km/s)'
        % (sel.sum(), nv, vel[sel].min(), vel[sel].max()))

    # ---- I_CO = Σ T×Δv ----
    T = cube[sel].astype(np.float64)          # (nch, ny, nx)
    nsel = sel.sum()
    good = np.isfinite(T)
    ngood = good.sum(axis=0)
    I = np.nansum(np.where(good, T, 0.0), axis=0) * dvel
    log('pixels with >=1 good channel: %d (%.1f%%)' % (int(np.sum(ngood >= 1)),
                                                         100 * np.mean(ngood >= 1)))
    bad = ngood < args.min_good_frac * nsel
    I = np.where(bad, np.nan, I)
    log('I_CO: finite=%d (%.1f%%), median=%.4g, max=%.4g K km/s'
        % (np.isfinite(I).sum(), 100 * np.isfinite(I).mean(),
           np.nanmedian(I), np.nanmax(I)))

    # ---- 逐像素 σ_ch：迭代 MAD（无线通道估计） ----
    # 把候选噪声通道外的强线通道逐步置 NaN，再对剩余通道求鲁棒 MAD
    Tw = T.copy()
    Tw[~good] = np.nan
    sig_ch = None
    for it in range(args.clip_iters):
        with np.errstate(all='ignore'):
            med = np.nanmedian(Tw, axis=0)
            dev = np.abs(Tw - med[None, :, :])
            mad = np.nanmedian(dev, axis=0)
        sig = 1.4826 * mad
        sig = np.where(np.isfinite(sig) & (sig > 0), sig, np.nan)
        if it < args.clip_iters - 1:
            Tw[dev > args.clip_sigma * sig[None, :, :]] = np.nan
        else:
            sig_ch = sig
    # 未收敛像素用首轮 sig 兜底
    with np.errstate(all='ignore'):
        med0 = np.nanmedian(np.where(good, T, np.nan), axis=0)
        dev0 = np.abs(np.where(good, T, np.nan) - med0[None, :, :])
        mad0 = np.nanmedian(dev0, axis=0)
    sig0 = 1.4826 * mad0
    sig_ch = np.where(np.isfinite(sig_ch), sig_ch, sig0)
    log('sigma_ch: finite=%d, median=%.4g, p95=%.4g K'
        % (np.isfinite(sig_ch).sum(), np.nanmedian(sig_ch),
           np.nanpercentile(sig_ch, 95)))

    # ---- σ_I = σ_ch × Δv × √(有效通道数) ----
    n_used = np.clip(ngood, 0, None)
    sig_I = sig_ch * dvel * np.sqrt(n_used)
    sig_I = np.where(bad | ~np.isfinite(sig_I), np.nan, sig_I)
    log('sigma_I: finite=%d, median=%.4g, p95=%.4g K km/s'
        % (np.isfinite(sig_I).sum(), np.nanmedian(sig_I),
           np.nanpercentile(sig_I, 95)))

    # ---- 写 FITS（2D 天球 WCS） ----
    w2d = WCS(hdr).celestial
    hdr2 = w2d.to_header()
    hdr2['BUNIT'] = 'K km/s'
    for name, arr, fname in [('I_CO', I, 'M31_CO_intensity.fits'),
                             ('sigma_I', sig_I, 'M31_CO_noise.fits')]:
        hdr3 = hdr2.copy()
        hdr3['HISTORY'] = 'made by make_co_maps.py: %s' % name
        hdr3['HISTORY'] = 'velocity window (km/s): %.1f..%.1f, dvel=%.3f' % (
            vel[sel].min(), vel[sel].max(), dvel)
        out = os.path.join(args.outdir, fname)
        fits.writeto(out, arr.astype(np.float32), hdr3, overwrite=True)
        log('saved', out)

    # ---- 诊断：平均谱存 png ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        spec = np.nanmean(cube.astype(np.float64), axis=(1, 2))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.step(vel, spec, where='mid', lw=1)
        ax.axvspan(vel[sel].min(), vel[sel].max(), color='orange', alpha=0.15,
                   label='integration window')
        ax.set_xlabel('v (km/s, LSRK)'); ax.set_ylabel('mean T (K)')
        ax.set_title('cube-averaged spectrum (diagnostic)')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(ROOT, 'tmp', 'co_diagnostic_spectrum.png'), dpi=150)
        plt.close(fig)
        log('saved tmp/co_diagnostic_spectrum.png')
    except Exception as e:
        log('spectrum png skipped:', e)
    hdul.close()
    log('ALL DONE')


if __name__ == '__main__':
    main()
