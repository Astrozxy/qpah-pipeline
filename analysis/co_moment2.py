#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
co_moment2.py — 从 CO(1-0) cube 计算速度弥散 sigma_CO，并对齐到 qPAH 网格。

物理动机：PAH 会被激波/强湍流破坏，而速度弥散是这些过程的示踪量。
sigma_CO 来自 cube 的二阶矩，与 DL07 的 SED 拟合完全独立。

输出：
    <cache>/COsd_ex_mean.npy / COsd_ex_cov.npy
    <out> 逐行 npz：mom0 / mom1 / sigma_CO（对应 --h5 的行）
"""
import os, argparse, time
import numpy as np
import h5py
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_exact


def reproj(data, wcs_in, wout, shape):
    V = np.isfinite(data)
    D0 = np.where(V, data, 0.0).astype(np.float64)
    Vf = V.astype(np.float64)
    num, _ = reproject_exact((D0, wcs_in), wout, shape_out=shape)
    den, _ = reproject_exact((Vf, wcs_in), wout, shape_out=shape)
    mean = np.full(shape, np.nan)
    ok = (den > 0.5) & np.isfinite(num) & np.isfinite(den) & (den > 0)
    mean[ok] = num[ok] / den[ok]
    return mean, den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--cache', default='sanity_cache')
    ap.add_argument('--cube', default='m31_co10_native.fits')
    ap.add_argument('--snr-min', type=float, default=4.0,
                    help='通道级 S/N 门槛（低于此的通道不参与矩计算）')
    ap.add_argument('--smoo', type=int, default=3,
                    help='谱平滑通道数（奇数）。cube 通道宽仅 2.6 km/s，CO 线远宽于此，'
                         '先用它压噪声再取峰值')
    ap.add_argument('--vwin', type=float, default=120.0,
                    help='第二遍只在该半宽窗口内（相对 mom1，km/s）参与矩计算 —— '
                         '防止远处噪声通道把二阶矩抬高')
    ap.add_argument('--pass1-snr', type=float, default=6.0,
                    help='第一遍（定 mom1）用的更严门槛')
    ap.add_argument('--h5', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    t0 = time.time()

    hq = fits.open(os.path.join(args.data, 'M31_S350_110_SSS_110_Model_All_qpah.fits'))
    wout = WCS(hq[0].header).celestial
    shape = hq[0].data.shape
    hq.close()

    hd = fits.open(os.path.join(args.data, args.cube))
    cube = hd[0].data.astype(np.float32)
    hdr = hd[0].header
    wc = WCS(hdr).celestial
    nz, ny, nx = cube.shape
    # 速度轴 (m/s -> km/s)
    i = np.arange(1, nz + 1, dtype=np.float64)
    vel = (hdr['CRVAL3'] + (i - hdr['CRPIX3']) * hdr['CDELT3']) / 1000.0   # km/s
    dv = abs(hdr['CDELT3']) / 1000.0
    print('cube %s | %d 通道 | v = %.1f .. %.1f km/s (dv=%.3f)' %
          (cube.shape, nz, vel.min(), vel.max(), dv))
    hd.close()

    # 噪声：用边缘通道的空间散布估计（每通道一个 sigma）
    sig_ch = np.nanstd(cube[:8], axis=(1, 2))
    sig_ch = np.append(sig_ch, np.nanstd(cube[-8:], axis=(1, 2)))
    sig_med = float(np.nanmedian(sig_ch))
    print('通道噪声中位 %.4f K' % sig_med)

    # 只在 |v| 窗口内、且高于门槛的通道参与矩计算
    base = np.where(np.isfinite(cube), cube, 0.0)

    # ---- 谱平滑（仅用于定峰值/中心，不用于积分）----
    if args.smoo and args.smoo > 1:
        k = args.smoo
        ker = np.ones(k, dtype=np.float32) / k
        pad = k // 2
        sm = np.apply_along_axis(lambda v: np.convolve(v, ker, mode='same'), 0, base)
    else:
        sm = base

    # ---- 第一遍：严格门槛定出中心速度 ----
    T1 = np.where(base >= args.pass1_snr * sig_med, base, 0.0)
    m0a = T1.sum(axis=0) * dv
    has = m0a > 0
    with np.errstate(invalid='ignore', divide='ignore'):
        m1 = np.where(has, (T1 * vel[:, None, None]).sum(axis=0) * dv / np.where(has, m0a, np.nan), np.nan)

    # ---- 第二遍：只在 ±vwin 窗口内、且过门槛的通道 ----
    win = np.abs(vel[:, None, None] - m1[None]) <= args.vwin
    T = np.where(win & (base >= args.snr_min * sig_med), base, 0.0)
    m0 = T.sum(axis=0) * dv
    has = m0 > 0
    with np.errstate(invalid='ignore', divide='ignore'):
        m1b = np.where(has, (T * vel[:, None, None]).sum(axis=0) * dv / np.where(has, m0, np.nan), np.nan)
        var = np.where(has, (T * (vel[:, None, None] - m1b[None]) ** 2).sum(axis=0) * dv
                       / np.where(has, m0, np.nan), np.nan)
    m2 = np.sqrt(np.maximum(var, 0))
    m1 = m1b
    # 通道数太少 => 二阶矩不可信
    nchan = (T > 0).sum(axis=0)
    m2 = np.where(nchan >= 4, m2, np.nan)

    # ---- CO 等效宽度：EW = ∫T dv / T_peak（以峰值通道为中心开窗）----
    ipk = np.argmax(sm, axis=0)
    peak = np.take_along_axis(sm, ipk[None], axis=0)[0]
    vpk = vel[ipk]
    win2 = np.abs(vel[:, None, None] - vpk[None]) <= args.vwin
    Te = np.where(win2 & (base >= args.snr_min * sig_med), base, 0.0)
    m0e = Te.sum(axis=0) * dv
    with np.errstate(invalid='ignore', divide='ignore'):
        ew = np.where(peak > args.snr_min * sig_med, m0e / np.where(peak > 0, peak, np.nan), np.nan)
    # 窗口必须完整覆盖（否则 EW 被截断）
    touch = (np.abs(vel[:, None, None] - vpk[None]) > args.vwin - 2 * dv) & (Te > 0)
    ew = np.where(touch.sum(axis=0) > 0, np.nan, ew)
    print('  CO_ew: 有效 %.1f%% | 中位 %.2f km/s (p16 %.2f p84 %.2f)'
          % (100 * np.mean(np.isfinite(ew)), np.nanmedian(ew),
             np.nanpercentile(ew[np.isfinite(ew)], 16), np.nanpercentile(ew[np.isfinite(ew)], 84)))
    del Te, win2, sm
    print('mom0 >0 占 %.1f%% | sigma_CO 有效占 %.1f%% | sigma 中位 %.2f km/s (p16 %.2f p84 %.2f)'
          % (100 * np.mean(m0 > 0), 100 * np.mean(np.isfinite(m2)),
             np.nanmedian(m2), np.nanpercentile(m2[np.isfinite(m2)], 16),
             np.nanpercentile(m2[np.isfinite(m2)], 84)))
    del cube, T, T1
    print('  通道数中位 %.0f | 用于二阶矩的像素里 nchan>=4 的占 %.1f%%'
          % (np.median(nchan[nchan > 0]), 100 * np.mean(nchan >= 4)))
    print('矩计算 %.0fs' % (time.time() - t0))

    out = {}
    for nm, arr in [('COsd', m2), ('COm0', m0), ('COm1', m1), ('COew', ew)]:
        cp = os.path.join(args.cache, nm + '_ex_mean.npy')
        cc = os.path.join(args.cache, nm + '_ex_cov.npy')
        if os.path.exists(cp) and os.path.exists(cc):
            al, cv = np.load(cp).astype(float), np.load(cc).astype(float)
        else:
            al, cv = reproj(arr, wc, wout, shape)
            np.save(cp, al.astype(np.float32)); np.save(cc, cv.astype(np.float32))
            print('  对齐 %-5s 完成 %.0fs' % (nm, time.time() - t0))
        out[nm] = (al, cv)

    with h5py.File(args.h5) as f:
        ra = f['ra'][:].astype(np.float64); dec = f['dec'][:].astype(np.float64)
    x, y = wout.world_to_pixel_values(ra, dec)
    ix = np.clip(np.round(x).astype(int), 0, shape[1] - 1)
    iy = np.clip(np.round(y).astype(int), 0, shape[0] - 1)
    res = {}
    for nm, (al, cv) in out.items():
        res[nm] = al[iy, ix].astype(np.float32)
        res[nm + '_cov'] = cv[iy, ix].astype(np.float32)
        print('  逐行 %-5s 有效 %.1f%% | 中位 %.3f' %
              (nm, 100 * np.isfinite(res[nm]).mean(), np.nanmedian(res[nm])))
    np.savez_compressed(args.out, **res)
    print('saved %s (%.2f MB)' % (args.out, os.path.getsize(args.out) / 1e6))


if __name__ == '__main__':
    main()
