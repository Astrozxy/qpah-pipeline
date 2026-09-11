#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
align_maps.py — 把特征图用 reproject_exact 面积加权平均对齐到 qPAH 网格。

输出（到 <repo>/sanity_cache/）：
    {CO,dust,HI,HI_ew,SFR}_ex_mean.npy / _ex_cov.npy   特征 block 平均 + coverage
    CO_sigma_ex_mean.npy / _ex_cov.npy                 CO 1σ 噪声对齐（保守：波束内相关，不做√N）
    CO_det_ex_mean.npy / _ex_cov.npy                   qPAH 像素内 native S/N>3 的探测占比(0~1)

说明：
- 目标网格 = qPAH (1250x1250, 10"/px, RA-TAN)。
- CO：native I_CO 与 σ_I 由 make_co_maps.py 从 m31_co10_native 立方体生成
  （data/M31_CO_intensity.fits, data/M31_CO_noise.fits；暂忽略银河系前景）。
  detection = I/σ>3 的 native mask。
- NaN/覆盖处理：无效像素不进平均；coverage<=0.5 置 NaN。
- 幂等：已存在缓存则跳过。
"""
import os, time
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_exact

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
CACHE = os.path.join(ROOT, 'sanity_cache')
os.makedirs(CACHE, exist_ok=True)

T0 = time.time()
def log(*a):
    print('[%.0fs]' % (time.time() - T0), *a, flush=True)


def reproj_exact(data, wcs_in, wout, shape):
    """面积加权 block 平均。返回 (mean, coverage)。"""
    V = np.isfinite(data)
    D0 = np.where(V, data, 0.0).astype(np.float64)
    num, _ = reproject_exact((D0, wcs_in), wout, shape_out=shape)
    den, _ = reproject_exact((V.astype(np.float64), wcs_in), wout, shape_out=shape)
    mean = np.full(shape, np.nan)
    ok = (den > 0.5) & np.isfinite(num) & np.isfinite(den) & (den > 0)
    mean[ok] = num[ok] / den[ok]
    return mean, den


def align(name, data, wcs_in, wout, shape, cov_min=0.5):
    p = os.path.join(CACHE, name + '_ex_mean.npy')
    q = os.path.join(CACHE, name + '_ex_cov.npy')
    if os.path.exists(p) and os.path.exists(q):
        log(name, 'cached, skip')
        return np.load(p), np.load(q)
    t = time.time()
    mean, cov = reproj_exact(data, wcs_in, wout, shape)
    mean[~((cov > cov_min) & np.isfinite(mean))] = np.nan
    np.save(p, mean)
    np.save(q, cov)
    log(name, 'aligned in %.0fs' % (time.time() - t))
    return mean, cov


def align_detection(mask, wcs_in, wout, shape, cov_min=0.5):
    """mask(0/1 native) -> 每个 qPAH 像素内"被 CO 数据覆盖的 native 像素中探测占比"。
    返回 (frac, coverage)，coverage=有效 CO 数据占 qPAH 像素面积比例。
    """
    name = 'CO_det'
    p = os.path.join(CACHE, name + '_ex_mean.npy')
    q = os.path.join(CACHE, name + '_ex_cov.npy')
    if os.path.exists(p) and os.path.exists(q):
        log(name, 'cached, skip')
        return np.load(p), np.load(q)
    t = time.time()
    num, _ = reproject_exact((mask.astype(np.float64), wcs_in), wout, shape_out=shape)
    den, _ = reproject_exact((np.isfinite(mask).astype(np.float64), wcs_in),
                             wout, shape_out=shape)
    frac = np.full(shape, np.nan)
    ok = (den > cov_min) & np.isfinite(num)
    frac[ok] = num[ok] / den[ok]
    np.save(p, frac)
    np.save(q, den)
    log(name, 'aligned in %.0fs' % (time.time() - t))
    return frac, den


def load2d(fn):
    hd = fits.open(os.path.join(DATA, fn))
    d = hd[0].data.astype(np.float64, copy=False)
    w = WCS(hd[0].header)
    hd.close()
    return d, w


def main():
    hq = fits.open(os.path.join(DATA, 'M31_S350_110_SSS_110_Model_All_qpah.fits'))
    wout = WCS(hq[0].header)
    shape = hq[0].data.shape
    hq.close()
    log('target grid', shape)

    # 清掉旧的 CO 系列缓存（旧的立方体 moment0 与新物理单位 I_CO 不同），强制重建
    for nm in ['CO', 'CO_sigma', 'CO_det']:
        for suf in ['_ex_mean.npy', '_ex_cov.npy']:
            fp = os.path.join(CACHE, nm + suf)
            if os.path.exists(fp):
                os.remove(fp)
                log('removed stale cache:', fp)

    native = {}
    native['HI'] = load2d('m31_C+D+tp_hilores_120pc_strict_mom0.fits')
    native['HI_ew'] = load2d('m31_C+D+tp_hilores_120pc_strict_ew.fits')
    native['SFR'] = load2d('M31-SFR.fits')
    native['dust'] = load2d('M31_dust_NH2_sm24.fits')

    # ---- CO（新处理：I 与 σ 来自 make_co_maps.py） ----
    co_i_file = os.path.join(DATA, 'M31_CO_intensity.fits')
    co_s_file = os.path.join(DATA, 'M31_CO_noise.fits')
    if not (os.path.exists(co_i_file) and os.path.exists(co_s_file)):
        raise FileNotFoundError('缺少 CO intensity/noise 图，请先运行 make_co_maps.py: '
                                + co_i_file)
    I_co, w_co = load2d('M31_CO_intensity.fits')
    sig_co, w_sig = load2d('M31_CO_noise.fits')
    det_native = (I_co / np.maximum(sig_co, 1e-12) > 3.0) & (sig_co > 0) & np.isfinite(I_co)
    native['CO'] = (I_co, w_co)
    log('native CO coverage: I finite=%d, det(S/N>3)=%d'
        % (int(np.isfinite(I_co).sum()), int(det_native.sum())))

    # ---- 对齐 ----
    for k in ['CO', 'dust', 'HI', 'HI_ew', 'SFR']:
        d, w = native[k]
        mean, cov = align(k, d, w, wout, shape)
        log('%s: finite(mean)=%d  cov>0.5=%d' % (k, np.isfinite(mean).sum(),
                                                 (cov > 0.5).sum()))
    # CO 噪声与探测
    align('CO_sigma', sig_co, w_sig, wout, shape)
    det_frac, det_cov = align_detection(det_native.astype(np.float64), w_co, wout, shape)
    log('CO_sigma: finite=%d' % int(np.isfinite(
        np.load(os.path.join(CACHE, 'CO_sigma_ex_mean.npy'))).sum()))
    log('CO_det: finite(frac)=%d, median(frac|det-px)=%.3f'
        % (int(np.isfinite(det_frac).sum()), np.nanmedian(det_frac[det_frac > 0])))

    log('ALL DONE ->', CACHE)


if __name__ == '__main__':
    main()
