#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dust_for_zeros.py — 加 dust(850μm) 能否让模型碰到 qPAH=0？

关键：加 dust 会把样本限制在 SCUBA-2 覆盖区（约 11% 网格），
所以必须同时跑一个【同子样本、不加 dust】的对照，才能把 dust 的贡献和
"换样本" 的贡献分开。

对每个模型报：全体 chi2、**在 qpah==0 像素上的 chi2 / pull 中位 / 预测中位**、
预测最小值，以及全体 pull 分布。
"""
import os, argparse
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mplconfig')
import numpy as np
import h5py

SPA = {'R': 'R', 'x': 'x', 'y': 'y', 'logR': 'logR'}
RA0, DEC0, INC, PA = 10.6847, 41.2690, np.radians(77.7), np.radians(37.7)


def spatial(ra, dec):
    dx = (ra - RA0) * np.cos(np.radians(DEC0)); dy = dec - DEC0
    x = (dx * np.sin(PA) + dy * np.cos(PA)); y = (-dx * np.cos(PA) + dy * np.sin(PA)) / np.cos(INC)
    R = np.sqrt(x ** 2 + y ** 2) * 3600 / (206265 / 744)
    return {'R': R, 'x': x * 3600 / (206265 / 744),
            'y': y * 3600 / (206265 / 744), 'logR': np.log10(R + 0.1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', required=True)
    ap.add_argument('--err-floor', type=float, default=0.2)
    args = ap.parse_args()

    with h5py.File(args.h5) as f:
        need = ['ra', 'dec', 'qpah', 'qpah_err', 'CO_sigma', 'dust_density',
                'H1', 'sfr', 'CO', 'H1_ew']
        d = {k: f[k][:] for k in need if k in f}
    dust_ok = np.isfinite(d['dust_density']) & (d['dust_density'] > 0)
    print('样本 %d | dust 覆盖 %d (%.1f%%) | 其中 qpah==0 的 %d'
          % (len(d['qpah']), dust_ok.sum(), 100 * dust_ok.mean(),
             ((d['qpah'] == 0) & dust_ok).sum()))
    print('全样本 qpah==0 的 %d' % (d['qpah'] == 0).sum())
    print()

    sp = spatial(d['ra'].astype(float), d['dec'].astype(float))

    def run(tag, feats, mask, use_spa=()):
        y = d['qpah'][mask].astype(float)
        ye = d['qpah_err'][mask].astype(float)
        sig = np.sqrt(ye ** 2 + args.err_floor ** 2)
        ra = d['ra'][mask].astype(float); dec = d['dec'][mask].astype(float)
        raw = np.maximum(np.column_stack([d[k][mask] for k in feats]).astype(float), 0)
        np.random.seed(0)
        ang = (np.arctan2(dec - 41.25, ra - 10.75) + 2 * np.pi) % (2 * np.pi)
        sec = np.digitize(ang, np.linspace(0, 2 * np.pi, 11)) - 1
        tr = np.random.choice(np.arange(10), size=6, replace=False)
        in_tr = np.isin(sec, tr)
        is0 = (y == 0)
        fl = np.array([max(1e-9, 0.01 * np.median(raw[in_tr, j][raw[in_tr, j] > 0]))
                       for j in range(raw.shape[1])])
        parts = [np.log10(raw + fl[None, :])]
        for s in use_spa:
            parts.append(sp[s][mask][:, None])
        Z = np.column_stack(parts)
        mu, sd = Z[in_tr].mean(0), Z[in_tr].std(0) + 1e-12
        D = np.column_stack([np.ones(len(Z)), (Z - mu) / sd])
        w = 1.0 / sig[in_tr] ** 2
        c = np.linalg.lstsq(D[in_tr].T @ (w[:, None] * D[in_tr]),
                            D[in_tr].T @ (w * y[in_tr]), rcond=None)[0]
        pull = (D @ c - y) / sig
        pred = D @ c
        coefs = ' '.join('%s=%+.3f' % (f_, v) for f_, v in zip(list(feats) + list(use_spa), c[1:]))
        print('%-34s n=%6d | p=%2d | chi2_all=%7.4f | chi2(0值)=%8.3f | pull中位(0)=%+.3f '
              '| 预测中位(0)=%.3f | min=%.3f'
              % (tag, mask.sum(), D.shape[1], np.mean(pull ** 2),
                 np.mean(pull[is0] ** 2) if is0.sum() else np.nan,
                 np.median(pull[is0]) if is0.sum() else np.nan,
                 np.median(pred[is0]) if is0.sum() else np.nan, pred.min()))
        print('      %s' % coefs)
        return np.mean(pull[is0] ** 2) if is0.sum() else np.nan

    allm = np.ones(len(d['qpah']), bool)
    print('=== 全样本（无 dust）===')
    run('V1  base(4)  [全样本]', ['H1', 'sfr', 'CO', 'H1_ew'], allm)
    run('V2  base+R+y+x  [全样本]', ['H1', 'sfr', 'CO', 'H1_ew'], allm, ('R', 'y', 'x'))
    print()
    print('=== 限制到 dust 覆盖区（对照：先看"换样本"本身的效应）===')
    run('V3  base(4)  [dust覆盖区]', ['H1', 'sfr', 'CO', 'H1_ew'], dust_ok)
    run('V4  base+R+y+x  [dust覆盖区]', ['H1', 'sfr', 'CO', 'H1_ew'], dust_ok, ('R', 'y', 'x'))
    print()
    print('=== 加上 dust ===')
    run('V5  +dust', ['dust_density', 'H1', 'sfr', 'CO', 'H1_ew'], dust_ok)
    run('V6  +dust+R+y+x', ['dust_density', 'H1', 'sfr', 'CO', 'H1_ew'], dust_ok, ('R', 'y', 'x'))


if __name__ == '__main__':
    main()
