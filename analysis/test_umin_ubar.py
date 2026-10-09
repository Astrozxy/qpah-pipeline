#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_umin_ubar.py — 加上 U_min / U_bar 能否让模型碰到 qPAH=0？

数据来源：analysis/export_umin_ubar.py 在服务器上把两张 U 图对齐到 qPAH 网格后，
        导出的【每行值】 npz（行序与 --h5 完全一致，未做分析筛选）。

对每个特征集报：全体 chi2、**qpah==0 像素上的 chi2 / pull 中位 / 预测中位**、
预测最小值、以及各系数。同时报 U 图的覆盖率（它可能比 dust 更宽或更窄）。

用法:
  python test_umin_ubar.py --h5 .../M31_1d_array_nodust_qe2.h5 \
      --u .../umin_ubar_rows.npz
"""
import os, argparse
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mplconfig')
import numpy as np
import h5py

BASE = ['H1', 'sfr', 'CO', 'H1_ew']
RA0, DEC0, INC, PA = 10.6847, 41.2690, np.radians(77.7), np.radians(37.7)


def spatial(ra, dec):
    dx = (ra - RA0) * np.cos(np.radians(DEC0)); dy = dec - DEC0
    x = dx * np.cos(PA) + dy * np.sin(PA)
    y = (-dx * np.sin(PA) + dy * np.cos(PA)) / np.cos(INC)
    k = 3600.0 / (206265.0 / 744.0)
    R = np.sqrt(x ** 2 + y ** 2) * k
    return {'R': R, 'x': x * k, 'y': y * k, 'logR': np.log10(R + 0.1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', required=True)
    ap.add_argument('--u', required=True, help='umin_ubar_rows.npz')
    ap.add_argument('--err-floor', type=float, default=0.2)
    ap.add_argument('--zero-co-snr', type=float, default=None,
                    help='可选：CO/σ <= 该值时把 CO 置 0')
    args = ap.parse_args()

    with h5py.File(args.h5) as f:
        need = ['ra', 'dec', 'qpah', 'qpah_err', 'CO_sigma'] + BASE
        d = {k: f[k][:] for k in need}
    n = len(d['qpah'])
    zu = np.load(args.u)
    uk = [k for k in zu.files if not k.endswith('_cov')]
    print('样本 %d | U 图: %s' % (n, uk))
    for k in uk:
        v = zu[k]
        if len(v) != n:
            raise SystemExit('U 图行数 %d != h5 行数 %d（导出时用的 h5 不一致）' % (len(v), n))
        print('  %-6s 有效 %d (%.1f%%) | 中位=%.4g | cov 中位=%.3f'
              % (k, np.isfinite(v).sum(), 100 * np.isfinite(v).mean(),
                 np.nanmedian(v), np.median(zu.get(k + '_cov', np.ones(n)))))

    ra, dec = d['ra'].astype(float), d['dec'].astype(float)
    sp = spatial(ra, dec)
    np.random.seed(0)
    ang = (np.arctan2(dec - 41.25, ra - 10.75) + 2 * np.pi) % (2 * np.pi)
    sec = np.digitize(ang, np.linspace(0, 2 * np.pi, 11)) - 1
    tr = np.random.choice(np.arange(10), size=6, replace=False)
    in_tr = np.isin(sec, tr)
    X = np.maximum(np.column_stack([d[k] for k in BASE]).astype(float), 0)
    y = d['qpah'].astype(float)
    sig = np.sqrt(d['qpah_err'].astype(float) ** 2 + args.err_floor ** 2)
    if args.zero_co_snr is not None:
        snr = X[:, 2] / np.maximum(d['CO_sigma'].astype(float), 1e-30)
        X[snr <= args.zero_co_snr, 2] = 0.0

    print()
    hdr = '%-30s %6s %5s %11s %11s %11s %11s %9s' % (
        '特征集', 'n', 'p', 'chi2_all', 'chi2(0值)', 'pull中位(0)', '预测中位(0)', 'min预测')
    print(hdr); print('-' * len(hdr))

    def run(tag, cols, use_all=True):
        ok = np.ones(len(y), bool)
        for c in cols:
            ok &= np.isfinite(c)
        if ok.sum() < 500:
            print('%-30s  有效样本不足 (%d)' % (tag, ok.sum())); return
        yy, ss, ii = y[ok], sig[ok], in_tr[ok]
        is0 = (yy == 0)
        raw = np.column_stack([c[ok] for c in cols])
        raw = np.maximum(raw, 0)
        fl = np.array([max(1e-9, 0.01 * np.median(raw[ii, j][raw[ii, j] > 0]))
                       for j in range(raw.shape[1])])
        Z = np.log10(raw + fl[None, :])
        for s in SPASEL:
            Z = np.column_stack([Z, sp[s][ok]])
        mu, sd = Z[ii].mean(0), Z[ii].std(0) + 1e-12
        D = np.column_stack([np.ones(len(Z)), (Z - mu) / sd])
        w = 1.0 / ss[ii] ** 2
        c = np.linalg.lstsq(D[ii].T @ (w[:, None] * D[ii]), D[ii].T @ (w * yy[ii]), rcond=None)[0]
        pull = (D @ c - yy) / ss
        pred = D @ c
        print('%-30s %6d %5d %11.4f %11.4f %11.3f %11.3f %9.3f'
              % (tag, ok.sum(), D.shape[1], np.mean(pull ** 2),
                 np.mean(pull[is0] ** 2) if is0.sum() else np.nan,
                 np.median(pull[is0]) if is0.sum() else np.nan,
                 np.median(pred[is0]) if is0.sum() else np.nan, pred.min()))
        print('      ' + '  '.join('%s=%+.4f' % (nm, v) for nm, v in
                                   zip(NAMES, c[1:])))

    SPASEL = []
    NAMES = BASE
    run('V0 base(4)', [X[:, j] for j in range(4)])
    for k in uk:
        NAMES = BASE + [k]
        run('V+%s' % k, [X[:, j] for j in range(4)] + [zu[k]])
    if len(uk) == 2:
        NAMES = BASE + uk
        run('V+both', [X[:, j] for j in range(4)] + [zu[k] for k in uk])
        SPASEL = ['R', 'y', 'x']
        NAMES = BASE + uk + SPASEL
        run('V+both+R+y+x', [X[:, j] for j in range(4)] + [zu[k] for k in uk])
        SPASEL = []
    NAMES = BASE + ['R', 'y', 'x']
    SPASEL = ['R', 'y', 'x']
    run('V0+R+y+x (对照)', [X[:, j] for j in range(4)])
    SPASEL = []


if __name__ == '__main__':
    main()
