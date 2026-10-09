#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
model_design.py — 目标含 qPAH==0（下界堆积）时，哪种【设计】能真正够到 0？

背景：观测 qPAH 分布在 0 处有硬尖峰（约 3.7% 恰好为 0，4.9% < 0.5）。
低端像素在参数空间里很突出（sfr z=-1.26、H1 z=-0.71、CO z=-0.54、H1_ew z=-0.72）。
现有的 log10 线性模型【设计上】能输出 0 甚至负值，但拟合不愿去。

本脚本比较若干设计（都是线性最小二乘，只是基函数不同），重点看：
  - 全体 chi2
  - **在 qpah==0 像素上**的 chi2 / pull 中位 / 预测中位
  - 预测最小值，以及有多少 0 值像素被预测到 <0.5%

设计：
  D1 线性（现状）        : [1, z_j]                      p=5
  D2 线性+平方           : [1, z_j, z_j^2]               p=9
  D3 线性+全二次         : [1, z_j, z_j^2, z_j z_k]      p=15
  D4 原始量线性（不取 log）: [1, x_j]                     p=5
  D5 原始量+平方          : [1, x_j, x_j^2]               p=9
"""
import os, argparse
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mplconfig')
import numpy as np
import h5py

FEATS = ['H1', 'sfr', 'CO', 'H1_ew']


def basis(Zs, Xr, kind):
    n, k = Zs.shape
    cols = [np.ones(n)]
    if kind == 'D1':
        cols += [Zs[:, j] for j in range(k)]
    elif kind == 'D2':
        cols += [Zs[:, j] for j in range(k)]
        cols += [Zs[:, j] ** 2 for j in range(k)]
    elif kind == 'D3':
        cols += [Zs[:, j] for j in range(k)]
        cols += [Zs[:, j] ** 2 for j in range(k)]
        for a in range(k):
            for b in range(a + 1, k):
                cols.append(Zs[:, a] * Zs[:, b])
    elif kind == 'D4':
        cols += [Xr[:, j] for j in range(k)]
    elif kind == 'D5':
        cols += [Xr[:, j] for j in range(k)]
        cols += [Xr[:, j] ** 2 for j in range(k)]
    return np.column_stack(cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', required=True)
    ap.add_argument('--err-floor', type=float, default=0.2)
    args = ap.parse_args()

    with h5py.File(args.h5) as f:
        d = {k: f[k][:] for k in ['ra', 'dec', 'qpah', 'qpah_err', 'CO_sigma'] + FEATS}
    ra, dec = d['ra'].astype(float), d['dec'].astype(float)
    X = np.maximum(np.column_stack([d[k] for k in FEATS]).astype(float), 0)
    y = d['qpah'].astype(float)
    sig = np.sqrt(d['qpah_err'].astype(float) ** 2 + args.err_floor ** 2)
    is0 = (y == 0)
    np.random.seed(0)
    ang = (np.arctan2(dec - 41.25, ra - 10.75) + 2 * np.pi) % (2 * np.pi)
    sec = np.digitize(ang, np.linspace(0, 2 * np.pi, 11)) - 1
    tr = np.random.choice(np.arange(10), size=6, replace=False)
    in_tr = np.isin(sec, tr)
    print('样本 n=%d | qpah==0 %d (%.2f%%) | qpah<0.5 %.2f%%'
          % (len(y), is0.sum(), 100 * is0.mean(), 100 * np.mean(y < 0.5)))
    print()
    hdr = '%-6s %5s %9s %9s %10s %11s %11s %11s %10s' % (
        '设计', 'p', 'chi2_tr', 'chi2_te', 'chi2_all', 'chi2(0值)', 'pull中位(0)', '预测中位(0)', 'min预测')
    print(hdr); print('-' * len(hdr))

    fl = np.array([max(1e-9, 0.01 * np.median(X[in_tr, j][X[in_tr, j] > 0])) for j in range(4)])
    Z = np.log10(X + fl[None, :])
    mu, sd = Z[in_tr].mean(0), Z[in_tr].std(0) + 1e-12
    Zs = (Z - mu) / sd

    for kind, name in [('D1', '线性(现状)'), ('D2', '线性+平方'), ('D3', '线性+全二次'),
                       ('D4', '原始量线性'), ('D5', '原始量+平方')]:
        D = basis(Zs, X, kind)
        w = 1.0 / sig[in_tr] ** 2
        A = D[in_tr].T @ (w[:, None] * D[in_tr])
        b = D[in_tr].T @ (w * y[in_tr])
        c = np.linalg.lstsq(A, b, rcond=None)[0]
        pull = (D @ c - y) / sig
        pred = D @ c
        print('%-6s %5d %9.4f %9.4f %10.4f %11.4f %11.3f %11.3f %10.3f'
              % (name, D.shape[1], np.mean(pull[in_tr] ** 2), np.mean(pull[~in_tr] ** 2),
                 np.mean(pull ** 2), np.mean(pull[is0] ** 2), np.median(pull[is0]),
                 np.median(pred[is0]), pred.min()))
    print()
    print('参考：qpah>0 的观测 p1=%.2f p50=%.2f ；0 值像素的 qpah_err 中位=%.3f'
          % (np.percentile(y[~is0], 1), np.median(y[~is0]), np.median(d['qpah_err'][is0])))


if __name__ == '__main__':
    main()
