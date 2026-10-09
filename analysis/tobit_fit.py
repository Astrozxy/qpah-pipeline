#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tobit_fit.py — 含下界截断（qPAH 在 0 处被 DL07 的 qPAH>=0 约束截断）的拟合。

为什么需要它
------------
观测 qPAH 在 0 处有硬尖峰（约 3.7% 恰好为 0）。这不是"曲线不够弯"——把基函数
从 5 参数加到 15 参数（全二次）后，0 值像素的 chi2 只从 13.31 降到 12.53。
根因是【截断】：DL07 拟合约束 qPAH>=0，所有"真值<=0"的情形都堆在 0 这一点，
而任何光滑函数都做不出点质量。

正确写法（Tobit / 一侧截断似然）：
  未截断点 (y>0) :  L = N(y | pred, sigma)          -> -logL = (pred-y)^2/(2 sigma^2)
  截断点   (y==0):  真值 y* <= 0
                    L = P(y* <= 0) = Phi(-pred/sigma) -> -logL = -log Phi(-pred/sigma)

  NLL = (1/2) sum_uncens (pred-y)^2/sigma^2  -  sum_cens log Phi(-pred/sigma)

设计（基函数）沿用现状：pred = b + sum_j w_j * (log10(x_j+floor_j)-mean_j)/std_j
额外可加二次项（--quadratic）看是否还需要。

输出：系数、NLL、未截断点的 chi2、截断点的"越界率" P(y*<=0) 与 -logL 贡献，
      以及各口径下的预测分布。
"""
import os, argparse
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mplconfig')
import numpy as np
import h5py
from scipy.optimize import minimize
from scipy.stats import norm

FEATS = ['H1', 'sfr', 'CO', 'H1_ew']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', required=True)
    ap.add_argument('--err-floor', type=float, default=0.2)
    ap.add_argument('--quadratic', action='store_true', help='基函数额外加 z_j^2')
    ap.add_argument('--maxiter', type=int, default=3000)
    args = ap.parse_args()

    with h5py.File(args.h5) as f:
        d = {k: f[k][:] for k in ['ra', 'dec', 'qpah', 'qpah_err', 'CO_sigma'] + FEATS}
    ra, dec = d['ra'].astype(float), d['dec'].astype(float)
    X = np.maximum(np.column_stack([d[k] for k in FEATS]).astype(float), 0)
    y = d['qpah'].astype(float)
    sig = np.sqrt(d['qpah_err'].astype(float) ** 2 + args.err_floor ** 2)
    cens = (y <= 0)
    np.random.seed(0)
    ang = (np.arctan2(dec - 41.25, ra - 10.75) + 2 * np.pi) % (2 * np.pi)
    sec = np.digitize(ang, np.linspace(0, 2 * np.pi, 11)) - 1
    tr = np.random.choice(np.arange(10), size=6, replace=False)
    in_tr = np.isin(sec, tr)

    fl = np.array([max(1e-9, 0.01 * np.median(X[in_tr, j][X[in_tr, j] > 0])) for j in range(4)])
    Z = np.log10(X + fl[None, :])
    mu, sd = Z[in_tr].mean(0), Z[in_tr].std(0) + 1e-12
    Zs = (Z - mu) / sd
    cols = [np.ones(len(Zs))] + [Zs[:, j] for j in range(4)]
    if args.quadratic:
        cols += [Zs[:, j] ** 2 for j in range(4)]
    D = np.column_stack(cols)
    p = D.shape[1]
    print('样本 n=%d | 截断点(qpah<=0) %d (%.2f%%) | 基函数 p=%d'
          % (len(y), cens.sum(), 100 * cens.mean(), p))

    def nll(c):
        pr = D @ c
        r = (pr - y) / sig
        out = 0.5 * np.sum(r[~cens] ** 2)
        # 截断点: -log Phi(-pr/sig)
        zz = -pr[cens] / sig[cens]
        out += -np.sum(norm.logcdf(zz))
        return out

    # 初值：普通最小二乘
    w = 1.0 / sig[in_tr] ** 2
    A = D[in_tr].T @ (w[:, None] * D[in_tr])
    b = D[in_tr].T @ (w * y[in_tr])
    c0 = np.linalg.lstsq(A, b, rcond=None)[0]
    print('OLS 初值 NLL=%.1f' % nll(c0))
    res = minimize(nll, c0, method='Nelder-Mead',
                   options=dict(maxiter=args.maxiter, xatol=1e-4, fatol=1e-3))
    c = res.x
    print('Tobit 收敛: %s | NLL=%.1f' % (res.success, res.fun))
    print()

    for tag, cc in [('OLS  ', c0), ('Tobit', c)]:
        pr = D @ cc
        r = (pr - y) / sig
        print('%s: 系数 截距=%+.4f | ' % (tag, cc[0]) +
              '  '.join('%s=%+.4f' % (f_, v) for f_, v in zip(FEATS, cc[1:5])) +
              (('  | ' + '  '.join('%s^2=%+.4f' % (f_, v) for f_, v in zip(FEATS, cc[5:])))
               if args.quadratic else ''))
        print('        未截断点 chi2=%.4f (n=%d) | 截断点 chi2(按对称误差算，仅供参考)=%.3f'
              % (np.mean(r[~cens] ** 2), (~cens).sum(), np.mean(r[cens] ** 2)))
        print('        截断点: 预测中位=%.3f | P(y*<=0)=Phi(-pred/sig) 中位=%.3f | 其中 P<0.5 的占 %.1f%%'
              % (np.median(pr[cens]), np.median(norm.cdf(-pr[cens] / sig[cens])),
                 100 * np.mean(norm.cdf(-pr[cens] / sig[cens]) < 0.5)))
        print('        全体对称 chi2=%.4f | min预测=%.3f' % (np.mean(r ** 2), pr.min()))
        print()


if __name__ == '__main__':
    main()
