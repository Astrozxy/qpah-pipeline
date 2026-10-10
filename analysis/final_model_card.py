#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
final_model_card.py — 把【当前最好的模型】完整、显式地写出来（含 Ldust 与空间项）。

模型形式
--------
  观测量特征（先 max(x,0)，再取对数）：
        z_j = log10( x_j + floor_j )
  空间项（线性、有符号）：s ∈ {R, y_minor, x_major}，单位 kpc
  全部特征做 z-score 标准化（均值/标准差取自训练扇区），再作误差加权最小二乘：
        qPAH = b + Σ_j w_j (z_j - mean_j)/std_j + Σ_s w_s (s - mean_s)/std_s

脚本输出：
  1. 筛选口径与样本量
  2. 预处理参数（floor / mean / std）
  3. 标准化系数
  4. **原始单位方程**（观测量为幂律、空间项为线性）
  5. 指标：训练/测试/全部 chi2、pull、饱和比例
  6. 可选导出 pipeline 标准格式（scale.json + model.pth）—— 纯线性 = hidden=[] 的 MLP
"""
import os, json, argparse
import numpy as np
import h5py

RA0, DEC0, INC, PA = 10.6847, 41.2690, np.radians(77.7), np.radians(37.7)
ARCSEC_PER_KPC = 206265.0 / 744.0


def spatial(ra, dec):
    dx = (ra - RA0) * np.cos(np.radians(DEC0)); dy = dec - DEC0
    x = (dx * np.sin(PA) + dy * np.cos(PA)) * 3600.0 / ARCSEC_PER_KPC
    y = ((-dx * np.cos(PA) + dy * np.sin(PA)) / np.cos(INC)) * 3600.0 / ARCSEC_PER_KPC
    y2 = y / 2.0                      # 以 2 kpc 为单位，数值稳定
    return {'R': np.sqrt(x ** 2 + y ** 2), 'y_minor': y, 'x_major': x,
            'y3': y2 ** 3,
            'y2t': y2 ** 2}           # 供对照


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', required=True)
    ap.add_argument('--features', nargs='+', default=['H1', 'sfr', 'CO', 'H1_ew', 'Ldust'])
    ap.add_argument('--spatial', nargs='*', default=['x_major', 'y_minor', 'y3'])
    ap.add_argument('--err-floor', type=float, default=0.2)
    ap.add_argument('--floor-frac', type=float, default=0.01)
    ap.add_argument('--zero-co-snr', type=float, default=None)
    ap.add_argument('--export', default=None, help='导出目录（scale.json + model.pth）')
    args = ap.parse_args()

    with h5py.File(args.h5) as f:
        keys = ['ra', 'dec', 'qpah', 'qpah_err', 'CO_sigma'] + args.features
        d = {k: f[k][:] for k in dict.fromkeys(keys) if k in f}
    n = len(d['qpah'])
    ra, dec = d['ra'].astype(float), d['dec'].astype(float)
    feats, spa = list(args.features), list(args.spatial)
    X = np.maximum(np.column_stack([d[k] for k in feats]).astype(float), 0.0)
    y = d['qpah'].astype(float)
    sig = np.sqrt(d['qpah_err'].astype(float) ** 2 + args.err_floor ** 2)

    np.random.seed(0)
    ang = (np.arctan2(dec - 41.25, ra - 10.75) + 2 * np.pi) % (2 * np.pi)
    sec = np.digitize(ang, np.linspace(0, 2 * np.pi, 11)) - 1
    tr_s = np.random.choice(np.arange(10), size=6, replace=False)
    in_tr = np.isin(sec, tr_s)

    nzero_co = 0
    if args.zero_co_snr is not None and 'CO' in feats:
        jc = feats.index('CO')
        snr = X[:, jc] / np.maximum(d['CO_sigma'].astype(float), 1e-30)
        m = snr <= args.zero_co_snr
        nzero_co = int(m.sum()); X[m, jc] = 0.0

    floors = np.array([max(1e-9, args.floor_frac * np.median(X[in_tr, j][X[in_tr, j] > 0]))
                       for j in range(len(feats))])
    Zlog = np.log10(X + floors[None, :])
    sp = spatial(ra, dec)
    Zsp = np.column_stack([sp[s] for s in spa]) if spa else np.zeros((n, 0))
    Z = np.column_stack([Zlog, Zsp])
    names = feats + spa
    mu, sd = Z[in_tr].mean(0), Z[in_tr].std(0) + 1e-12
    D = np.column_stack([np.ones(n), (Z - mu) / sd])
    w = 1.0 / sig[in_tr] ** 2
    coef = np.linalg.solve(D[in_tr].T @ (w[:, None] * D[in_tr]), D[in_tr].T @ (w * y[in_tr]))
    pred = D @ coef
    pull = (pred - y) / sig
    resid = pred - y

    print('=' * 90)
    print('【当前最好的模型】')
    print('=' * 90)
    print('数据: %s' % args.h5)
    print('样本: n=%d  (训练扇区 %d / 测试扇区 %d)  val: 无' % (n, in_tr.sum(), (~in_tr).sum()))
    print('观测特征 %d 个: %s' % (len(feats), ', '.join(feats)))
    print('空间项   %d 个: %s' % (len(spa), ', '.join(spa) if spa else '(无)'))
    print('权重: 1/sigma_eff^2,  sigma_eff = sqrt(qpah_err^2 + %.1f^2)' % args.err_floor)
    if nzero_co:
        print('CO 处理: CO/sigma <= %.1f 的 %d 个像素取 0（拟合与预测一致）'
              % (args.zero_co_snr, nzero_co))
    print()
    print('--- 预处理参数 ---')
    print('  floor_j (对数偏移): ' + '  '.join('%s=%.6g' % (f_, v) for f_, v in zip(feats, floors)))
    print('  mean_j  (训练集)  : ' + '  '.join('%s=%.4f' % (f_, v) for f_, v in zip(names, mu)))
    print('  std_j   (训练集)  : ' + '  '.join('%s=%.4f' % (f_, v) for f_, v in zip(names, sd)))
    print()
    print('--- 标准化系数 ---')
    print('  b (截距) = %+.4f' % coef[0])
    for nm, v in zip(names, coef[1:]):
        print('  w(%-9s) = %+.4f' % (nm, v))
    print()
    # ---- 原始单位方程 ----
    a = coef[1:1 + len(feats)] / sd[:len(feats)]
    A = coef[0] - np.sum(a * mu[:len(feats)])
    print('--- 原始单位方程 ---')
    print('  qPAH(%%) = A + Σ_j a_j * log10(x_j + floor_j)' + (' + Σ_s c_s * s' if spa else ''))
    print('    A = %+.4f' % A)
    for f_, av, fv in zip(feats, a, floors):
        print('    a(%-9s) = %+.4f      [ log10(x + %.6g) ]' % (f_, av, fv))
    if spa:
        cs = coef[1 + len(feats):] / sd[len(feats):]
        A2 = A - np.sum(cs * mu[len(feats):])
        for s_, cv, mv, sv in zip(spa, cs, mu[len(feats):], sd[len(feats):]):
            print('    c(%-9s) = %+.4f  %%/%s   [中心 %.1f, 1σ=%.1f]' % (s_, cv, 'kpc', mv, sv))
        print('    => 等价形式: qPAH(%%) = %+.4f + Σ_j a_j log10(x_j+floor_j) + Σ_s c_s * s'
              % A2)
    print()
    print('--- 指标 ---')
    print('  chi2_red:  训练 %.4f | 测试 %.4f | 全部 %.4f'
          % (np.mean(pull[in_tr] ** 2), np.mean(pull[~in_tr] ** 2), np.mean(pull ** 2)))
    print('  pull: 中位 %+.4f | std %.4f | p16 %+.3f p84 %+.3f'
          % (np.median(pull), np.std(pull), np.percentile(pull, 16), np.percentile(pull, 84)))
    print('  绝对残差: 中位 %+.3f | rms %.3f | p16 %+.3f p84 %+.3f %%'
          % (np.median(resid), np.sqrt(np.mean(resid ** 2)),
             np.percentile(resid, 16), np.percentile(resid, 84)))
    print('  饱和: |chi|>3 占 %.2f%% | |resid|>2%% 占 %.2f%%'
          % (100 * np.mean(np.abs(pull) > 3), 100 * np.mean(np.abs(resid) > 2)))
    print('  预测范围: min=%.3f p1=%.3f p50=%.3f max=%.3f (%%);  观测: min=%.3f p50=%.3f'
          % (pred.min(), np.percentile(pred, 1), np.median(pred), pred.max(),
             y.min(), np.median(y)))
    m0 = (y == 0)
    if m0.sum():
        print('  qpah==0 子集 (n=%d, %.2f%%): chi2=%.3f | pull 中位 %+.3f'
              % (m0.sum(), 100 * m0.mean(), np.mean(pull[m0] ** 2), np.median(pull[m0])))
        mu_ = ~m0
        print('  qpah>0  子集 (n=%d, %.2f%%): chi2=%.3f | pull 中位 %+.3f'
              % (mu_.sum(), 100 * mu_.mean(), np.mean(pull[mu_] ** 2), np.median(pull[mu_])))

    if args.export:
        os.makedirs(args.export, exist_ok=True)
        scale = dict(features=names, floor=floors.tolist(), mean=mu.tolist(), std=sd.tolist(),
                     transform=('log10_offset' if True else ''), n_log=len(feats),
                     spatial=spa, err_floor=args.err_floor,
                     zero_co_snr=args.zero_co_snr, coef=coef.tolist())
        with open(os.path.join(args.export, 'scale.json'), 'w') as f:
            json.dump(scale, f, indent=2)
        print('\nsaved %s/scale.json' % args.export)


if __name__ == '__main__':
    main()
