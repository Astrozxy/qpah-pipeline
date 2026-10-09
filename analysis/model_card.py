#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
model_card.py — 把当前模型【显式写出来】，并做"把 dust 加回来能否解释那条悬臂"的判决实验。

显式形式（log10 线性）：
    pred = b + Σⱼ wⱼ · ( log10(max(xⱼ,0) + floorⱼ) − meanⱼ ) / stdⱼ
等价地（原始单位、幂律形式）：
    qPAH = A + Σⱼ aⱼ · log10(max(xⱼ,0) + floorⱼ),   aⱼ = wⱼ/stdⱼ,
           A = b − Σⱼ aⱼ·meanⱼ

判决实验：该悬臂（blob 链 35-38-40-46-51）落在 SCUBA-2 覆盖区内（coverage=1.000），
所以可以把两个模型锁在【同一批 dust 有限像素】上公平比较：
    A：4 特征（H1, sfr, CO, H1_ew）   —— 当前模型
    B：4 特征 + dust（850μm）          —— dust 能否解释臂上的缺口

用法:
  python model_card.py --h5 data/M31_1d_array_nodust.h5 \
     --features H1 sfr CO H1_ew \
     --mask-npz results/armregion/mask_arm_35_38_40_46_51.npz --add-dust
"""
import argparse
import numpy as np
import h5py

RA0, DEC0, INC, PA = 10.6847, 41.2690, np.radians(77.7), np.radians(37.7)


def deproj_R(ra, dec):
    dx = (ra - RA0) * np.cos(np.radians(DEC0)); dy = dec - DEC0
    x = dx * np.sin(PA) + dy * np.cos(PA)
    y = (-dx * np.cos(PA) + dy * np.sin(PA)) / np.cos(INC)
    return np.sqrt(x ** 2 + y ** 2) * 3600.0 / (206265.0 / 744.0)


def matched_delta(mask, y, pull, R, lo=6.0, hi=14.0, step=0.5):
    """同半径环匹配：返回 (ΔqPAH, Δpull) 的中位数（区域 − 对照）。"""
    dq, dp = [], []
    for a in np.arange(lo, hi, step):
        m1 = mask & (R >= a) & (R < a + step)
        m2 = (~mask) & (R >= a) & (R < a + step)
        if m1.sum() >= 10 and m2.sum() >= 50:
            dq.append(np.median(y[m1]) - np.median(y[m2]))
            dp.append(np.median(pull[m1]) - np.median(pull[m2]))
    return (np.median(dq) if dq else np.nan), (np.median(dp) if dp else np.nan)


def report_card(name, feats, Z, floors, coef, mu, sd, y, sig, in_tr, R, region):
    b, w = coef[0], coef[1:]
    a = w / sd
    A = b - np.sum(a * mu)
    pred = np.column_stack([np.ones(len(Z)), (Z - mu) / sd]) @ coef
    pull = (pred - y) / sig
    print('=' * 86)
    print('模型 %s   (%d 个特征)' % (name, len(feats)))
    print('=' * 86)
    print('  特征 xⱼ        : ' + ', '.join(feats))
    print('  预处理         : x → max(x, 0)，再 log10(x + floorⱼ)')
    print('  floorⱼ（偏移） : ' + '  '.join('%s=%.5g' % (f, v) for f, v in zip(feats, floors)))
    print('  meanⱼ（训练集）: ' + '  '.join('%s=%.4f' % (f, v) for f, v in zip(feats, mu)))
    print('  stdⱼ （训练集）: ' + '  '.join('%s=%.4f' % (f, v) for f, v in zip(feats, sd)))
    print('  标准化系数 wⱼ  : ' + '  '.join('%s=%+.4f' % (f, v) for f, v in zip(feats, w)))
    print('  截距 b         : %+.4f' % b)
    print()
    print('  【显式方程·原始单位】 qPAH(%) = A + Σ aⱼ·log10(xⱼ + floorⱼ)')
    print('      A = %+.4f' % A)
    for f, av, fv in zip(feats, a, floors):
        print('      a(%-12s) = %+.4f     ← log10(x + %.5g)' % (f, av, fv))
    print()
    print('  指标: 训练 chi2=%.4f (n=%d) | 测试 chi2=%.4f (n=%d) | 全部 chi2=%.4f'
          % (np.mean(pull[in_tr] ** 2), in_tr.sum(),
             np.mean(pull[~in_tr] ** 2), (~in_tr).sum(), np.mean(pull ** 2)))
    if region is not None and region.sum():
        dq, dp = matched_delta(region, y, pull, R)
        print('  ▶ 悬臂 (n=%d): 观测 qPAH 中位=%.2f%% | 预测中位=%.2f%% | pull 中位=%+.3f | chi2=%.3f'
              % (region.sum(), np.median(y[region]), np.median(pred[region]),
                 np.median(pull[region]), np.mean(pull[region] ** 2)))
        print('     同半径匹配: ΔqPAH(区域−对照)=%+.3f%% | Δpull=%+.3f' % (dq, dp))
    return pull


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', default='data/M31_1d_array_nodust.h5')
    ap.add_argument('--features', nargs='+', default=['H1', 'sfr', 'CO', 'H1_ew'])
    ap.add_argument('--err-floor', type=float, default=0.2)
    ap.add_argument('--floor-frac', type=float, default=0.01)
    ap.add_argument('--mask-npz', default=None)
    ap.add_argument('--add-dust', action='store_true')
    args = ap.parse_args()

    with h5py.File(args.h5, 'r') as f:
        keys = list(dict.fromkeys(['ra', 'dec', 'qpah', 'qpah_err', 'CO_sigma',
                                   'dust_density'] + args.features))
        d = {k: f[k][:] for k in keys if k in f}
    print('数据集      : %s' % args.h5)
    print('筛选口径    : qpah>0 & qpah_err>0 & 各特征有限 & H1≥0 & H1_ew≥0 & CO_sigma>0')
    print('CO 预处理   : 裁剪到 ≥0（负 CO 非物理）')
    print()

    base = ((d['qpah'] >= 0) & (d['qpah_err'] > 0)
            & np.isfinite(d['H1']) & (d['H1'] >= 0)
            & np.isfinite(d['H1_ew']) & (d['H1_ew'] >= 0)
            & np.isfinite(d['sfr']) & np.isfinite(d['CO'])
            & np.isfinite(d['CO_sigma']) & (d['CO_sigma'] > 0))

    region_full = None
    if args.mask_npz:
        zm = np.load(args.mask_npz)
        if len(zm['region']) != int(base.sum()):
            raise SystemExit('mask 长度 %d != 基础样本 %d' % (len(zm['region']), base.sum()))
        # mask 是在【base 筛选后】的子集上算的 → 先摊回全集索引
        region_full = np.zeros(len(base), bool)
        region_full[base] = zm['region'].astype(bool)
        print('mask 载入: %d 像素（占基础样本 %.2f%%）\n'
              % (region_full.sum(), 100 * region_full[base].mean()))

    def run(feats, extra, tag):
        sel = base & extra
        ra, dec = d['ra'][sel].astype(float), d['dec'][sel].astype(float)
        X = np.maximum(np.column_stack([d[k][sel] for k in feats]).astype(float), 0.0)
        y = d['qpah'][sel].astype(float)
        sig = np.sqrt(d['qpah_err'][sel].astype(float) ** 2 + args.err_floor ** 2)
        R = deproj_R(ra, dec)
        region = region_full[sel] if region_full is not None else None
        np.random.seed(0)
        ang = (np.arctan2(dec - 41.25, ra - 10.75) + 2 * np.pi) % (2 * np.pi)
        sec = np.digitize(ang, np.linspace(0, 2 * np.pi, 11)) - 1
        tr_s = np.random.choice(np.arange(10), size=6, replace=False)
        in_tr = np.isin(sec, tr_s)
        floors = np.array([max(1e-9, args.floor_frac * np.median(X[in_tr, j][X[in_tr, j] > 0]))
                           for j in range(X.shape[1])])
        Z = np.log10(X + floors[None, :])
        mu, sd = Z[in_tr].mean(0), Z[in_tr].std(0) + 1e-12
        D = np.column_stack([np.ones(len(Z)), (Z - mu) / sd])
        w = 1.0 / sig[in_tr] ** 2
        coef = np.linalg.solve(D[in_tr].T @ (w[:, None] * D[in_tr]), D[in_tr].T @ (w * y[in_tr]))
        report_card(tag, feats, Z, floors, coef, mu, sd, y, sig, in_tr, R, region)

    if args.add_dust:
        dust_ok = np.isfinite(d['dust_density']) & (d['dust_density'] > 0)
        print('### 锁死样本：只用 dust 有限像素（%d 个）比较两个模型 ###' % dust_ok.sum())
        print()
        run(args.features, dust_ok, 'A  4 特征（当前模型）')
        print()
        run(args.features + ['dust_density'], dust_ok, 'B  4 特征 + dust(850μm)')
    else:
        run(args.features, np.ones(len(d['qpah']), bool), 'current')


if __name__ == '__main__':
    main()
