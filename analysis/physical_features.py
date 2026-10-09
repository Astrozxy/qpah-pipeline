#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
physical_features.py — 在 log10 线性框架下引入【物理量】，并对比不同特征集。

背景
----
基线（5 个观测量、log 线性、误差加权）已经打到 χ²_red = 1.0105 ± 0.023，
即已经解释到观测误差允许的极限。所以本脚本的目的不是"降 χ²"，而是：

  (a) 物理可解释性：用 gas/dust、分子比、恒星形成效率等替换/补充观测量；
  (b) 外推稳健性；
  (c) 摆脱覆盖率瓶颈：dust_density（SCUBA-2 850μm）系数≈0，却只覆盖网格 11.2%
      （次低的 CO 有 26.3%）——若去掉它不掉精度，可用面积能涨 ~3 倍。

**只用独立观测量构造物理量**（H1 / CO / SCUBA-2 850μm / SFR / H1_ew），
刻意不使用 DL07 自己的输出（U_bar、SurfBr_Mdust…），避免与 qPAH 标签循环论证。

约定（与 interpret.py 一致）：
  Σ_HI  = ALPHA_HI · I_HI      [M☉/pc²],  ALPHA_HI = 1.823e18 · 8.00e-21
  Σ_H2  = X_CO · I_CO          [M☉/pc²],  X_CO     = 2.0e20 · 8.00e-21 · 2

用法:
  python qpah_pipeline/analysis/physical_features.py --h5 data/M31_1d_array_full.h5
"""
import os, json, argparse
import numpy as np
import h5py

C_MASS = 8.00e-21
ALPHA_HI = 1.823e18 * C_MASS          # 0.014584 M☉/pc² per K km/s
X_CO = 2.0e20 * C_MASS * 2.0          # 3.2      M☉/pc² per K km/s
EPS = 1e-6
BEAM_ARCSEC, PIX_ARCSEC = 24.9, 10.0

RAW = ['dust_density', 'H1', 'sfr', 'CO', 'H1_ew']


def _med_pos(x):
    p = x[x > 0]
    return float(np.median(p)) if p.size else 1.0


def add_physical(d):
    """由观测量构造物理量（全部 ≥0，便于 log 变换）。

    分母都用【该量的 1% 中位正值】做 floor，而不是 1e-6 —— 否则 dust 很小时
    gas_dust = Σ_gas/dust 会爆炸，把 log 线性拟合带偏。
    """
    H1 = np.maximum(d['H1'].astype(np.float64), 0.0)
    CO = np.maximum(d['CO'].astype(np.float64), 0.0)
    dust = np.maximum(d['dust_density'].astype(np.float64), 0.0)   # 注意：单位是 mJy/beam(850μm)
    sfr = np.maximum(d['sfr'].astype(np.float64), 0.0)
    H1ew = np.maximum(d['H1_ew'].astype(np.float64), 0.0)

    Sig_HI = ALPHA_HI * H1
    Sig_H2 = X_CO * CO
    Sig_gas = Sig_HI + Sig_H2
    f_dust, f_gas, f_h1 = (0.01 * _med_pos(v) for v in (dust, Sig_gas, H1))

    out = dict(d)
    # 原始特征也覆盖成裁剪后的版本（SFR/CO 有负值，否则 log10 会出 NaN）
    out['H1'], out['CO'] = H1, CO
    out['dust_density'], out['sfr'], out['H1_ew'] = dust, sfr, H1ew
    out['Sig_HI'] = Sig_HI
    out['Sig_H2'] = Sig_H2
    out['Sig_gas'] = Sig_gas
    # 气体 / 尘埃（分母是 850μm 流量，非尘埃质量 ⇒ 严格说是"每单位 850μm 的气体"，
    # 若尘埃温度变化不大则 ∝ 真实的 gas/dust）
    out['gas_dust'] = Sig_gas / (dust + f_dust)
    out['f_mol'] = Sig_H2 / (Sig_gas + f_gas)      # 分子气体占比
    out['sfe'] = sfr / (Sig_gas + f_gas)           # 恒星形成效率 SFR/Σ_gas
    out['co_hi'] = CO / (H1 + f_h1)
    out['ew_per_H1'] = H1ew / (H1 + f_h1)
    return out


SETS = {
    'A  baseline(5)':      ['dust_density', 'H1', 'sfr', 'CO', 'H1_ew'],
    'B  +gas_dust(6)':     ['dust_density', 'H1', 'sfr', 'CO', 'H1_ew', 'gas_dust'],
    'C  physical(3)':      ['gas_dust', 'sfe', 'f_mol'],
    'D  no-dust(4)':       ['H1', 'sfr', 'CO', 'H1_ew'],
    'E  gas-based(5)':     ['dust_density', 'Sig_gas', 'sfr', 'f_mol', 'H1_ew'],
    'F  +mol,sfe(7)':      ['dust_density', 'H1', 'sfr', 'CO', 'H1_ew', 'f_mol', 'sfe'],
    'G  H1,CO only(2)':    ['H1', 'CO'],
    'H  gas,gasdust(3)':   ['Sig_gas', 'gas_dust', 'sfr'],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', default='data/M31_1d_array_full.h5')
    ap.add_argument('--err-floor', type=float, default=0.2)
    ap.add_argument('--snr-thresh', type=float, default=3.0)
    ap.add_argument('--floor-frac', type=float, default=0.01)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    with h5py.File(args.h5, 'r') as f:
        keys = ['ra', 'dec', 'qpah', 'qpah_err', 'CO_sigma'] + RAW
        d = {k: f[k][:] for k in keys if k in f}
    sel = ((d['qpah'] >= 0) & (d['qpah_err'] > 0) & (d['dust_density'] > 0)
           & (d['H1'] >= 0) & (d['H1_ew'] >= 0)
           & np.isfinite(d['CO']) & np.isfinite(d['CO_sigma']) & (d['CO_sigma'] > 0))
    d = {k: v[sel] for k, v in d.items()}
    d = add_physical(d)
    ra, dec = d['ra'].astype(np.float64), d['dec'].astype(np.float64)
    y = d['qpah'].astype(np.float64)
    sig = np.sqrt(d['qpah_err'].astype(np.float64) ** 2 + args.err_floor ** 2)
    co_sig = np.maximum(d['CO_sigma'].astype(np.float64), 1e-9)

    np.random.seed(0)
    ang = (np.arctan2(dec - 41.25, ra - 10.75) + 2 * np.pi) % (2 * np.pi)
    sec = np.digitize(ang, np.linspace(0, 2 * np.pi, 11)) - 1
    tr_s = np.random.choice(np.arange(10), size=6, replace=False)
    te_s = np.setdiff1d(np.arange(10), tr_s)
    in_tr = np.isin(sec, tr_s)

    redund = 1.133 * BEAM_ARCSEC ** 2 / PIX_ARCSEC ** 2
    n_eff = (~in_tr).sum() / redund
    chi2_se = float(np.sqrt(2.0 / n_eff))
    print('=' * 84)
    print('训练扇区全部像素 %d | 测试 %d（有效独立 ≈%.0f） | χ² 1σ 误差 ±%.3f'
          % (in_tr.sum(), (~in_tr).sum(), n_eff, chi2_se))
    print('=' * 84)
    print('%-20s %3s %9s %9s %9s %7s' % ('特征集', 'p', 'chi2_red', 'pull_med', 'pull_std', 'rms%'))
    print('-' * 84)

    res = {}
    for name, feats in SETS.items():
        X = np.column_stack([d[f] for f in feats]).astype(np.float64)
        floors = np.array([max(1e-9, args.floor_frac * np.median(X[in_tr, j][X[in_tr, j] > 0]))
                           for j in range(X.shape[1])])
        Z = np.log10(X + floors[None, :])
        mu, sd = Z[in_tr].mean(0), Z[in_tr].std(0) + 1e-12
        Ztr = np.column_stack([np.ones(int(in_tr.sum())), (Z[in_tr] - mu) / sd])
        Zte = np.column_stack([np.ones(int((~in_tr).sum())), (Z[~in_tr] - mu) / sd])
        w = 1.0 / sig[in_tr] ** 2
        coef = np.linalg.solve(Ztr.T @ (w[:, None] * Ztr), Ztr.T @ (w * y[in_tr]))
        pred = Zte @ coef
        pull = (pred - y[~in_tr]) / sig[~in_tr]
        c2 = float(np.mean(pull ** 2))
        rms = float(np.sqrt(np.mean((pred - y[~in_tr]) ** 2)))
        print('%-20s %3d %9.4f %+9.3f %9.3f %7.3f'
              % (name, len(feats), c2, np.median(pull), np.std(pull), rms))
        res[name] = dict(features=feats, p=len(feats), chi2=c2,
                         pull_med=float(np.median(pull)), pull_std=float(np.std(pull)),
                         rms=rms, coef=coef.tolist(), floors=floors.tolist(),
                         feature_mean=mu.tolist(), feature_std=sd.tolist())
        if name[0] in 'ACDG':
            print('    系数: ' + '  '.join('%s=%+.3f' % (f, c)
                                          for f, c in zip(feats, coef[1:])))
    print('-' * 84)
    print('判据：与 A 的差异要 > %.3f（1σ）才值得当回事。' % chi2_se)
    print('参考：MLP [8,8] 1.126 / [32,32] 1.069 / 旧 coalt_full 0.970（含泄漏）')

    if args.out:
        json.dump(res, open(args.out, 'w'), indent=1)
        print('saved', args.out)


if __name__ == '__main__':
    main()
