#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
linear_baseline.py — 线性 / 对数线性基线：任何"物理模型"都必须先击败的对手。

问题背景：波束 24.9″ vs 10″ 像素 ⇒ 有效独立样本只有名义行数的 ~1/7。
所以"用神经网络"这件事本身缺乏统计依据，应该先问：一个 6 参数的显式模型能走多远？

口径（与 train_co_alternate.py 严格一致）：
  - 同样的像素筛选、同样的按方位角 10 扇区 6/4 划分（seed=0）；
  - CO 输入裁剪到 ≥0（负 CO 不物理，且 Stage-1 只在正 CO 上训练过）；
  - 误差加权最小二乘，权重 w = 1/σ_eff²，σ_eff = sqrt(qpah_err² + err_floor²)。
    这是 MLP 所用异方差 Pseudo-Huber 在二次区的等价形式
    （δ=10 且残差 ~1σ ⇒ phuber ≈ r²/2），因此两者可比。
  - 空间测试扇区完全不参与拟合，只在最后评估。

模型（都含截距）：
  raw : qPAH = a + Σ bᵢ·xᵢ                      （6 参数）
  log : qPAH = a + Σ bᵢ·log10(xᵢ + floorᵢ)      （6 参数；幂律/对数形式）

训练集两种：
  det : 只用 CO/σ>3 的【训练扇区】探测像素（= MLP Stage-1 的训练集）
  all : 训练扇区全部像素

用法:
  python qpah_pipeline/analysis/linear_baseline.py --h5 data/M31_1d_array_full.h5
  python qpah_pipeline/analysis/linear_baseline.py --out json/linear_baseline.json
"""
import os, json, argparse
import numpy as np
import h5py

FEATURES = ['dust_density', 'H1', 'sfr', 'CO', 'H1_ew']
BEAM_ARCSEC = 24.9          # qPAH 图的有效波束（Draine+2014 的 S350 PSF）
PIX_ARCSEC = 10.0           # 网格像素


def wls(Xd, y, w):
    """加权最小二乘（Xd 已含常数列）。返回系数。"""
    A = Xd.T @ (w[:, None] * Xd)
    b = Xd.T @ (w * y)
    return np.linalg.solve(A, b)


def evalm(coef, Xd, y, sig):
    pred = Xd @ coef
    pull = (pred - y) / sig
    return dict(chi2=float(np.mean(pull ** 2)), pmed=float(np.median(pull)),
                pstd=float(np.std(pull)),
                p16=float(np.percentile(pull, 16)), p84=float(np.percentile(pull, 84)),
                rms=float(np.sqrt(np.mean((pred - y) ** 2))),
                frac_neg=float(np.mean(pred < 0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', default='data/M31_1d_array_full.h5')
    ap.add_argument('--err-floor', type=float, default=0.2)
    ap.add_argument('--snr-thresh', type=float, default=3.0)
    ap.add_argument('--out', default=None, help='可选：结果存 json')
    ap.add_argument('--features', nargs='+', default=None,
                    help='用哪些特征（默认 5 个全用）。例：--features H1 sfr CO H1_ew '
                         '（去掉 dust，配合 build_dataset.py --require 使用）')
    args = ap.parse_args()
    feats = args.features or list(FEATURES)

    with h5py.File(args.h5, 'r') as f:
        keys = ['ra', 'dec', 'qpah', 'qpah_err', 'CO_sigma'] + feats
        d = {k: f[k][:] for k in keys if k in f}
    sel = ((d['qpah'] > 0) & (d['qpah_err'] > 0))
    for f_ in feats:                                   # 只要求用到的特征有限
        sel &= np.isfinite(d[f_])
    if 'dust_density' in feats:
        sel &= d['dust_density'] > 0                   # 0 = 平滑零填充/缺数据
    if 'H1' in feats:
        sel &= d['H1'] >= 0
    if 'H1_ew' in feats:
        sel &= d['H1_ew'] >= 0
    if 'CO' in feats:
        sel &= np.isfinite(d['CO_sigma']) & (d['CO_sigma'] > 0)
    ra, dec = d['ra'][sel].astype(np.float64), d['dec'][sel].astype(np.float64)
    Xr = np.column_stack([d[k][sel] for k in feats]).astype(np.float64)
    jco = feats.index('CO') if 'CO' in feats else None
    X = Xr.copy()
    if jco is not None:
        X[:, jco] = np.maximum(X[:, jco], 0.0)         # raw：仅 CO 裁剪 ≥0
    y = d['qpah'][sel].astype(np.float64)
    ye = d['qpah_err'][sel].astype(np.float64)
    co_sig = (np.maximum(d['CO_sigma'][sel].astype(np.float64), 1e-9)
              if 'CO_sigma' in d else np.full(len(y), np.inf))
    sig = np.sqrt(ye ** 2 + args.err_floor ** 2)
    det = ((Xr[:, jco] / co_sig) > args.snr_thresh
           if jco is not None else np.zeros(len(y), dtype=bool))

    # ---- 与 train_co_alternate.py 完全一致的扇区划分 ----
    np.random.seed(0)
    ang = (np.arctan2(dec - 41.25, ra - 10.75) + 2 * np.pi) % (2 * np.pi)
    sec = np.digitize(ang, np.linspace(0, 2 * np.pi, 11)) - 1
    tr_s = np.random.choice(np.arange(10), size=6, replace=False)
    te_s = np.setdiff1d(np.arange(10), tr_s)
    in_tr = np.isin(sec, tr_s)

    # ---- 有效独立样本数（波束决定，与网格无关）----
    omega_beam = 1.133 * BEAM_ARCSEC ** 2
    redund = omega_beam / PIX_ARCSEC ** 2
    n_te = int((~in_tr).sum())
    n_eff_te = n_te / redund
    chi2_se = float(np.sqrt(2.0 / n_eff_te))

    print('=' * 78)
    print('训练扇区 %s | 测试扇区 %s' % (tr_s.tolist(), te_s.tolist()))
    print('有效行: 总 %d | 训练 %d (其中 CO/σ>%.0f 探测 %d) | 测试 %d'
          % (sel.sum(), in_tr.sum(), args.snr_thresh, (det & in_tr).sum(), n_te))
    print('波束 %.1f″ / 像素 %.1f″ ⇒ 波束内 %.2f 个像素 ⇒ 有效独立样本 ≈ 名义/%.2f'
          % (BEAM_ARCSEC, PIX_ARCSEC, redund, redund))
    print('  → 测试集有效独立样本 ≈ %.0f（名义 %d）' % (n_eff_te, n_te))
    print('  → χ²_red 的 1σ 统计误差 ≈ ±%.3f' % chi2_se)
    print('=' * 78)

    # ---- 设计矩阵：raw / raw0 / log ----
    # 注意：CO 已裁剪 ≥0；SFR 是背景减除产物、可能有负值（log 会炸），
    # 所以 raw0 把全部特征裁剪到 ≥0（它们物理上都是非负量）。
    print('特征 (%d): %s' % (len(feats), ' '.join(feats)))
    print('各特征负值占比: ' + '  '.join(
        '%s=%.2f%%' % (f, 100.0 * np.mean(X[:, j] < 0)) for j, f in enumerate(feats)))
    Xc = np.maximum(X, 0.0)
    floors = np.array([max(1e-6, 0.01 * np.median(Xc[:, j][Xc[:, j] > 0]))
                       for j in range(len(feats))])
    print('log 偏移 floor : ' + '  '.join('%s=%.3g' % (f, v)
                                          for f, v in zip(feats, floors)))

    results = {}
    tr_mask = {'all': in_tr, 'det': in_tr & det}
    Xd_all = {'raw': X,                                   # 与 MLP 输入一致（仅 CO≥0）
              'raw0': Xc,                                 # 全部特征 ≥0
              'log': np.log10(Xc + floors[None, :])}      # 幂律/对数形式

    # 标准化（用训练集统计量）——只影响系数量纲，不改变拟合
    print('%-8s %-8s %9s %9s %9s %9s %9s  %s'
          % ('train', 'model', 'chi2_red', 'pull_med', 'pull_std', 'p16', 'p84', 'rms(%)'))
    print('-' * 78)
    for tname, tmask in tr_mask.items():
        for mname, Xsrc in Xd_all.items():
            mu = Xsrc[tmask].mean(0)
            sd = Xsrc[tmask].std(0) + 1e-12
            Z = (Xsrc - mu) / sd
            Ztr = np.column_stack([np.ones(tmask.sum()), Z[tmask]])
            Zte = np.column_stack([np.ones((~in_tr).sum()), Z[~in_tr]])
            coef = wls(Ztr, y[tmask], 1.0 / sig[tmask] ** 2)
            r = evalm(coef, Zte, y[~in_tr], sig[~in_tr])
            print('%-8s %-8s %9.3f %+9.3f %9.3f %+9.3f %+9.3f %9.3f'
                  % (tname, mname, r['chi2'], r['pmed'], r['pstd'],
                     r['p16'], r['p84'], r['rms']))
            results['%s_%s' % (tname, mname)] = dict(
                chi2=r['chi2'], pull_med=r['pmed'], pull_std=r['pstd'],
                pull_p16=r['p16'], pull_p84=r['p84'], rms=r['rms'],
                frac_pred_neg=r['frac_neg'],
                coef_standardized=coef.tolist(),
                feature_mean=mu.tolist(), feature_std=sd.tolist(),
                log_floors=floors.tolist())
            # 标准化系数（截距之外）——正负与相对重要性一目了然
            if tname == 'det' and mname == 'log':
                print('    log 模型（标准化）截距=%.3f | ' % coef[0] +
                      ' '.join('%s=%+.3f' % (f, c) for f, c in zip(feats, coef[1:])))

    print('-' * 78)
    print('参考：MLP [8,8](129参数)  χ²=1.126 (ep1200) / 1.297 (ep4000)')
    print('      MLP [32,32](1281参数) χ²=1.069 (ep200) / 1.463 (ep500)')
    print('      旧基线 coalt_full（含测试扇区泄漏）χ²=0.970')
    print('提示：上面所有 χ² 的 1σ 统计误差约 ±%.3f —— 小于它的差异不要当真。' % chi2_se)

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(dict(n_test=n_te, n_eff_test=n_eff_te, chi2_se=chi2_se,
                       beam_arcsec=BEAM_ARCSEC, pixel_arcsec=PIX_ARCSEC,
                       redundancy=redund, results=results),
                  open(args.out, 'w'), indent=1)
        print('saved', args.out)


if __name__ == '__main__':
    main()
