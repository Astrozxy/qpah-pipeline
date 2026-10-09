#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fit_linear_model.py — 拟合 log10 线性模型，并【导出成 pipeline 标准格式】，
使得 plot_galaxy_map.py / predict_grid.py / render_pred_maps.py / interpret.py
都能直接用它。

技巧：纯线性模型 = `hidden=[]` 的 MLPRegressor（即单个 Linear(5,1)），
      所以只要写出标准 model.pth / scale.json 即可复用全部现成绘图代码。

模型:  pred = b + Σⱼ wⱼ · ( log10(max(xⱼ,0) + floorⱼ) − meanⱼ ) / stdⱼ
拟合:  误差加权最小二乘，权重 w = 1/σ_eff²，σ_eff = sqrt(qpah_err² + err_floor²)
       （= MLP 所用异方差 Pseudo-Huber 在二次区的等价形式，故两者可比）
训练:  默认用【训练扇区全部像素】（这才是对的做法——CO 探测像素只占 13%）

用法:
  python qpah_pipeline/analysis/fit_linear_model.py --h5 data/M31_1d_array_full.h5 \
      --tag linear_log
"""
import os, json, argparse
import numpy as np
import h5py

FEATURES = ['dust_density', 'H1', 'sfr', 'CO', 'H1_ew']
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, 'results')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', default='data/M31_1d_array_full.h5')
    ap.add_argument('--tag', default='linear_log')
    ap.add_argument('--err-floor', type=float, default=0.2)
    ap.add_argument('--snr-thresh', type=float, default=3.0)
    ap.add_argument('--train-mode', choices=['all', 'det'], default='all')
    ap.add_argument('--floor-frac', type=float, default=0.01)
    ap.add_argument('--dry-run', action='store_true', help='只算不存（本地无 torch 时用）')
    args = ap.parse_args()

    with h5py.File(args.h5, 'r') as f:
        keys = ['ra', 'dec', 'qpah', 'qpah_err', 'CO_sigma'] + FEATURES
        d = {k: f[k][:] for k in keys if k in f}
    sel = ((d['qpah'] >= 0) & (d['qpah_err'] > 0) & (d['dust_density'] > 0)
           & (d['H1'] >= 0) & (d['H1_ew'] >= 0)
           & np.isfinite(d['CO']) & np.isfinite(d['CO_sigma']) & (d['CO_sigma'] > 0))
    ra, dec = d['ra'][sel].astype(np.float64), d['dec'][sel].astype(np.float64)
    Xr = np.maximum(np.column_stack([d[k][sel] for k in FEATURES]).astype(np.float64), 0.0)
    y = d['qpah'][sel].astype(np.float64)
    ye = d['qpah_err'][sel].astype(np.float64)
    co_sig = np.maximum(d['CO_sigma'][sel].astype(np.float64), 1e-9)
    sig = np.sqrt(ye ** 2 + args.err_floor ** 2)
    det = (Xr[:, 3] / co_sig) > args.snr_thresh

    np.random.seed(0)
    ang = (np.arctan2(dec - 41.25, ra - 10.75) + 2 * np.pi) % (2 * np.pi)
    sec = np.digitize(ang, np.linspace(0, 2 * np.pi, 11)) - 1
    tr_s = np.random.choice(np.arange(10), size=6, replace=False)
    te_s = np.setdiff1d(np.arange(10), tr_s)
    in_tr = np.isin(sec, tr_s)
    tmask = (in_tr & det) if args.train_mode == 'det' else in_tr

    floors = {}
    for j, f in enumerate(FEATURES):
        pos = Xr[tmask, j][Xr[tmask, j] > 0]
        floors[f] = float(max(1e-6, args.floor_frac * np.median(pos)))

    def design(Xraw):
        Z = np.empty_like(Xraw)
        for j, f in enumerate(FEATURES):
            Z[:, j] = np.log10(Xraw[:, j] + floors[f])
        return Z

    Zall = design(Xr)
    mu = Zall[tmask].mean(0)
    sd = Zall[tmask].std(0) + 1e-12
    Ztr = np.column_stack([np.ones(int(tmask.sum())), (Zall[tmask] - mu) / sd])
    wgt = 1.0 / sig[tmask] ** 2
    coef = np.linalg.solve(Ztr.T @ (wgt[:, None] * Ztr), Ztr.T @ (wgt * y[tmask]))
    b, w = float(coef[0]), coef[1:]

    # ---- 测试扇区评估 ----
    te = ~in_tr
    Zte = np.column_stack([np.ones(int(te.sum())), (Zall[te] - mu) / sd])
    pred = Zte @ coef
    pull = (pred - y[te]) / sig[te]
    omega = 1.133 * 24.9 ** 2
    n_eff = int(te.sum()) / (omega / 100.0)
    chi2 = float(np.mean(pull ** 2))
    print('train-mode=%s | 训练像素 %d | 测试像素 %d (有效 ≈%.0f)'
          % (args.train_mode, tmask.sum(), te.sum(), n_eff))
    print('log10 偏移 floor: ' + '  '.join('%s=%.3g' % (f, floors[f]) for f in FEATURES))
    print('chi2_red = %.4f  (1σ 统计误差 ±%.3f)   pull med=%+.4f  std=%.4f'
          % (chi2, np.sqrt(2.0 / n_eff), np.median(pull), np.std(pull)))
    print('绝对残差 rms = %.4f %%   预测<0 占比 = %.3f%%'
          % (np.sqrt(np.mean((pred - y[te]) ** 2)), 100 * np.mean(pred < 0)))
    print('标准化系数: 截距=%+.3f | ' % b +
          '  '.join('%s=%+.3f' % (f, c) for f, c in zip(FEATURES, w)))
    print('（原始尺度）log10 幂律指数 = ' + '  '.join('%s=%+.3f' % (f, c) for f, c in zip(FEATURES, w)))

    if args.dry_run:
        print('--dry-run: 不写文件')
        return

    import torch
    outdir = os.path.join(RESULTS, args.tag)
    os.makedirs(outdir, exist_ok=True)
    torch.save({'net.0.weight': torch.tensor(w[None, :], dtype=torch.float32),
                'net.0.bias': torch.tensor([b], dtype=torch.float32)},
               os.path.join(outdir, 'model.pth'))
    json.dump({'features': FEATURES, 'mean': mu.tolist(), 'std': sd.tolist(),
               'transform': {f: {'floor': floors[f]} for f in FEATURES},
               'model_hidden': [], 'model_dropout': 0.0},
              open(os.path.join(outdir, 'scale.json'), 'w'), indent=1)
    json.dump(dict(model='log10 linear (6 params = Linear(5,1))',
                   train_mode=args.train_mode, h5=args.h5,
                   n_train=int(tmask.sum()), n_test=int(te.sum()), n_eff_test=n_eff,
                   chi2_red=chi2, chi2_se=float(np.sqrt(2.0 / n_eff)),
                   pull_med=float(np.median(pull)), pull_std=float(np.std(pull)),
                   pull_p16=float(np.percentile(pull, 16)),
                   pull_p84=float(np.percentile(pull, 84)),
                   rms_abs=float(np.sqrt(np.mean((pred - y[te]) ** 2))),
                   intercept=b, coef_standardized=w.tolist(),
                   feature_mean=mu.tolist(), feature_std=sd.tolist(), floors=floors),
              open(os.path.join(outdir, 'train.json'), 'w'), indent=1)
    print('saved ->', outdir)


if __name__ == '__main__':
    main()
