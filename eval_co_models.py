#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_co_models.py — 在同一 test 扇区上评估若干模型（一律用**测量 CO** 作输入）。

用途：判断"CO 交替更新（Stage-2/3）到底有没有用" ——
Stage-1 模型与 Stage-3 模型在完全相同输入下对比，差异只来自交替训练。

用法:
  python eval_co_models.py --tag coalt_full \
      --models "Stage1=qpah_pipeline/results/coalt_full/model_stage1.pth" \
               "Stage3=qpah_pipeline/results/coalt_full/model.pth"
"""
import os, json, argparse
import numpy as np
import torch
import torch.nn as nn
import h5py

FEAT = ['dust_density', 'H1', 'sfr', 'CO', 'H1_ew']
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, 32), nn.Tanh(),
                                 nn.Linear(32, 32), nn.Tanh(),
                                 nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).flatten()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', required=True, help='name=path 列表')
    ap.add_argument('--tag', default='coalt_full', help='用该 tag 的 scale.json')
    ap.add_argument('--h5', default='data/M31_1d_array_full.h5')
    ap.add_argument('--err-floor', type=float, default=0.2)
    args = ap.parse_args()

    with h5py.File(args.h5, 'r') as f:
        d = {k: f[k][:] for k in ['ra', 'dec', 'qpah', 'qpah_err', 'CO_sigma'] + FEAT}
    sel = ((d['qpah'] > 0) & (d['qpah_err'] > 0) & (d['dust_density'] > 0)
           & (d['H1'] >= 0) & (d['H1_ew'] >= 0)
           & np.isfinite(d['CO']) & np.isfinite(d['CO_sigma']) & (d['CO_sigma'] > 0))
    X = np.column_stack([d[k][sel] for k in FEAT]).astype(np.float32)
    y = d['qpah'][sel].astype(np.float32)
    ye = d['qpah_err'][sel].astype(np.float32)
    ra, dec = d['ra'][sel], d['dec'][sel]

    # 与训练脚本完全一致的扇区划分
    np.random.seed(0)
    ang = (np.arctan2(dec - 41.25, ra - 10.75) + 2 * np.pi) % (2 * np.pi)
    sec = np.digitize(ang, np.linspace(0, 2 * np.pi, 11)) - 1
    tr = np.random.choice(np.arange(10), size=6, replace=False)
    te = np.setdiff1d(np.arange(10), tr)
    tidx = np.where(np.isin(sec, te))[0]

    sc = json.load(open('qpah_pipeline/results/%s/scale.json' % args.tag))
    mean = np.array(sc['mean'])[None, :]
    std = np.array(sc['std'])[None, :]
    Xt = torch.tensor((X - mean) / std, dtype=torch.float32).to(dev)

    yt = y[tidx]
    se = np.sqrt(ye[tidx] ** 2 + args.err_floor ** 2)
    print('test pixels = %d (sectors %s)' % (len(tidx), te.tolist()))
    print('（不报 MSE/RMSLE：qPAH 观测误差异方差，等权平方误差无物理意义）')
    print('%-26s %9s %9s %9s %9s %9s' %
          ('model', 'chi2_red', 'pull_med', 'pull_std', 'pull_p16', 'pull_p84'))
    for spec in args.models:
        name, path = spec.split('=', 1)
        m = MLP().to(dev)
        m.load_state_dict(torch.load(path, map_location=dev, weights_only=True))
        m.eval()
        with torch.no_grad():
            pt = m(Xt[tidx]).cpu().numpy()
        pull = (pt - yt) / se
        print('%-26s %9.3f %+9.3f %9.3f %9.3f %9.3f' %
              (name, float(np.mean(pull ** 2)), float(np.median(pull)),
               float(np.std(pull)), float(np.percentile(pull, 16)),
               float(np.percentile(pull, 84))))


if __name__ == '__main__':
    main()
