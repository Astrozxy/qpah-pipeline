#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predict_grid.py — 用训练好的模型对 qPAH 全网格做逐像素预测（读 sanity_cache 对齐缓存）。

用法: python predict_grid.py --tag full   (默认 tag=full)
读取: results/<tag>/model.pth, results/<tag>/scale.json,
     sanity_cache/*_ex_mean.npy（由 align_maps.py 生成）
输出: data/qpah_pred_map_<tag>.fits, data/qpah_pred_resid_map_<tag>.fits
"""
import os, json, argparse
import numpy as np
import torch
import torch.nn as nn
from astropy.io import fits
import h5py

REPO = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(REPO)
DATA = os.path.join(ROOT, 'data')
CACHE = os.path.join(ROOT, 'sanity_cache')
FEATURES = ['dust_density', 'H1', 'sfr', 'CO', 'H1_ew']
GRID_KEY = {'dust_density': 'dust', 'H1': 'HI', 'sfr': 'SFR', 'CO': 'CO', 'H1_ew': 'HI_ew'}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def apply_transform(cols, tmeta):
    """按 scale.json 的 transform 规格对特征列做 log10 变换"""
    out = np.asarray(cols, dtype=float).copy()
    for j, f in enumerate(FEATURES):
        t = tmeta[f]
        x = np.maximum(out[:, j], 0.0)          # 与训练一致：负值 clip 到 0
        out[:, j] = np.log10(x + t['floor'])
    return out


class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden=(32, 32), dropout=0.0):
        super().__init__()
        layers, in_dim = [], input_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='full')
    ap.add_argument('--err-floor', type=float, default=0.2,
                    help='σ_eff = sqrt(qpah_err² + err_floor²)，需与训练一致')
    args = ap.parse_args()
    resdir = os.path.join(REPO, 'results', args.tag)

    scale = json.load(open(os.path.join(resdir, 'scale.json')))
    model = MLPRegressor(len(FEATURES),
                         hidden=tuple(scale.get('model_hidden', [32, 32])),
                         dropout=float(scale.get('model_dropout', 0.0)))
    model.load_state_dict(torch.load(os.path.join(resdir, 'model.pth'),
                                     map_location=device, weights_only=True))
    model.to(device).eval()
    mean = np.array(scale['mean'])[None, :]
    std = np.array(scale['std'])[None, :]
    tmeta = scale.get('transform')

    # 特征对齐图直接读 sanity_cache（与训练用的同一份产物）
    cols = np.column_stack([np.load(os.path.join(CACHE, GRID_KEY[k] + '_ex_mean.npy'))
                            .reshape(-1) for k in FEATURES])
    hq = fits.open(os.path.join(DATA, 'M31_S350_110_SSS_110_Model_All_qpah.fits'))
    qpah = (hq[0].data.astype(np.float64) * 100.0)
    hq.close()
    hu = fits.open(os.path.join(DATA, 'M31_S350_110_SSS_110_Model_All_qpah_unc.fits'))
    qpah_unc = (hu[0].data.astype(np.float64) * 100.0)
    hu.close()
    shape = qpah.shape
    fin = np.all(np.isfinite(cols), axis=1)
    pf = np.full(cols.shape[0], np.nan)
    if fin.sum():
        Xt = apply_transform(cols[fin], tmeta) if tmeta else cols[fin]
        xg = (Xt - mean) / std
        with torch.no_grad():
            pf[fin] = model(torch.tensor(xg, dtype=torch.float32).to(device)
                            ).cpu().numpy().flatten()
    pred = pf.reshape(shape)
    hd = fits.open(os.path.join(DATA, 'M31_S350_110_SSS_110_Model_All_qpah.fits'))
    hdr = hd[0].header.copy()
    hd.close()
    fits.writeto(os.path.join(DATA, 'qpah_pred_map_%s.fits' % args.tag),
                 pred.astype(np.float32), hdr, overwrite=True)
    resid = np.full(shape, np.nan)
    m = np.isfinite(pred) & np.isfinite(qpah)
    resid[m] = pred[m] - qpah[m]
    fits.writeto(os.path.join(DATA, 'qpah_pred_resid_map_%s.fits' % args.tag),
                 resid.astype(np.float32), hdr, overwrite=True)
    # 相对误差 χ = (pred − obs)/σ_eff：qPAH 观测误差是异方差的，绝对残差不可比
    sig_eff = np.sqrt(qpah_unc ** 2 + args.err_floor ** 2)
    chi = np.full(shape, np.nan)
    mc = m & np.isfinite(sig_eff) & (sig_eff > 0)
    chi[mc] = resid[mc] / sig_eff[mc]
    fits.writeto(os.path.join(DATA, 'qpah_chi_map_%s.fits' % args.tag),
                 chi.astype(np.float32), hdr, overwrite=True)
    print('pred finite=%d  range=[%.3f, %.3f]' % (np.isfinite(pred).sum(),
                                                  np.nanmin(pred), np.nanmax(pred)))
    print('chi=(pred-obs)/sigma_eff: finite=%d  med=%.3f  p16=%.3f p84=%.3f'
          % (np.isfinite(chi).sum(), np.nanmedian(chi),
             np.nanpercentile(chi, 16), np.nanpercentile(chi, 84)))
    print('saved data/qpah_pred_map_%s.fits (+ resid, + chi map)' % args.tag)
    print('ALL DONE')


if __name__ == '__main__':
    main()
