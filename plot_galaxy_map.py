#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_galaxy_map.py — 全样本 RA-Dec 三面板图（观测 / 预测 / 残差）。
复刻 M31_power_law.ipynb 的 tmp/M31_model.pdf 图：
  - 用训练好的模型对全部有效样本逐像素预测；
  - 3 面板：Observation / Prediction / chi = (pred-obs)/sigma_eff（相对误差，
    因为 qPAH 观测误差是异方差的，绝对残差不可比）；
  - 叠加测试扇区（test sectors）的径向虚线 + 边框高亮。

用法:
  python plot_galaxy_map.py --tag coalt_full [--dataset data/M31_1d_array_full.h5]

输出: results/<tag>/fig_galaxy_map.png (+ .pdf)
"""
import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import torch
import torch.nn as nn
import h5py

REPO = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(REPO)
DATA = os.path.join(ROOT, 'data')
RESULTS = os.path.join(REPO, 'results')
FEATURES = ['dust_density', 'H1', 'sfr', 'CO', 'H1_ew']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random_state = 0


class MLPRegressor(nn.Module):
    def __init__(self, input_dim=5, hidden=(32, 32), dropout=0.0):
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
        return self.net(x).flatten()


def set_seed(seed=random_state):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='coalt_full')
    ap.add_argument('--dataset', default=None)
    ap.add_argument('--err-floor', type=float, default=0.2,
                    help='σ_eff = sqrt(qpah_err² + err_floor²)，需与训练一致')
    args = ap.parse_args()
    resdir = os.path.join(RESULTS, args.tag)

    set_seed(random_state)
    h5path = args.dataset or os.path.join(DATA, 'M31_1d_array_full.h5')
    with h5py.File(h5path, 'r') as f:
        ra = f['ra'][:]
        dec = f['dec'][:]
        qpah = f['qpah'][:]
        qpah_err = f['qpah_err'][:]
        X = np.column_stack([f[k][:] for k in FEATURES])
        co_sigma = f['CO_sigma'][:] if 'CO_sigma' in f else np.full(len(qpah), np.inf)

    # 与 train_co_alternate.py load_h5 完全一致的筛选
    idx = ((qpah > 0) & (qpah_err > 0)
           & (X[:, 0] > 0) & (X[:, 1] >= 0) & (X[:, 4] >= 0)
           & np.isfinite(X[:, 3]) & np.isfinite(co_sigma) & (co_sigma > 0))
    ra = ra[idx]; dec = dec[idx]; qpah = qpah[idx]
    qpah_err = qpah_err[idx]; X = X[idx]
    print('valid n =', len(qpah), flush=True)

    # 扇区划分（与 train_co_alternate.py sector_split 一致）
    ang = np.arctan2(dec - 41.25, ra - 10.75)
    ang = (ang + 2 * np.pi) % (2 * np.pi)
    sec = np.digitize(ang, np.linspace(0, 2 * np.pi, 11)) - 1
    tr = np.random.choice(np.arange(10), size=6, replace=False)
    te = np.setdiff1d(np.arange(10), tr)
    test_sectors = te
    sector_bins = np.linspace(0, 2 * np.pi, 11)
    ra_center, dec_center = 10.75, 41.25
    print('train sectors', tr.tolist(), '| test sectors', te.tolist(), flush=True)

    # 加载模型 + scale，预测全部样本
    scale = json.load(open(os.path.join(resdir, 'scale.json')))
    model = MLPRegressor(
        hidden=tuple(scale.get('model_hidden', [32, 32])),
        dropout=float(scale.get('model_dropout', 0.0))).to(device)
    model.load_state_dict(torch.load(os.path.join(resdir, 'model.pth'),
                                     map_location=device, weights_only=True))
    model.eval()
    mean = np.array(scale['mean'])[None, :]
    std = np.array(scale['std'])[None, :]
    tmeta = scale.get('transform')
    Xt = np.asarray(X, dtype=float).copy()
    if tmeta:
        for j, f in enumerate(FEATURES):
            t = tmeta[f]
            Xt[:, j] = np.log10(np.maximum(Xt[:, j], 0.0) + t['floor'])
    Xn = (Xt - mean) / std
    with torch.no_grad():
        pred = model(torch.tensor(Xn, dtype=torch.float32).to(device)
                     ).cpu().numpy().flatten()

    # ==================== 绘图（复刻 notebook cell） ====================
    plt.style.use('default')
    plt.rcParams.update({'font.size': 15})
    xmin, xmax = 10.01, 11.5
    ymin, ymax = 40.6, 42.0
    R = 1.0
    OFFSET = 0.02

    def draw_radial_lines(ax, color):
        for s in test_sectors:
            t0, t1 = sector_bins[s], sector_bins[s + 1]
            ax.plot([ra_center, ra_center + R * np.cos(t0)],
                    [dec_center, dec_center + R * np.sin(t0)],
                    color=color, linestyle='--', linewidth=1.0, alpha=0.7)
            ax.plot([ra_center, ra_center + R * np.cos(t1)],
                    [dec_center, dec_center + R * np.sin(t1)],
                    color=color, linestyle='--', linewidth=1.0, alpha=0.7)

    def ray_rect_intersection(cx, cy, angle, xmin, xmax, ymin, ymax):
        t_vals = []
        if abs(np.cos(angle)) > 1e-12:
            t = (xmin - cx) / np.cos(angle)
            if t >= 0:
                y = cy + t * np.sin(angle)
                if ymin - 1e-12 <= y <= ymax + 1e-12:
                    t_vals.append((t, xmin, y))
            t = (xmax - cx) / np.cos(angle)
            if t >= 0:
                y = cy + t * np.sin(angle)
                if ymin - 1e-12 <= y <= ymax + 1e-12:
                    t_vals.append((t, xmax, y))
        if abs(np.sin(angle)) > 1e-12:
            t = (ymin - cy) / np.sin(angle)
            if t >= 0:
                x = cx + t * np.cos(angle)
                if xmin - 1e-12 <= x <= xmax + 1e-12:
                    t_vals.append((t, x, ymin))
            t = (ymax - cy) / np.sin(angle)
            if t >= 0:
                x = cx + t * np.cos(angle)
                if xmin - 1e-12 <= x <= xmax + 1e-12:
                    t_vals.append((t, x, ymax))
        if not t_vals:
            return None
        return min(t_vals, key=lambda item: item[0])[1:]

    def point_to_param(x, y, xmin, xmax, ymin, ymax):
        if abs(y - ymin) < 1e-10 and xmin <= x <= xmax:
            return x - xmin
        elif abs(x - xmax) < 1e-10 and ymin <= y <= ymax:
            return (xmax - xmin) + (y - ymin)
        elif abs(y - ymax) < 1e-10 and xmin <= x <= xmax:
            return (xmax - xmin) + (ymax - ymin) + (xmax - x)
        elif abs(x - xmin) < 1e-10 and ymin <= y <= ymax:
            return (xmax - xmin) + (ymax - ymin) + (xmax - xmin) + (ymax - y)
        else:
            raise ValueError("Point not on border")

    def param_to_point(t, xmin, xmax, ymin, ymax):
        L1 = xmax - xmin
        L2 = ymax - ymin
        P = 2 * (L1 + L2)
        t = t % P
        if t <= L1:
            return (xmin + t, ymin)
        elif t <= L1 + L2:
            return (xmax, ymin + (t - L1))
        elif t <= L1 + L2 + L1:
            return (xmax - (t - L1 - L2), ymax)
        else:
            return (xmin, ymax - (t - L1 - L2 - L1))

    def highlight_border_sectors(ax, color, offset=OFFSET):
        L1 = xmax - xmin
        L2 = ymax - ymin
        P = 2 * (L1 + L2)
        for s in test_sectors:
            p1 = ray_rect_intersection(ra_center, dec_center, sector_bins[s],
                                       xmin, xmax, ymin, ymax)
            p2 = ray_rect_intersection(ra_center, dec_center, sector_bins[s + 1],
                                       xmin, xmax, ymin, ymax)
            if p1 is None or p2 is None:
                continue
            t1 = point_to_param(p1[0], p1[1], xmin, xmax, ymin, ymax)
            t2 = point_to_param(p2[0], p2[1], xmin, xmax, ymin, ymax)
            if t2 < t1:
                t2 += P
            num_pts = max(100, int((t2 - t1) / P * 500))
            xs, ys = [], []
            for t in np.linspace(t1, t2, num_pts):
                x, y = param_to_point(t, xmin, xmax, ymin, ymax)
                xs.append(x)
                ys.append(y)
            xs_shift, ys_shift = [], []
            for x, y in zip(xs, ys):
                if abs(y - ymin) < 1e-10:
                    xs_shift.append(x)
                    ys_shift.append(ymin + offset)
                elif abs(x - xmax) < 1e-10:
                    xs_shift.append(xmax - offset)
                    ys_shift.append(y)
                elif abs(y - ymax) < 1e-10:
                    xs_shift.append(x)
                    ys_shift.append(ymax - offset)
                elif abs(x - xmin) < 1e-10:
                    xs_shift.append(xmin + offset)
                    ys_shift.append(y)
                else:
                    xs_shift.append(x)
                    ys_shift.append(y)
            ax.plot(xs_shift, ys_shift, color=color, linewidth=3.0,
                    alpha=0.9, solid_capstyle='butt')

    fig = plt.figure(figsize=(12, 3))
    cmap = cm.inferno
    cmap_diff = cm.coolwarm
    norm_qpah = Normalize(vmin=0, vmax=8)
    norm_diff = Normalize(vmin=-3, vmax=3)

    ax0 = fig.add_subplot(131)
    ax0.scatter(ra, dec, s=0.1, c=cmap(norm_qpah(qpah)), rasterized=True)
    draw_radial_lines(ax0, 'cyan')
    highlight_border_sectors(ax0, 'cyan')
    ax0.set_xlabel("RA (J2000)")
    ax0.set_ylabel("Dec (J2000)")
    ax0.set_title("Observation")

    ax1 = fig.add_subplot(132)
    ax1.scatter(ra, dec, s=0.2, c=cmap(norm_qpah(pred)), rasterized=True)
    draw_radial_lines(ax1, 'cyan')
    highlight_border_sectors(ax1, 'cyan')
    ax1.set_xlabel("RA (J2000)")
    ax1.set_title("Prediction")

    ax2 = fig.add_subplot(133)
    # 第 3 面板：相对误差 χ = (pred − obs)/σ_eff（观测误差是异方差的，绝对残差不可比）
    sig_eff = np.sqrt(qpah_err ** 2 + args.err_floor ** 2)
    chi = (pred - qpah) / sig_eff
    ax2.scatter(ra, dec, s=0.1, c=cmap_diff(norm_diff(chi)), rasterized=True)
    draw_radial_lines(ax2, 'magenta')
    highlight_border_sectors(ax2, 'magenta')
    ax2.set_xlabel("RA (J2000)")
    ax2.set_title(r"$\chi=(\mathrm{pred}-\mathrm{obs})/\sigma_{\rm eff}$")

    for ax in [ax0, ax1, ax2]:
        ax.tick_params(direction='in')
        ax.grid(True, linestyle=':')
        ax.tick_params(labelleft=False)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.invert_xaxis()
    ax0.tick_params(labelleft=True)

    cax_left = fig.add_axes([0.04, 0.15, 0.015, 0.70])
    sm_left = ScalarMappable(norm=norm_qpah, cmap=cmap)
    sm_left.set_array([])
    cbar_left = fig.colorbar(sm_left, cax=cax_left, orientation='vertical')
    cbar_left.set_label(r"$q_{\rm PAH}\ (\%)$")
    cbar_left.ax.yaxis.set_ticks_position('left')
    cbar_left.ax.yaxis.set_label_position('left')

    cax_res = fig.add_axes([0.92, 0.15, 0.015, 0.70])
    sm_res = ScalarMappable(norm=norm_diff, cmap=cmap_diff)
    sm_res.set_array([])
    cbar_res = fig.colorbar(sm_res, cax=cax_res, orientation='vertical')
    cbar_res.set_label(r"$\chi$  $(\sigma_{\rm eff})$")

    for cb in [cbar_left, cbar_res]:
        cb.ax.tick_params(direction='in')

    plt.subplots_adjust(wspace=0.0, bottom=0.17)
    png = os.path.join(resdir, 'fig_galaxy_map.png')
    pdf = os.path.join(resdir, 'fig_galaxy_map.pdf')
    plt.savefig(png, dpi=200)
    plt.savefig(pdf, dpi=200)
    print('saved', png, flush=True)
    print('saved', pdf, flush=True)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
