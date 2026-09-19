#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
interpret.py — 固定其它特征、扫 CO / 扫 HI 的模型解释图（M31_power_law cell 18/19 风格）。

用法: python interpret.py --tag full
读取: results/<tag>/model.pth + scale.json；数据集 data/M31_1d_array_<selection>.h5
     （--selection 与 --tag 一致时可省略；也可用 --dataset 直接指定）
输出: results/<tag>/fig_interpret_varCO.png, fig_interpret_varHI.png
说明:
  - 固定值取训练样本中位数；dust 输入按 dust/gas 质量比 k 与气体联动（同 notebook）；
  - 图中含"接近固定条件"子集的数据 16/50/84 百分位包络；
  - 只用于展示趋势，非因果。
"""
import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import h5py

REPO = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(REPO)
DATA = os.path.join(ROOT, 'data')
FEATURES = ['dust_density', 'H1', 'sfr', 'CO', 'H1_ew']
SEL_FILE = {'full': 'M31_1d_array_full.h5', 'agb': 'M31_1d_array_agb.h5'}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
C_mass = 8.00e-21
ALPHA_HI = 1.823e18 * C_mass
X_CO = 2.0e20 * C_mass * 2.0


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


def load_data(h5path):
    keys = ['qpah', 'sfr', 'CO', 'H1', 'H1_ew', 'dust_density']
    with h5py.File(h5path, 'r') as f:
        d = {k: f[k][:] for k in keys}
    idx = ((d['qpah'] > 0) & (d['dust_density'] > 0) & (d['H1'] >= 0) & (d['H1_ew'] >= 0))
    return {k: v[idx] for k, v in d.items()}


def envelope(x, y):
    o = np.argsort(x)
    xo, yo = x[o], y[o]
    nb = min(12, max(3, len(xo) // 10))
    ed = np.percentile(xo, np.linspace(0, 100, nb + 1))
    ed[-1] += 1e-6
    bi = np.digitize(xo, ed[:-1]) - 1
    bc = []; q16 = []; q50 = []; q84 = []
    for b in range(nb):
        m = bi == b
        if m.sum() >= 3:
            bc.append(np.median(xo[m]))
            q16.append(np.percentile(yo[m], 16))
            q50.append(np.percentile(yo[m], 50))
            q84.append(np.percentile(yo[m], 84))
    return np.array(bc), np.array(q16), np.array(q50), np.array(q84)


def predict(model, scale, Xraw):
    # 与 train.py 相同的 log10 变换（含 CO clip0）
    tmeta = scale.get('transform')
    if tmeta:
        Xt = np.asarray(Xraw, dtype=float).copy()
        for j, f in enumerate(FEATURES):
            t = tmeta[f]
            x = np.maximum(Xt[:, j], 0.0)        # 与训练一致：负值 clip 到 0
            Xt[:, j] = np.log10(x + t['floor'])
        Xraw = Xt
    m = np.array(scale['mean'])[None, :]
    s = np.array(scale['std'])[None, :]
    Xn = (Xraw - m) / s
    with torch.no_grad():
        return model(torch.tensor(Xn, dtype=torch.float32).to(device)).cpu().numpy().flatten()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selection', choices=['full', 'agb'], default='full')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--dataset', default=None)
    ap.add_argument('--co-max-pct', type=float, default=99.0,
                    help='绘图/扫描时 CO 的上限分位数（默认 99）。保留拐点后的下降趋势；'
                         '超出数据 p95/p99 的部分会在图上用虚线标出（样本稀疏区）')
    ap.add_argument('--h1-max-pct', type=float, default=95.0,
                    help='绘图/扫描时 H1 的上限分位数')
    args = ap.parse_args()
    tag = args.tag or args.selection
    resdir = os.path.join(REPO, 'results', tag)

    scale = json.load(open(os.path.join(resdir, 'scale.json')))
    model = MLPRegressor(len(FEATURES),
                         hidden=tuple(scale.get('model_hidden', [32, 32])),
                         dropout=float(scale.get('model_dropout', 0.0)))
    model.load_state_dict(torch.load(os.path.join(resdir, 'model.pth'),
                                     map_location=device, weights_only=True))
    model.to(device).eval()

    h5path = args.dataset or os.path.join(DATA, SEL_FILE[args.selection])
    ds = load_data(h5path)
    dust, H1, sfr, CO, H1ew = (ds[f] for f in FEATURES)
    qpah = ds['qpah']

    Sg = ALPHA_HI * H1 + X_CO * CO
    k = np.sum(Sg * dust) / np.sum(Sg ** 2)
    print('n=%d  k(dust/gas)=%.4e' % (len(qpah), k), flush=True)
    med = dict(H1=np.median(H1), sfr=np.median(sfr), H1_ew=np.median(H1ew),
               CO=np.median(CO))

    def panel(xv, yp, xd, yd, xlab, title, color):
        bc, q16, q50, q84 = envelope(xd, yd)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(xd, yd, s=4, alpha=0.2, c='gray', label='data close subset')
        ax.plot(bc, q50, 'o-', c='darkorange', ms=3, lw=1.3, label='data 50th')
        ax.plot(bc, q16, 's--', c='darkorange', ms=2.5, lw=1, label='data 16/84th')
        ax.plot(bc, q84, 's--', c='darkorange', ms=2.5, lw=1)
        ax.plot(xv, yp, '-', c=color, lw=2, label='model')
        ax.set_xlabel(xlab); ax.set_ylabel('qPAH (%)')
        ax.set_title(title + '  [k=%.2e]' % k)
        ax.grid(True, ls=':'); ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(resdir, name), dpi=150)
        plt.close(fig)
        print('saved', os.path.join(resdir, name), flush=True)

    # 变 CO
    m1 = ((np.abs(H1 - med['H1']) <= 0.33 * H1.std()) &
          (np.abs(sfr - med['sfr']) <= 0.33 * sfr.std()) &
          (np.abs(H1ew - med['H1_ew']) <= 0.33 * H1ew.std()))
    lo, hi = (np.percentile(CO[m1], 5), np.percentile(CO[m1], 95)) if m1.sum() >= 20 \
        else (np.percentile(CO, 5), np.percentile(CO, 95))
    xv = np.linspace(lo, hi, 200)
    X1 = np.zeros((len(xv), 5))
    X1[:, 0] = k * (ALPHA_HI * med['H1'] + X_CO * xv)
    X1[:, 1] = med['H1']; X1[:, 2] = med['sfr']; X1[:, 3] = xv; X1[:, 4] = med['H1_ew']
    name = 'fig_interpret_varCO.png'
    panel(xv, predict(model, scale, X1), CO[m1], qpah[m1],
          'CO intensity (K km s$^{-1}$)', 'Fixed H1/SFR/H1_EW medians, vary CO', 'b')

    # 变 H1
    m2 = ((np.abs(CO - med['CO']) <= 0.5 * CO.std()) &
          (np.abs(sfr - med['sfr']) <= 0.5 * sfr.std()) &
          (np.abs(H1ew - med['H1_ew']) <= 0.5 * H1ew.std()))
    lo2, hi2 = (np.percentile(H1[m2], 5), np.percentile(H1[m2], 95)) if m2.sum() >= 20 \
        else (np.percentile(H1, 5), np.percentile(H1, 95))
    xv2 = np.linspace(lo2, hi2, 200)
    X2 = np.zeros((len(xv2), 5))
    X2[:, 0] = k * (ALPHA_HI * xv2 + X_CO * med['CO'])
    X2[:, 1] = xv2; X2[:, 2] = med['sfr']; X2[:, 3] = med['CO']; X2[:, 4] = med['H1_ew']
    name = 'fig_interpret_varHI.png'
    panel(xv2, predict(model, scale, X2), H1[m2], qpah[m2],
          'H1 intensity (K km s$^{-1}$)', 'Fixed CO/SFR/H1_EW medians, vary H1', 'r')

    # ===== 质量面密度版（对应 M31_power_law notebook: HI+CO 转 M_sun pc^-2） =====
    Sigma_HI = ALPHA_HI * H1          # M_sun pc^-2
    Sigma_H2 = X_CO * CO              # M_sun pc^-2
    print('alpha_HI(mass)=%.3e  X_CO(mass)=%.3e  k=%.4e'
          % (ALPHA_HI, X_CO, k), flush=True)
    fixed_sfr = med['sfr']
    fixed_h1ew = med['H1_ew']

    # ---- 场景 A: 固定 HI，变化 CO -> x 轴 Sigma(H2) ----
    fixed_h1 = med['H1']
    # 扫描上限用「条件子集」决定：其它特征接近中位的真实样本里 CO 的实际范围。
    # 全体分位数会严重高估（CO 与 dust/H1 强相关，corr~0.7，固定中位推到高 CO 是零样本区）。
    # 扫描范围用全体数据的 p1 - p{co_max_pct}：保留完整趋势，
    # 特别是拐点(CO~1.5)之后的下降 —— 那是碳竞争(CO 与 PAH 争夺碳原子)的偏效应，
    # 在边际关系里被 dust-CO 混杂掩盖（corr(CO,dust)=+0.73），多变量模型才能学出来。
    # 超出 p95/p99 的部分样本稀疏，图上用虚线标出，读者可自行判断外推区。
    co_min, co_max = np.percentile(CO, [1.0, args.co_max_pct])
    co_p50, co_p95, co_p99 = np.percentile(CO, [50, 95, 99])
    co_vals = np.linspace(co_min, co_max, 200)
    n_above95 = int((CO > co_p95).sum())
    print('scan range: CO in [%.2f, %.2f] (p1-p%.1f) | p50=%.2f p95=%.2f p99=%.2f '
          '| CO>p95 仅 %d px (%.2f%%)'
          % (co_min, co_max, args.co_max_pct, co_p50, co_p95, co_p99,
             n_above95, 100.0 * n_above95 / len(CO)), flush=True)
    Sigma_HI_fix = ALPHA_HI * fixed_h1
    Sigma_H2_vals = X_CO * co_vals
    dustA = k * (Sigma_HI_fix + Sigma_H2_vals)
    XA = np.zeros((len(co_vals), 5))
    XA[:, 0] = dustA; XA[:, 1] = fixed_h1; XA[:, 2] = fixed_sfr
    XA[:, 3] = co_vals; XA[:, 4] = fixed_h1ew
    yA = predict(model, scale, XA)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(Sigma_H2_vals, yA, 'b-', lw=2, label='model')
    # 数据支持范围：p95/p99 之外样本很稀疏（占比见日志），供读者判断外推区
    ax.axvline(X_CO * co_p95, color='gray', ls=':', lw=1.2,
               label='data p95 (CO=%.2f)' % co_p95)
    ax.axvline(X_CO * co_p99, color='gray', ls='--', lw=1.2,
               label='data p99 (CO=%.2f)' % co_p99)
    # 数据里的拐点：多元回归 qPAH~a*CO+b*CO^2 给 b<0（碳竞争），CO*≈1.5
    ax.axvline(X_CO * 1.48, color='crimson', ls='-.', lw=1.0, alpha=0.7,
               label=r'data pivot CO$\approx$1.5 (b$<$0)')
    ax.set_xlabel(r'$\Sigma(\mathrm{H}_2)$ (M$_\odot$ pc$^{-2}$)')
    ax.set_ylabel(r'Predicted $q_{\rm PAH}$ (%)')
    ax.set_title(r'Fixed $\Sigma(\mathrm{HI})$ = %.2f M$_\odot$ pc$^{-2}$, k=%.3e'
                 % (Sigma_HI_fix, k))
    ax.set_xlim(left=0); ax.grid(True); ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(resdir, 'fig_interpret_SigmaH2.png'), dpi=150)
    plt.close(fig)
    print('saved results/%s/fig_interpret_SigmaH2.png' % tag, flush=True)

    # ---- 场景 B: 固定 CO，变化 HI -> x 轴 Sigma(HI) ----
    fixed_co = med['CO']
    h1_lo, h1_hi = np.percentile(H1, 5), np.percentile(H1, args.h1_max_pct)
    h1_vals = np.linspace(h1_lo, h1_hi, 200)
    Sigma_HI_vals = ALPHA_HI * h1_vals
    Sigma_H2_fix = X_CO * fixed_co
    dustB = k * (Sigma_HI_vals + Sigma_H2_fix)
    XB = np.zeros((len(h1_vals), 5))
    XB[:, 0] = dustB; XB[:, 1] = h1_vals; XB[:, 2] = fixed_sfr
    XB[:, 3] = fixed_co; XB[:, 4] = fixed_h1ew
    yB = predict(model, scale, XB)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(Sigma_HI_vals, yB, 'r-', lw=2)
    ax.set_xlabel(r'$\Sigma(\mathrm{HI})$ (M$_\odot$ pc$^{-2}$)')
    ax.set_ylabel(r'Predicted $q_{\rm PAH}$ (%)')
    ax.set_title(r'Fixed $\Sigma(\mathrm{H}_2)$ = %.3f M$_\odot$ pc$^{-2}$, k=%.3e'
                 % (Sigma_H2_fix, k))
    ax.set_xlim(left=0); ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(resdir, 'fig_interpret_SigmaHI.png'), dpi=150)
    plt.close(fig)
    print('saved results/%s/fig_interpret_SigmaHI.png' % tag, flush=True)

    # ---- 场景 C: 2D 等值图 Sigma(H2) x Sigma(HI) ----
    ng = 60
    co_grid = np.linspace(co_min, co_max, ng)
    h1_grid = np.linspace(h1_lo, h1_hi, ng)
    COM, H1M = np.meshgrid(co_grid, h1_grid)
    S2M = X_CO * COM
    SIM = ALPHA_HI * H1M
    dustM = k * (SIM + S2M)
    XC = np.column_stack([dustM.ravel(), H1M.ravel(),
                          np.full(COM.size, fixed_sfr),
                          COM.ravel(),
                          np.full(COM.size, fixed_h1ew)])
    yC = predict(model, scale, XC).reshape(COM.shape)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.contourf(S2M, SIM, yC, levels=50, cmap='viridis')
    fig.colorbar(im, ax=ax).set_label(r'Predicted $q_{\rm PAH}$ (%)')
    ax.set_xlabel(r'$\Sigma(\mathrm{H}_2)$ (M$_\odot$ pc$^{-2}$)')
    ax.set_ylabel(r'$\Sigma(\mathrm{HI})$ (M$_\odot$ pc$^{-2}$)')
    ax.set_title(r'Fixed SFR, H1_ew; k=%.3e' % k)
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.join(resdir, 'fig_interpret_Sigma2D.png'), dpi=150)
    plt.close(fig)

    # ===== 双组合图（2D + 1D 剖面，notebook "最终修正版"） =====
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    def _combined(ax2d, fig, ax1d, mode):
        vmin = np.nanmin(yC[yC > 0])
        vmax = np.nanmax(yC)
        levels = np.logspace(np.log10(vmin), np.log10(5), 20)
        im = ax2d.contourf(S2M, SIM, yC, levels=levels, cmap='coolwarm_r',
                           norm=LogNorm(vmin=vmin, vmax=vmax))
        ax2d.set_xlabel(r'$\Sigma(\mathrm{H}_2)=\alpha_{\mathrm{CO}}W_{\mathrm{CO}}$ (M$_\odot$ pc$^{-2}$)')
        ax2d.set_ylabel(r'$\Sigma(\mathrm{HI})$ (M$_\odot$ pc$^{-2}$)')
        ax2d.set_xlim(left=0); ax2d.set_ylim(bottom=0)
        if mode == 'HI':
            ax2d.axhline(y=Sigma_HI_fix, color='magenta', ls='--', lw=2,
                         label=fr'$\Sigma(HI)$ = {Sigma_HI_fix:.1f}')
            ax1d.plot(Sigma_H2_vals, yA, 'magenta', lw=2)
            ax1d.set_xlabel(r'$\Sigma(\mathrm{H}_2)$ (M$_\odot$ pc$^{-2}$)')
        else:
            ax2d.axvline(x=Sigma_H2_fix, color='magenta', ls='--', lw=2,
                         label=fr'$\Sigma(H_2)$ = {Sigma_H2_fix:.1f}')
            ax1d.plot(Sigma_HI_vals, yB, 'magenta', lw=2)
            ax1d.set_xlabel(r'$\Sigma(\mathrm{HI})$ (M$_\odot$ pc$^{-2}$)')
        ax2d.legend(loc='upper right', fontsize=10)
        ax1d.set_ylabel(r'Predicted $q_{\rm PAH}$ (%)')
        ax1d.grid(True); ax1d.set_xlim(left=0)
        div = make_axes_locatable(ax2d)
        cax = div.append_axes('left', size='5%', pad=0.8)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_ticks([2, 3, 4, 5]); cbar.set_ticklabels(['2', '3', '4', '5'])
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')
        cbar.set_label(r'Predicted $q_{\rm PAH}$ (%)')
        return im

    fig1, (aL1, aR1) = plt.subplots(1, 2, figsize=(12, 5),
                                    gridspec_kw={'width_ratios': [1.2, 1]})
    _combined(aL1, fig1, aR1, 'HI')
    fig1.subplots_adjust(left=0.12, right=0.98, wspace=0.25, top=0.95, bottom=0.15)
    fig1.savefig(os.path.join(resdir, 'fig_interpret_Combined_fixed_HI.png'), dpi=200)
    plt.close(fig1)
    print('saved results/%s/fig_interpret_Combined_fixed_HI.png' % tag, flush=True)

    fig2, (aL2, aR2) = plt.subplots(1, 2, figsize=(12, 5),
                                    gridspec_kw={'width_ratios': [1.2, 1]})
    _combined(aL2, fig2, aR2, 'CO')
    fig2.subplots_adjust(left=0.12, right=0.98, wspace=0.25, top=0.95, bottom=0.15)
    fig2.savefig(os.path.join(resdir, 'fig_interpret_Combined_fixed_CO.png'), dpi=200)
    plt.close(fig2)
    print('saved results/%s/fig_interpret_Combined_fixed_CO.png' % tag, flush=True)
    print('ALL DONE')


if __name__ == '__main__':
    main()
