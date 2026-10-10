#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_full_maps.py — 4 面板全样本图：观测 / 预测 / 相对误差 / 绝对误差。

模型：log10 线性 + 可选的【空间项】。
  - 观测量特征：x → max(x,0) → log10(x + floor)，floor 取训练集正值的 1% 中位
  - 空间项（若给出 --spatial）：直接线性进入设计矩阵（**不做 log**）
      x    : 去投影长轴坐标 (kpc, 有符号) —— 东北/西南
      y    : 去投影短轴坐标 (kpc, 有符号) —— 近/远侧
      R    : 去投影半径 (kpc)
      logR : log10(R + 0.1)
  两类都做 z-score 标准化（统计量取自训练扇区）。
  拟合：误差加权最小二乘，权重 = 1/σ_eff²，σ_eff=sqrt(qpah_err²+err_floor²)。

与前作一致：**不标测试扇区、不做扇区筛选**，覆盖数据集全部像素；
模型指标只在角标给出。

用法:
  python make_full_maps.py --h5 data/M31_1d_array_nodust.h5 \
      --features H1 sfr CO H1_ew --spatial x \
      --out results/fullmaps/fig_4panel_base_x.png
"""
import os, argparse
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mplconfig')
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

DEFAULT_FEATS = ['H1', 'sfr', 'CO', 'H1_ew']
RA0, DEC0, INC, PA = 10.6847, 41.2690, np.radians(77.7), np.radians(37.7)
ARCSEC_PER_KPC = 206265.0 / 744.0


def spatial_terms(ra, dec, which):
    """去投影空间项（与 spatial_terms.py / arm_* 脚本同一套几何）。"""
    dx = (ra - RA0) * np.cos(np.radians(DEC0)); dy = dec - DEC0
    x = (dx * np.sin(PA) + dy * np.cos(PA)) * 3600.0 / ARCSEC_PER_KPC
    y = ((-dx * np.cos(PA) + dy * np.sin(PA)) / np.cos(INC)) * 3600.0 / ARCSEC_PER_KPC
    R = np.sqrt(x ** 2 + y ** 2)
    y2 = y / 2.0
    pool = {'x': x, 'y': y, 'x_major': x, 'y_minor': y,
            'y3': y2 ** 3, 'y2t': y2 ** 2,
            'R': R, 'logR': np.log10(R + 0.1)}
    return [pool[w] for w in which]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', default='data/M31_1d_array_nodust.h5')
    ap.add_argument('--features', nargs='+', default=DEFAULT_FEATS)
    ap.add_argument('--spatial', nargs='*', default=[],
                    choices=['x', 'y', 'x_major', 'y_minor', 'y3', 'y2t', 'R', 'logR'],
                    help='要加入的空间项（可多个，例如 --spatial x 或 --spatial R y x）')
    ap.add_argument('--err-floor', type=float, default=0.2)
    ap.add_argument('--out', default='results/fullmaps/fig_full_4panel.png')
    ap.add_argument('--qmax', type=float, default=6.0, help='qPAH 色标上限 (%)')
    ap.add_argument('--chi-vmax', type=float, default=3.0, help='相对误差色标范围 ±(sigma)')
    ap.add_argument('--abs-vmax', type=float, default=2.0, help='绝对误差色标范围 ±(%)')
    ap.add_argument('--with-abs', action='store_true',
                    help='额外加第 4 个绝对误差面板（默认不加：拟合与评估都用 1/sigma^2 加权，'
                         '未加权的绝对误差图会被高 sigma 像素主导，不具诊断价值）')
    ap.add_argument('--qpah-err-max', type=float, default=None,
                    help='只保留 qpah_err <= 该值的目标像素（默认不切）')
    ap.add_argument('--train-co-snr', type=float, default=None,
                    help='只用 CO/CO_sigma > 该值的像素【拟合】（CO 是当前最差的图）')
    ap.add_argument('--extra-panel', default=None,
                    help='追加一个面板，显示 h5 中的该列（如 dust850 / Ldust / H1），'
                         '使用与前三格完全相同的坐标与样本')
    ap.add_argument('--extra-title', default=None)
    ap.add_argument('--extra-log', action='store_true', help='额外面板用对数色标')
    ap.add_argument('--extra-unit', default='')
    ap.add_argument('--mark-sectors', action='store_true',
                    help='标出训练/测试扇区：画 10 条扇区边界射线，给测试扇区加淡色底纹并逐扇区标注')
    ap.add_argument('--mark-radius', type=float, default=None,
                    help='画出该去投影半径(kpc)的椭圆（例如 18）')
    ap.add_argument('--sector-ra', type=float, default=10.75)
    ap.add_argument('--sector-dec', type=float, default=41.25)
    ap.add_argument('--outline-npz', default=None,
                    help='可选：npz（含 ra/dec/region）—— 在三个面板上把这个区域的像素'
                         '用醒目的点标出来，便于确认"讨论的是同一块"')
    ap.add_argument('--outline-color', default='lime')
    ap.add_argument('--dust-edge-min', type=float, default=None,
                    help='剔除到 850um footprint 边界小于该像素数（读 h5 的 edge_dust850 列）')
    ap.add_argument('--bin-weight', nargs=5, type=float, default=None,
                    metavar=('LO', 'HI', 'NBINS', 'POW', 'FLOOR'),
                    help='按 qPAH 分箱逆频加权：w *= 1/(count+floor)^pow')
    ap.add_argument('--soft-co-snr', type=float, default=None,
                    help='CO 软阈值：CO_eff = max(CO - t*sigma_CO, 0)（连续，无自由参数）')
    ap.add_argument('--zero-co-snr', type=float, default=None,
                    help='【预测】时把 CO/CO_sigma <= 该值的像素的 CO 置 0（未探测=无 CO）')
    args = ap.parse_args()
    feats, spa = args.features, list(args.spatial)

    with h5py.File(args.h5, 'r') as f:
        keys = ['ra', 'dec', 'qpah', 'qpah_err', 'CO_sigma', 'edge_dust850'] + feats
        d = {k: f[k][:] for k in dict.fromkeys(keys) if k in f}
    sel = (d['qpah'] >= 0) & (d['qpah_err'] > 0)
    if args.qpah_err_max is not None:
        sel &= d['qpah_err'] <= args.qpah_err_max
    for k in feats:
        sel &= np.isfinite(d[k])
    if 'dust_density' in feats:
        sel &= d['dust_density'] > 0
    if 'dust850' in feats:
        sel &= np.isfinite(d['dust850']) & (d['dust850'] > 0)
    if args.dust_edge_min is not None and 'edge_dust850' in d:
        sel &= d['edge_dust850'] >= args.dust_edge_min
    if 'H1' in feats:
        sel &= d['H1'] >= 0
    if 'H1_ew' in feats:
        sel &= d['H1_ew'] >= 0
    if 'CO' in feats:
        sel &= np.isfinite(d['CO_sigma']) & (d['CO_sigma'] > 0)
    ra, dec = d['ra'][sel].astype(float), d['dec'][sel].astype(float)
    X = np.maximum(np.column_stack([d[k][sel] for k in feats]).astype(float), 0.0)
    y = d['qpah'][sel].astype(float)
    ye = d['qpah_err'][sel].astype(float)
    sig = np.sqrt(ye ** 2 + args.err_floor ** 2)

    np.random.seed(0)
    ang = (np.arctan2(dec - 41.25, ra - 10.75) + 2 * np.pi) % (2 * np.pi)
    sec = np.digitize(ang, np.linspace(0, 2 * np.pi, 11)) - 1
    tr_s = np.random.choice(np.arange(10), size=6, replace=False)
    in_tr = np.isin(sec, tr_s)

    # ---- CO 的探测/未探测处理（CO 是当前最差的图）----
    jco = feats.index('CO') if 'CO' in feats else None
    snr = None
    if jco is not None:
        snr = X[:, jco] / np.maximum(d['CO_sigma'][sel].astype(float), 1e-30)
    fit_mask = in_tr.copy()
    if jco is not None and args.train_co_snr is not None:
        fit_mask = in_tr & (snr > args.train_co_snr)
    # 未探测（CO/σ <= 阈值）一律取 0：拟合与预测【都】用这个"探测即值、未探测即零"的 CO
    Xp = X.copy()
    nzero = 0
    if jco is not None and args.soft_co_snr is not None:
        thr = args.soft_co_snr * np.maximum(d['CO_sigma'][sel].astype(float), 1e-30)
        nzero = int((X[:, jco] <= thr).sum())
        X[:, jco] = np.maximum(X[:, jco] - thr, 0.0)
        Xp = X.copy()          # 预测必须用同一个 CO 处理（否则拟合/预测口径不一致）
    elif jco is not None and args.zero_co_snr is not None:
        zc = snr <= args.zero_co_snr
        nzero = int(zc.sum())
        Xp[zc, jco] = 0.0
        X = Xp.copy()

    floors = np.array([max(1e-9, 0.01 * np.median(X[fit_mask, j][X[fit_mask, j] > 0]))
                       for j in range(X.shape[1])])
    names = list(feats) + spa

    def _design(Xmat):
        pr = [np.log10(Xmat + floors[None, :])]
        if spa:
            pr += [np.asarray(s)[:, None] for s in spatial_terms(ra, dec, spa)]
        return np.column_stack(pr)

    Zf, Zp = _design(X), _design(Xp)
    mu, sd = Zf[fit_mask].mean(0), Zf[fit_mask].std(0) + 1e-12
    Df = np.column_stack([np.ones(len(Zf)), (Zf - mu) / sd])
    w = 1.0 / sig[fit_mask] ** 2
    if args.bin_weight is not None:
        lo, hi, nb, pw, fo = args.bin_weight
        nb = int(nb)
        be = np.linspace(lo, hi, nb + 1)
        bidx = np.clip(np.digitize(y, be) - 1, 0, nb - 1)
        bc = np.bincount(bidx[fit_mask], minlength=nb).astype(float)
        bf = 1.0 / (bc + fo) ** pw
        w = w * bf[bidx[fit_mask]]
        sh = bc * bf / np.sum(bc * bf)
        print('逆频加权: bins=[%g,%g]x%d pow=%g floor=%g | 每箱权重占比 %s'
              % (lo, hi, nb, pw, fo, ' '.join('%.1f%%' % (100 * x) for x in sh)))
    coef = np.linalg.solve(Df[fit_mask].T @ (w[:, None] * Df[fit_mask]),
                           Df[fit_mask].T @ (w * y[fit_mask]))
    pred = np.column_stack([np.ones(len(Zp)), (Zp - mu) / sd]) @ coef
    Z = Zf
    resid = pred - y
    chi = resid / sig

    def met(m):
        p = chi[m]
        return float(np.mean(p ** 2)), float(np.median(p)), float(np.std(p))

    c2_tr, _, _ = met(in_tr)
    c2_te, pm_te, ps_te = met(~in_tr)
    c2_all, pm_all, ps_all = met(np.ones(len(y), bool))
    print('n=%d (train %d / test %d) | 观测量 %s | 空间项 %s%s'
          % (len(y), in_tr.sum(), (~in_tr).sum(), ' '.join(feats),
             ' '.join(spa) if spa else '(无)',
             ('  | cut: qpah_err<=%.2f' % args.qpah_err_max) if args.qpah_err_max is not None else ''))
    if jco is not None and args.train_co_snr is not None:
        print('CO: 拟合只用 %d 个 CO>%.1fsigma 的像素（占训练集 %.1f%%）'
              % (fit_mask.sum(), args.train_co_snr, 100 * fit_mask.sum() / max(1, in_tr.sum())))
    if nzero:
        print('CO: 软阈值 %.1f sigma -> %d 个像素(%.1f%%)落到 0'
              % (args.soft_co_snr, nzero, 100 * nzero / len(y)))
    print('chi2  fit集=%.4f  train扇区=%.4f  test=%.4f  all=%.4f'
          % (np.mean(chi[fit_mask] ** 2), c2_tr, c2_te, c2_all))
    print('pull(all) 中位=%+.4f  std=%.4f  [p16=%+.3f p84=%+.3f]'
          % (pm_all, ps_all, np.percentile(chi, 16), np.percentile(chi, 84)))
    print('绝对残差 med=%+.3f p16=%+.3f p84=%+.3f rms=%.3f'
          % (np.median(resid), np.percentile(resid, 16), np.percentile(resid, 84),
             np.sqrt(np.mean(resid ** 2))))
    print('标准化系数: 截距=%+.4f | ' % coef[0] +
          '  '.join('%s=%+.4f' % (n, c) for n, c in zip(names, coef[1:])))

    # ---------------- 4 面板 ----------------
    order = np.argsort(np.abs(chi))[::-1]        # 大残差画在上面
    nrm_q = Normalize(0, args.qmax)
    nrm_c = Normalize(-args.chi_vmax, args.chi_vmax)     # 固定 ±3 sigma
    nrm_a = Normalize(-args.abs_vmax, args.abs_vmax)     # 固定 ±2 %
    print('色标: qPAH 0–%.1f%% | 相对误差 ±%.1fσ%s'
          % (args.qmax, args.chi_vmax,
             (' | 绝对误差 ±%.1f%%' % args.abs_vmax) if args.with_abs else ''))
    print('超出色标范围（会被裁到端点色）: |χ|>%.1fσ 占 %.2f%% ； |resid|>%.1f%% 占 %.2f%%'
          % (args.chi_vmax, 100 * np.mean(np.abs(chi) > args.chi_vmax),
             args.abs_vmax, 100 * np.mean(np.abs(resid) > args.abs_vmax)))

    panels = [
        (r'Observation  $q_{\rm PAH}$', y, 'inferno', nrm_q, r'$q_{\rm PAH}$ (%)'),
        (r'Prediction  $q_{\rm PAH}$', pred, 'inferno', nrm_q, r'$q_{\rm PAH}$ (%)'),
        (r'Relative error  $\chi=(\mathrm{pred}-\mathrm{obs})/\sigma_{\rm eff}$',
         chi, 'coolwarm', nrm_c, r'$\chi$   ($\sigma_{\rm eff}$)'),
    ]
    if args.with_abs:
        panels.append((r'Absolute error  $\mathrm{pred}-\mathrm{obs}$',
                       resid, 'coolwarm', nrm_a, r'$\mathrm{pred}-\mathrm{obs}$   (%)'))
    if args.extra_panel:
        with h5py.File(args.h5, 'r') as f:
            ev = f[args.extra_panel][:][sel].astype(float)
        ev = np.where(np.isfinite(ev) & (ev > 0), ev, np.nan) if args.extra_log else ev
        g = ev[np.isfinite(ev)]
        if args.extra_log:
            from matplotlib.colors import LogNorm
            nrm_e = LogNorm(vmin=np.percentile(g, 2), vmax=np.percentile(g, 98))
        else:
            nrm_e = Normalize(np.percentile(g, 2), np.percentile(g, 98))
        ti = args.extra_title or args.extra_panel
        panels.append((ti, ev, 'inferno', nrm_e,
                       '%s %s' % (args.extra_panel, args.extra_unit)))
    # 可选：叠加区域标记。用 RA/Dec 精确匹配到当前样本的下标
    ol_ra = ol_dec = None
    if args.outline_npz:
        zo = np.load(args.outline_npz)
        m = zo['region'].astype(bool)
        from scipy.spatial import cKDTree
        tt = cKDTree(np.column_stack([ra, dec]))
        ddo, iio = tt.query(np.column_stack([zo['ra'][m], zo['dec'][m]]), k=1)
        sel_o = iio[ddo < 1e-7]
        ol_ra, ol_dec = ra[sel_o], dec[sel_o]
        print('outline: %s -> 匹配到当前样本 %d / %d 像素'
              % (args.outline_npz, len(sel_o), m.sum()))

    # ---- 预先算好扇区射线与 R=const 椭圆（都在 RA/Dec 平面）----
    sec_rays, te_fill = [], []
    if args.mark_sectors:
        th = np.linspace(0, 2 * np.pi, 11)
        rmax = 0.95  # 度，恰好覆盖数据 footprint
        for t in th:
            sec_rays.append((args.sector_ra + rmax * np.sin(t) / np.cos(np.radians(args.sector_dec)),
                             args.sector_dec + rmax * np.cos(t)))
        # 测试扇区（浅色填充多边形：中心 + 两条边界射线到 rmax）
        for k in range(10):
            t0, t1 = th[k], th[k + 1]
            poly_ra = [args.sector_ra,
                       args.sector_ra + rmax * np.sin(t0) / np.cos(np.radians(args.sector_dec)),
                       args.sector_ra + rmax * np.sin(t1) / np.cos(np.radians(args.sector_dec))]
            poly_de = [args.sector_dec, args.sector_dec + rmax * np.cos(t0),
                       args.sector_dec + rmax * np.cos(t1)]
            te_fill.append((k, poly_ra, poly_de))
        # 扇区编号用的方位（外端标注位置）
        lab_at = [(args.sector_ra + 0.78 * np.sin(0.5 * (th[k] + th[k + 1])) / np.cos(np.radians(args.sector_dec)),
                   args.sector_dec + 0.78 * np.cos(0.5 * (th[k] + th[k + 1]))) for k in range(10)]
    rad_curve = None
    if args.mark_radius:
        PA_, INC_ = np.radians(37.7), np.radians(77.7)
        s_deg = (206265.0 / 744.0) / 3600.0        # kpc -> 度
        phi = np.linspace(0, 2 * np.pi, 721)
        xr = args.mark_radius * np.cos(phi); yr = args.mark_radius * np.sin(phi)
        dxr = s_deg * (xr * np.sin(PA_) - yr * np.cos(INC_) * np.cos(PA_))
        dyr = s_deg * (xr * np.cos(PA_) + yr * np.cos(INC_) * np.sin(PA_))
        rad_curve = (10.6847 + dxr / np.cos(np.radians(41.2690)), 41.2690 + dyr)

    fig, ax = plt.subplots(1, len(panels), figsize=(6.6 * len(panels), 6.6))
    for a, (title, val, cmap, nrm, clab) in zip(ax, panels):
        if args.mark_sectors:
            for k, pr, pd in te_fill:
                if k in set(tr_s.tolist()):
                    continue                      # 只给【测试】扇区加底纹
                a.fill(pr, pd, color='0.55', alpha=0.16, zorder=0, lw=0)
            for (rr, dd_), in zip(sec_rays):
                a.plot([args.sector_ra, rr], [args.sector_dec, dd_], color='k',
                       lw=0.7, ls='--', alpha=.55, zorder=1)
            for k, (lr, ld_) in enumerate(lab_at):
                is_tr = k in set(tr_s.tolist())
                a.text(lr, ld_, 'T%d' % k if is_tr else 'V%d' % k, fontsize=9, ha='center',
                       va='center', zorder=6, color='k' if is_tr else 'crimson',
                       bbox=dict(fc='white', alpha=.65, ec='none', pad=0.6))
        if rad_curve is not None:
            a.plot(rad_curve[0], rad_curve[1], color='lime', lw=1.5, ls='-', zorder=3)
        sc = a.scatter(ra[order], dec[order], s=1.0, c=val[order], cmap=cmap,
                       norm=nrm, rasterized=True, linewidths=0)
        if ol_ra is not None:
            a.scatter(ol_ra, ol_dec, s=3.2, facecolors='none',
                      edgecolors=args.outline_color, linewidths=0.35,
                      rasterized=True, label='region under study (n=%d)' % len(ol_ra))
            a.legend(loc='lower left', fontsize=10, framealpha=.85, markerscale=3)
        if a is ax[0]:
            hs = []
            if args.mark_radius:
                hs.append(plt.Line2D([], [], color='lime', lw=1.5,
                                     label=r'$R=%.0f$ kpc' % args.mark_radius))
            if args.mark_sectors:
                hs.append(plt.Line2D([], [], color='k', lw=1.2, label='T = train sector'))
                hs.append(plt.Line2D([], [], color='crimson', lw=1.2, label='V = held-out sector'))
                hs.append(plt.Rectangle((0, 0), 1, 1, fc='0.55', alpha=.25, ec='none',
                                        label='held-out sectors (shaded)'))
            if hs:
                a.legend(handles=hs, loc='lower left', fontsize=10, framealpha=.85)
        a.set_title(title, fontsize=15)
        a.set_xlabel('RA (J2000)')
        a.set_aspect('equal'); a.invert_xaxis(); a.grid(True, ls=':', alpha=.35)
        a.tick_params(direction='in')
        if args.mark_sectors or args.mark_radius:
            a.set_xlim(ra.max() + 0.12, ra.min() - 0.12)
            a.set_ylim(dec.min() - 0.12, dec.max() + 0.12)
    ax[0].set_ylabel('Dec (J2000)')

    txt = ('n = %d  (train %d / test %d)\n'
           'test: $\\chi^2_{\\rm red}$=%.3f, pull %+.3f$\\pm$%.3f\n'
           'all : $\\chi^2_{\\rm red}$=%.3f, pull %+.3f$\\pm$%.3f')
    # 放在左下角：3 面板时中间面板的 colorbar 正好在 x=0.5，居中的脚注会被压住
    fig.text(0.012, -0.015, txt % (len(y), int(in_tr.sum()), int((~in_tr).sum()),
                                   c2_te, pm_te, ps_te, c2_all, pm_all, ps_all),
             ha='left', va='top', fontsize=11)
    fig.suptitle('log10-linear: %d observables (%s)%s  —  all pixels, no sector selection'
                 % (len(feats), ', '.join(feats),
                    ('  +  spatial: ' + ', '.join(spa)) if spa else ''), fontsize=14)
    fig.tight_layout(rect=[0, 0.0, 1, 0.945])
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.savefig(args.out, dpi=170, bbox_inches='tight'); plt.close(fig)
    print('saved', args.out)


if __name__ == '__main__':
    main()
