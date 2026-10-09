#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
spatial_terms.py — 在 log10 线性模型里显式加入【空间项】：去投影半径 R、盘面坐标 x/y。

动机
----
模型只有 5 个自由参数，而有效独立样本约 1.6e4（波束 24.9" / 像素 10" ⇒ 冗余 7.02）。
每参数 ≈2100 个有效样本 ⇒ 加 2-3 个空间项在统计上毫不勉强。
物理依据：χ² 随半径强变化（R<5 kpc ≈0.5，R>12 kpc ≈2-5），且 M31 的南北不对称早已有之，
很可能就是金属度梯度的表现。

空间项（只用于建模，不进任何标签）：
    R  : 去投影半径 (kpc)                      —— 径向依赖
    y  : 去投影短轴坐标 (kpc，有符号)           —— 近/远侧（南北不对称）
    x  : 去投影长轴坐标 (kpc，有符号)           —— 东北/西南不对称

对照特征集:
    A  base(4)                : H1 sfr CO H1_ew
    B  base+R                 : +R
    C  base+R+y               : +R +y
    D  base+R+y+x             : +R +y +x
    E  base+logR              : +log10(R)

用法:
  python spatial_terms.py --h5 data/M31_1d_array_nodust.h5 \
      --mask-npz results/armregion/mask_arm_35_38_40_46_51.npz
"""
import os, argparse
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mplconfig')
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_FEATS = ['H1', 'sfr', 'CO', 'H1_ew']
RA0, DEC0, INC, PA = 10.6847, 41.2690, np.radians(77.7), np.radians(37.7)
ARCSEC_PER_KPC = 206265.0 / 744.0


def spatial(ra, dec):
    """返回 (R, x_major, y_minor)，单位 kpc。x 沿长轴(PA 方向)，y 沿短轴(去投影)。"""
    dx = (ra - RA0) * np.cos(np.radians(DEC0)); dy = dec - DEC0
    x = dx * np.cos(PA) + dy * np.sin(PA)
    y = (-dx * np.sin(PA) + dy * np.cos(PA)) / np.cos(INC)
    k = 3600.0 / ARCSEC_PER_KPC
    return np.sqrt(x ** 2 + y ** 2) * k, x * k, y * k


def fit_eval(Zraw, y, sig, in_tr, region):
    mu, sd = Zraw[in_tr].mean(0), Zraw[in_tr].std(0) + 1e-12
    D = np.column_stack([np.ones(len(Zraw)), (Zraw - mu) / sd])
    w = 1.0 / sig[in_tr] ** 2
    coef = np.linalg.solve(D[in_tr].T @ (w[:, None] * D[in_tr]), D[in_tr].T @ (w * y[in_tr]))
    pull = (D @ coef - y) / sig
    out = dict(coef=coef, mu=mu, sd=sd, pull=pull,
               chi2_tr=float(np.mean(pull[in_tr] ** 2)),
               chi2_te=float(np.mean(pull[~in_tr] ** 2)),
               chi2_all=float(np.mean(pull ** 2)),
               pmed=float(np.median(pull)), pstd=float(np.std(pull)))
    out['arm_pull'] = float(np.median(pull[region])) if region is not None and region.any() else np.nan
    out['arm_chi2'] = float(np.mean(pull[region] ** 2)) if region is not None and region.any() else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', default='data/M31_1d_array_nodust.h5')
    ap.add_argument('--err-floor', type=float, default=0.2)
    ap.add_argument('--floor-frac', type=float, default=0.01)
    ap.add_argument('--mask-npz', default=None)
    ap.add_argument('--outdir', default='results/spatial_terms')
    args = ap.parse_args()

    with h5py.File(args.h5, 'r') as f:
        keys = ['ra', 'dec', 'qpah', 'qpah_err', 'CO_sigma'] + BASE_FEATS
        d = {k: f[k][:] for k in keys if k in f}
    sel = ((d['qpah'] >= 0) & (d['qpah_err'] > 0) & np.isfinite(d['H1']) & (d['H1'] >= 0)
           & np.isfinite(d['H1_ew']) & (d['H1_ew'] >= 0) & np.isfinite(d['sfr'])
           & np.isfinite(d['CO']) & np.isfinite(d['CO_sigma']) & (d['CO_sigma'] > 0))
    ra, dec = d['ra'][sel].astype(float), d['dec'][sel].astype(float)
    X = np.maximum(np.column_stack([d[k][sel] for k in BASE_FEATS]).astype(float), 0.0)
    y = d['qpah'][sel].astype(float)
    sig = np.sqrt(d['qpah_err'][sel].astype(float) ** 2 + args.err_floor ** 2)
    n = len(y)
    R, X_MAJ, Y_MIN = spatial(ra, dec)   # 长轴 / 短轴（此前变量名写反了）

    region = None
    if args.mask_npz:
        zm = np.load(args.mask_npz)
        if len(zm['region']) != n:
            raise SystemExit('mask 长度 %d != 样本 %d' % (len(zm['region']), n))
        region = zm['region'].astype(bool)

    np.random.seed(0)
    ang = (np.arctan2(dec - 41.25, ra - 10.75) + 2 * np.pi) % (2 * np.pi)
    sec = np.digitize(ang, np.linspace(0, 2 * np.pi, 11)) - 1
    tr_s = np.random.choice(np.arange(10), size=6, replace=False)
    in_tr = np.isin(sec, tr_s)

    floors = np.array([max(1e-9, args.floor_frac * np.median(X[in_tr, j][X[in_tr, j] > 0]))
                       for j in range(X.shape[1])])
    Zlog = np.log10(X + floors[None, :])
    logR = np.log10(R + 0.1)

    SETS = [
        ('A  base(4)',        [Zlog],                    5),
        ('B  base+R',         [Zlog, R[:, None]],        6),
        ('C  base+R+y(短轴/南北)', [Zlog, R[:, None], Y_MIN[:, None]], 7),
        ('D  base+R+y+x',     [Zlog, R[:, None], Y_MIN[:, None], X_MAJ[:, None]], 8),
        ('E  base+logR',      [Zlog, logR[:, None]],     6),
        ('F  base+logR+y+x',  [Zlog, logR[:, None], Y_MIN[:, None], X_MAJ[:, None]], 8),
    ]
    n_eff = n / 7.02
    print('样本 n=%d | 有效独立 ≈%.0f | 每参数有效样本（5 参数时）≈ %.0f'
          % (n, n_eff, n_eff / 5))
    print('训练扇区 n=%d | 测试扇区 n=%d' % (in_tr.sum(), (~in_tr).sum()))
    print()
    hdr = '%-20s %3s %11s %11s %11s %10s %11s %11s' % (
        '特征集', 'p', 'chi2_train', 'chi2_test', 'chi2_all', 'pull_all', 'arm_pull', 'arm_chi2')
    print(hdr); print('-' * len(hdr))
    res = {}
    for name, parts, p in SETS:
        Zraw = np.column_stack(parts)
        out = fit_eval(Zraw, y, sig, in_tr, region)
        res[name] = (out, p, Zraw)
        print('%-20s %3d %11.4f %11.4f %11.4f %+10.3f %+11.3f %11.3f'
              % (name, p, out['chi2_tr'], out['chi2_te'], out['chi2_all'],
                 out['pmed'], out['arm_pull'], out['arm_chi2']))

    print()
    print('=== 分半径的 chi2：A(基线) vs C(base+R+y) ===')
    edges = [0, 5, 7, 9, 11, 13, 15, 30]
    print('%8s %10s %12s %12s' % ('R(kpc)', 'n_test', 'chi2_A', 'chi2_C'))
    for a, b in zip(edges[:-1], edges[1:]):
        m = (~in_tr) & (R >= a) & (R < b)
        if m.sum() < 50:
            continue
        pa = res['A  base(4)'][0]['pull'][m]
        pc = res['C  base+R+y'][0]['pull'][m]
        print('%4d-%3d %10d %12.3f %12.3f' % (a, b, m.sum(),
                                              np.mean(pa ** 2), np.mean(pc ** 2)))

    # ---------------- 图 ----------------
    os.makedirs(args.outdir, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    edges2 = np.arange(0, 26, 1.0)
    for nm, col in [('A  base(4)', 'gray'), ('B  base+R', 'steelblue'),
                    ('C  base+R+y', 'crimson')]:
        out = res[nm][0]; bc, bm, bn = [], [], []
        for a, b in zip(edges2[:-1], edges2[1:]):
            m = (R >= a) & (R < b)
            if m.sum() >= 100:
                bc.append(0.5 * (a + b)); bm.append(np.mean(out['pull'][m] ** 2))
                bn.append(m.sum())
        ax[0].plot(bc, bm, 'o-', color=col, lw=2, ms=4, label=nm)
    ax[0].axhline(1.0, color='k', ls='--', lw=1)
    ax[0].set_xlabel('R (kpc, deprojected)'); ax[0].set_ylabel(r'$\chi^2_{\rm red}$')
    ax[0].set_title('chi2 vs radius: base vs +R'); ax[0].legend(fontsize=9); ax[0].grid(alpha=.3)

    for nm, col in [('A  base(4)', 'gray'), ('B  base+R', 'steelblue'), ('C  base+R+y', 'crimson')]:
        out = res[nm][0]; bc, bq = [], []
        for a, b in zip(edges2[:-1], edges2[1:]):
            m = (R >= a) & (R < b)
            if m.sum() >= 100:
                bc.append(0.5 * (a + b)); bq.append(np.median(out['pull'][m]))
        ax[1].plot(bc, bq, 'o-', color=col, lw=2, ms=4, label=nm)
    ax[1].axhline(0.0, color='k', ls='--', lw=1)
    ax[1].set_xlabel('R (kpc)'); ax[1].set_ylabel('pull median')
    ax[1].set_title('pull median vs radius'); ax[1].legend(fontsize=9); ax[1].grid(alpha=.3)

    names = [s[0] for s in SETS]
    ax[2].bar(range(len(names)), [res[n][0]['chi2_all'] for n in names], color='steelblue')
    ax[2].axhline(1.0, color='k', ls='--', lw=1)
    ax[2].set_xticks(range(len(names)))
    ax[2].set_xticklabels([n.split()[0] for n in names])
    ax[2].set_ylabel(r'$\chi^2_{\rm red}$ (all pixels)')
    ax[2].set_title('all-pixel chi2 by feature set'); ax[2].grid(alpha=.3, axis='y')
    fig.tight_layout()
    f = os.path.join(args.outdir, 'fig_spatial_terms.png')
    fig.savefig(f, dpi=160, bbox_inches='tight'); plt.close(fig)
    print('\nsaved', f)
    # 系数表
    print()
    for nm in ['B  base+R', 'C  base+R+y', 'D  base+R+y+x']:
        out, p, _ = res[nm]
        cols = BASE_FEATS + ['R', 'y_minor', 'x_major', 'logR']
        print('%s: 截距=%+.3f | ' % (nm, out['coef'][0]) +
              '  '.join('%s=%+.3f' % (c, v) for c, v in zip(cols[:p - 1], out['coef'][1:])))


if __name__ == '__main__':
    main()
