#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
02_compare_datasets.py — 对比 matched h5 与旧 h5（不做训练）。

对比内容：
1) 两个数据集在 M31_power_law.ipynb 相同筛选下的样本量与每个量的分布
   (qpah, CO, H1, H1_ew, sfr, dust_density)。
2) matched h5 内部自带逐像素配对：同一 qPAH 像素上
   新方法 <col>（exact block） vs 旧方法 <col>_oldcenter（中心最近邻），
   输出差异统计（线性百分位 + 正值 log10 分布）与散点图。

输出：
    tmp/dataset_comparison.json / .txt
    sanity_output/fig6_dataset_oldcenter_vs_new.png
"""
import os, time, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, 'data')
OUT = os.path.join(ROOT, 'sanity_output')
TMP = os.path.join(ROOT, 'tmp')
os.makedirs(OUT, exist_ok=True)
os.makedirs(TMP, exist_ok=True)

T0 = time.time()
def log(*a):
    print('[%.0fs]' % (time.time() - T0), *a, flush=True)


def training_filter(*arrays):
    """M31_power_law.ipynb 中的筛选：qpah>0 & qpah_err>0 & dust>0 & H1>=0 & H1_ew>=0"""
    qpah, qpah_err, dust, H1, H1_ew = arrays
    idx = (qpah > 0) & (qpah_err > 0) & (dust > 0) & (H1 >= 0) & (H1_ew >= 0)
    return idx


def qtable(v):
    v = np.asarray(v)
    fin = v[np.isfinite(v)]
    if fin.size == 0:
        return dict(n=0)
    q = np.percentile(fin, [2.5, 16, 50, 84, 97.5])
    return dict(n=int(fin.size), p025=float(q[0]), p16=float(q[1]),
                median=float(q[2]), p84=float(q[3]), p975=float(q[4]))


def main():
    report = ['']
    def p(*a):
        s = ' '.join(str(x) for x in a)
        print(s, flush=True)
        report.append(s)

    # ---------- 1. 两个数据集概览 ----------
    matched_path = os.path.join(DATA, 'M31_1d_array_matched.h5')
    old_path = os.path.join(DATA, 'M31_1d_array.h5')
    feats = {'qpah': 'qpah', 'CO': 'CO', 'H1': 'H1', 'H1_ew': 'H1_ew',
             'sfr': 'sfr', 'dust_density': 'dust_density'}

    read_keys = {'qpah', 'qpah_err', 'CO', 'H1', 'H1_ew', 'sfr', 'dust_density'}
    with h5py.File(old_path, 'r') as f:
        old_raw = {k: f[k][:] for k in read_keys}
    with h5py.File(matched_path, 'r') as f:
        m_raw = {k: f[k][:] for k in read_keys}
        m_oc = {k: f[k + '_oldcenter'][:] for k in ['CO', 'H1', 'H1_ew', 'sfr', 'dust_density']}

    idx_old = training_filter(old_raw['qpah'], old_raw['qpah_err'], old_raw['dust_density'],
                              old_raw['H1'], old_raw['H1_ew'])
    idx_new = training_filter(m_raw['qpah'], m_raw['qpah_err'], m_raw['dust_density'],
                              m_raw['H1'], m_raw['H1_ew'])
    p('old h5 总行数:', old_raw['qpah'].size, ' 筛选后:', int(idx_old.sum()))
    p('matched h5 总行数:', m_raw['qpah'].size, ' 筛选后:', int(idx_new.sum()))

    summary = {'old': {}, 'matched': {}}
    p('\n%-14s %-26s %-26s' % ('quantity', 'OLD (center-sample h5)', 'MATCHED (exact block h5)'))
    for k in feats:
        vo = qtable(old_raw[k][idx_old])
        vn = qtable(m_raw[k][idx_new])
        summary['old'][k] = vo
        summary['matched'][k] = vn
        p('%-14s n=%6d med=%-12.4g p16/84=[%.3g, %.3g]  |  n=%6d med=%-12.4g p16/84=[%.3g, %.3g]'
          % (k, vo['n'], vo['median'], vo['p16'], vo['p84'],
             vn['n'], vn['median'], vn['p16'], vn['p84']))

    # ---------- 2. matched h5 内部逐像素配对：exact block vs old center ----------
    p('\n=== paired (matched h5 内, 同 qPAH 像素, exact block vs oldcenter) ===')
    pair_summary = {}
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    axl = list(axes.flat)
    for (col,) in [('CO',), ('H1',), ('H1_ew',), ('sfr',), ('dust_density',)]:
        new = m_raw[col][idx_new]
        old = m_oc[col][idx_new]
        both = np.isfinite(new) & np.isfinite(old)
        n = int(both.sum())
        diff = new[both] - old[both]
        st = dict(n=n)
        if n:
            st['diff_pct'] = [float(x) for x in np.percentile(diff, [5, 16, 50, 84, 95])]
            st['absdiff_p50_p90'] = [float(x) for x in np.percentile(np.abs(diff), [50, 90])]
        pos = both & (new > 0) & (old > 0)
        if pos.sum() > 100:
            dl = np.log10(new[pos]) - np.log10(old[pos])
            st['log10_pct'] = [float(x) for x in np.percentile(dl, [16, 50, 84])]
            st['median_abs_dlog'] = float(np.median(np.abs(dl)))
        pair_summary[col] = st
        p('%-14s n=%d  diff p16/p50/p84=[%.4g, %.4g, %.4g]  |Δ| p50/p90=[%.3g, %.3g]'
          % (col, st['n'], st.get('diff_pct', [np.nan]*5)[1], st.get('diff_pct', [np.nan]*5)[2],
             st.get('diff_pct', [np.nan]*5)[3], st.get('absdiff_p50_p90', [np.nan, np.nan])[0],
             st.get('absdiff_p50_p90', [np.nan, np.nan])[1]))
        if 'log10_pct' in st:
            p('   log10(new/old): p16/p50/p84=[%.4f, %.4f, %.4f]  med|Δlog|=%.4f'
              % tuple(st['log10_pct'] + [st['median_abs_dlog']]))

        ax = axl.pop(0)
        x = old[both]
        y = new[both]
        if x.size == 0:
            ax.text(0.5, 0.5, 'no data', ha='center'); ax.set_title(col); continue
        if col == 'sfr':
            # SFR 动态范围大，先取 log10 再 hexbin
            xl = np.log10(x[x > 0]); yl = np.log10(y[x > 0])
            if xl.size > 100:
                v = np.concatenate([xl, yl])
                lo = np.nanpercentile(v, 0.1); hi = np.nanpercentile(v, 99.9)
                hb = ax.hexbin(xl, yl, gridsize=100, bins='log', cmap='viridis', mincnt=1,
                               extent=(lo, hi, lo, hi))
                fig.colorbar(hb, ax=ax, shrink=0.85)
                ax.plot([lo, hi], [lo, hi], 'r--', lw=1)
                ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
                t0, t1 = int(np.ceil(lo)), int(np.floor(hi))
                tk = list(range(t0, t1 + 1))
                ax.set_xticks(tk); ax.set_yticks(tk)
                ax.set_xticklabels([r'$10^{%d}$' % t for t in tk])
                ax.set_yticklabels([r'$10^{%d}$' % t for t in tk])
            else:
                ax.scatter(x, y, s=0.3)
        else:
            if x.size > 500:
                hb = ax.hexbin(x, y, gridsize=80, bins='log', cmap='viridis', mincnt=1)
                fig.colorbar(hb, ax=ax, shrink=0.85)
            else:
                ax.scatter(x, y, s=0.3)
            lim = (min(np.nanmin(x), np.nanmin(y)), max(np.nanmax(x), np.nanmax(y)))
            ax.plot(lim, lim, 'r--', lw=1)
        ax.set_xlabel('oldcenter'); ax.set_ylabel('exact block')
        ax.set_title('%s  (paired n=%d)' % (col, n))

    for ax in axl:
        ax.axis('off')
    fig.suptitle('Matched h5 rows: exact block (new) vs old center-sample (old)')
    fig.savefig(os.path.join(OUT, 'fig6_dataset_oldcenter_vs_new.png'), dpi=140)
    plt.close(fig)
    p('figure -> sanity_output/fig6_dataset_oldcenter_vs_new.png')

    out = dict(old_summary=summary['old'], matched_summary=summary['matched'],
               paired=pair_summary, old_rows_filtered=int(idx_old.sum()),
               matched_rows_filtered=int(idx_new.sum()))
    with open(os.path.join(TMP, 'dataset_comparison.json'), 'w') as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(TMP, 'dataset_comparison.txt'), 'w') as f:
        f.write('\n'.join(report))
    log('DONE -> tmp/dataset_comparison.json / .txt')


if __name__ == '__main__':
    main()
