#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
co_alternate_rounds.py — 检查 CO 交替优化是否收敛（读 results/<tag>/co_true_train.npz）。

train_co_alternate.py 在 --alt-rounds>1 时，会把每轮 2a / 2b 结束时的 CO_true 快照
写进 npz（键 co_true_s2a / co_true_s2b，形状 (R, N)）。本脚本回答三个问题：

  1. 每轮 2b 相对同轮 2a 把 CO 推动了多大？（2b 是否在"推翻"2a 的结论）
  2. 相邻轮之间 CO_true 还漂不漂？（漂移应随轮数下降 = 收敛）
  3. 各轮的 CO_true 一致性如何？（相关系数 / 位移）

用法:
  python qpah_pipeline/analysis/co_alternate_rounds.py \
      --npz qpah_pipeline/results/coalt_full/co_true_train.npz
"""
import argparse
import numpy as np


def dist(a, b, sig):
    """|a-b|/sig 的 (中位, p84, max)。"""
    d = np.abs(a - b) / sig
    return float(np.median(d)), float(np.percentile(d, 84)), float(d.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True, help='results/<tag>/co_true_train.npz')
    ap.add_argument('--out', default=None, help='可选：把逐轮摘要存成 json')
    args = ap.parse_args()

    z = np.load(args.npz)
    if 'co_true_s2a' not in z or 'co_true_s2b' not in z:
        raise SystemExit('该 npz 没有 co_true_s2a/co_true_s2b —— '
                         '需用 --alt-rounds>1（默认 3）重跑训练。现有键: %s'
                         % sorted(z.keys()))
    s2a = z['co_true_s2a'].astype(np.float64)     # (R, N)
    s2b = z['co_true_s2b'].astype(np.float64)
    sig = z['co_sigma'].astype(np.float64)
    meas = z['co_meas'].astype(np.float64)
    R = s2a.shape[0]

    print('rounds R = %d | train pixels N = %d' % (R, s2a.shape[1]))
    print('（单位都是 |ΔCO|/σ_CO；σ_CO 为每像素 CO 测量误差）\n')
    print('%-6s %-26s %-26s' % ('round', '|CO(2b_r)-CO(2a_r)|/sig', '|CO(2b_r)-CO(2b_{r-1})|/sig'))
    print('%-6s %-26s %-26s' % ('', 'med     p84     max', 'med     p84     max'))

    rows, prev = [], None
    for r in range(R):
        m_ab = dist(s2b[r], s2a[r], sig)
        if prev is None:
            m_rr = (float('nan'), float('nan'), float('nan'))
        else:
            m_rr = dist(s2b[r], prev, sig)
        print('%-6d %7.3f %7.3f %7.2f     %7.3f %7.3f %7.2f'
              % (r + 1, m_ab[0], m_ab[1], m_ab[2], m_rr[0], m_rr[1], m_rr[2]))
        rows.append(dict(round=r + 1, move_2b_vs_2a_med=m_ab[0],
                         move_vs_prev_round_med=(None if prev is None else m_rr[0])))
        prev = s2b[r]

    print('\n起点(第1轮2a) → 终点(末轮2b): med=%.3f p84=%.3f max=%.2f'
          % dist(s2b[-1], s2a[0], sig))
    print('相对测量值 |CO_true-CO_meas|/sig: 第1轮2a med=%.3f → 末轮2b med=%.3f'
          % (np.median(np.abs(s2a[0] - meas) / sig),
             np.median(np.abs(s2b[-1] - meas) / sig)))

    if R > 1:
        print('\n相邻轮相关系数（→1 表示轮间不再变化 = 收敛）:')
        for r in range(1, R):
            c = float(np.corrcoef(s2b[r - 1], s2b[r])[0, 1])
            print('  round %d vs %d : %.5f' % (r, r + 1, c))

    if args.out:
        import json
        json.dump(rows, open(args.out, 'w'), indent=1)
        print('\nsaved', args.out)


if __name__ == '__main__':
    main()
