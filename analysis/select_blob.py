#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
select_blob.py — 从 arm_blob_labels.npz 里按编号取出一块（或把几块合并成一条链），
写成下游脚本能吃的 mask npz。

用法:
  # 单块
  python select_blob.py --labels results/armregion/arm_blob_labels.npz \
      --blob-id 25 --out results/armregion/mask_blob25.npz
  # 一条悬臂（多块合并）
  python select_blob.py --labels results/armregion/arm_blob_labels.npz \
      --blob-id 35 38 40 46 51 --out results/armregion/mask_arm_35_51.npz
"""
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', default='results/armregion/arm_blob_labels.npz')
    ap.add_argument('--blob-id', type=int, nargs='+', required=True,
                    help='一个或多个编号；多个则合并（用于串成一条悬臂）')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    z = np.load(args.labels)
    lab = z['px_lab']
    region = np.isin(lab, args.blob_id)
    if region.sum() == 0:
        raise SystemExit('编号 %s 都不存在；可用编号: %s'
                         % (args.blob_id, np.unique(lab[lab > 0]).tolist()))
    found = [int(b) for b in args.blob_id if np.any(lab == b)]
    missing = [int(b) for b in args.blob_id if not np.any(lab == b)]
    R = z.get('R')
    print('合并编号 %s（%d 块）→ %d 像素' % (found, len(found), int(region.sum())))
    if missing:
        print('  ⚠ 不存在（已跳过）: %s' % missing)
    if R is not None:
        print('  R med=%.1f kpc (5-95: %.1f-%.1f) | RA %.3f..%.3f | Dec %.3f..%.3f'
              % (np.median(R[region]), np.percentile(R[region], 5),
                 np.percentile(R[region], 95),
                 z['ra'][region].min(), z['ra'][region].max(),
                 z['dec'][region].min(), z['dec'][region].max()))
    print('  chi2=%.3f（其余 %.3f）'
          % (np.mean(z['chi'][region] ** 2), np.mean(z['chi'][~region] ** 2)))
    np.savez(args.out, ra=z['ra'], dec=z['dec'], region=region, chi=z['chi'])
    print('saved', args.out)


if __name__ == '__main__':
    main()
