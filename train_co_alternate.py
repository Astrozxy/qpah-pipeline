#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_co_alternate.py — CO forward-modelling 交替训练（Stage-1 + Stage-2a/2b × N 轮）。

数据集要求：build_dataset.py 新版产物（data/M31_1d_array_full.h5 或 agb，
含 CO_sigma / CO_det 列）。

口径（现行，2026-10 起）：**没有 val**。
  - 10 个方位扇区 → 6 个训练扇区 / 4 个【空间测试扇区】；
  - 训练扇区【全部像素】进迭代优化，**不切 val、不早停、不选模型**；
  - 空间测试扇区完全不参与训练、也不参与任何训练期判断，只在最后评估 + 画图。

流程（全部固定轮数，预先约定，训练中不看任何留出数据）：
  Stage-1 : 只用 CO/CO_sigma > 3 的探测像素训练 MLP，跑满 --stage1-epochs
            （异方差 Pseudo-Huber：σ_eff=sqrt(qpah_err²+floor²)，CO 固定为测量值；
             --no-err-weight 回退等权）；
  Stage-2 : 【交替优化】重复 --alt-rounds 轮，每轮做两步：
      2a: 冻结模型，只更新每像素 CO_true（高斯先验 (CO_true−CO_meas)²/σ²，CO_true≥0）；
      2b: 同时放开模型参数与 CO_true，联合微调，CO 先验始终保留防漂移。
  第 1 轮 CO_true 初值 = 测量 CO；之后 warm start（拿上一轮末态继续）。
  每轮末存 model_round<r>.pth + co_true_round<r>.npy（仅事后诊断）。
  最终模型 = 末轮末态（预先约定，不做任何选优）。

用法:
  python train_co_alternate.py --selection full
  python train_co_alternate.py --selection agb --tag coalt_agb

输出: results/<tag>/{model_stage1.pth, model.pth, co_true_final.npy, scale.json,
                    train.json, co_true_train.npz, model_round<r>.pth, ...}
"""
import os, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import h5py

REPO = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(REPO)
DATA = os.path.join(ROOT, 'data')
RESULTS = os.path.join(REPO, 'results')
FEATURES = ['dust_density', 'H1', 'sfr', 'CO', 'H1_ew']
SEL_FILE = {'full': 'M31_1d_array_full.h5', 'agb': 'M31_1d_array_agb.h5'}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_T0 = time.time()
random_state = 0


def log(*a):
    print('[%.0fs]' % (time.time() - _T0), *a, flush=True)


def set_seed(seed=random_state):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def phuber(r, delta=10.0):
    return delta ** 2 * (torch.sqrt(1 + (r / delta) ** 2) - 1)


def load_config(path):
    """读 JSON 配置；文件不存在则返回 {}（全部用内置默认值）。"""
    if path and os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def cfg_get(cfg, *keys, default=None):
    """嵌套取值：cfg_get(cfg, 'stage1', 'lr', default=0.01)。"""
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def model_norm(model):
    """模型全部参数的 L2 范数（用于观察参数是否"乱飘"）。"""
    with torch.no_grad():
        return float(torch.sqrt(sum((p ** 2).sum() for p in model.parameters())))


def layer_norms(model):
    """逐层参数范数（诊断用）。"""
    with torch.no_grad():
        return {n: float(p.norm()) for n, p in model.named_parameters()}


def load_h5(h5path):
    keys = ['ra', 'dec', 'qpah', 'qpah_err', 'sfr', 'CO', 'H1', 'H1_ew',
            'dust_density', 'CO_sigma', 'CO_det']
    with h5py.File(h5path, 'r') as f:
        d = {k: f[k][:] for k in keys if k in f}
    base = ((d['qpah'] > 0) & (d['qpah_err'] > 0) & (d['dust_density'] > 0)
            & (d['H1'] >= 0) & (d['H1_ew'] >= 0))
    have_co = ('CO_sigma' in d) and (np.isfinite(d['CO']) & np.isfinite(d['CO_sigma'])
                                     & (d['CO_sigma'] > 0))
    sel = base & have_co
    out = {k: v[sel] for k, v in d.items()}
    out['n_total'] = int(sel.sum())
    return out


def sector_split(ds, n_sectors=10, train_frac=0.6, seed=0):
    """按方位角分 n_sectors 个扇区：取 train_frac 作训练扇区，其余作【空间测试扇区】。

    注意（现行口径）：**不再从训练扇区里切 val**。训练扇区的全部像素都进迭代优化；
    空间测试扇区完全不参与训练、也不参与任何训练期判断（无早停、无选模型），
    只在最后做评估与画图。
    """
    ang = np.arctan2(ds['dec'] - 41.25, ds['ra'] - 10.75)
    ang = (ang + 2 * np.pi) % (2 * np.pi)
    sec = np.digitize(ang, np.linspace(0, 2 * np.pi, n_sectors + 1)) - 1
    n_tr = int(round(n_sectors * train_frac))
    tr = np.random.choice(np.arange(n_sectors), size=n_tr, replace=False)
    te = np.setdiff1d(np.arange(n_sectors), tr)
    train_idx = np.where(np.isin(sec, tr))[0]
    test_idx = np.where(np.isin(sec, te))[0]
    return tr, te, train_idx, test_idx


def main():
    # ---- 先解析 --config，用 JSON 里的值作为其余参数的默认值 ----
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument('--config', default=os.path.join(REPO, 'train_config.json'))
    cfg_path = _pre.parse_known_args()[0].config
    cfg = load_config(cfg_path)

    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=cfg_path)
    ap.add_argument('--selection', choices=['full', 'agb'],
                    default=cfg_get(cfg, 'data', 'selection', default='full'))
    ap.add_argument('--dataset', default=cfg_get(cfg, 'data', 'h5', default=None))
    ap.add_argument('--tag', default=None)
    ap.add_argument('--snr-thresh', type=float,
                    default=cfg_get(cfg, 'data', 'snr_thresh', default=3.0))
    ap.add_argument('--no-co-update', action='store_true',
                    help='完全不允许 CO 更新：跳过 Stage-2（不做 CO 反推/联合微调），'
                         '最终模型 = Stage-1 模型（纯监督基线）')
    ap.add_argument('--clip-co-neg', action='store_true',
                    help='输入时把 CO 观测裁剪到 ≥0（max(CO,0)）：负 CO 不物理，'
                         '且 Stage-1 只在正 CO 上训练过。裁剪会同时作用于标准化/训练/评估')
    ap.add_argument('--stage1-epochs', type=int,
                    default=cfg_get(cfg, 'stage1', 'epochs', default=500),
                    help='Stage-1 固定轮数（无 val 早停，跑满）')
    ap.add_argument('--stage1-ckpt-every', type=int, default=0,
                    help='Stage-1 快照间隔（epoch；0 = 用 stage1_epochs//10），'
                         '快照存 results/<tag>/stage1_ckpt/，供事后逐 epoch 诊断')
    ap.add_argument('--stage1-lr', type=float,
                    default=cfg_get(cfg, 'stage1', 'lr', default=0.01))
    ap.add_argument('--co-only-steps', type=int,
                    default=cfg_get(cfg, 'stage2', 'co_only_steps', default=1000))
    ap.add_argument('--alt-rounds', type=int,
                    default=cfg_get(cfg, 'stage2', 'alt_rounds', default=5),
                    help='Stage-2 交替优化轮数（每轮 = 2a 更新 CO_true + 2b 联合微调）')
    ap.add_argument('--co-lr', type=float,
                    default=cfg_get(cfg, 'stage2', 'co_lr', default=1e-3))
    ap.add_argument('--joint-epochs', type=int,
                    default=cfg_get(cfg, 'stage2', 'joint_epochs', default=100),
                    help='每轮 2b 的固定轮数（无 val 早停，跑满）')
    ap.add_argument('--joint-model-lr', type=float,
                    default=cfg_get(cfg, 'stage2', 'joint_model_lr', default=1e-4))
    ap.add_argument('--joint-co-lr', type=float,
                    default=cfg_get(cfg, 'stage2', 'joint_co_lr', default=1e-3))
    ap.add_argument('--prior-lambda', type=float,
                    default=cfg_get(cfg, 'stage2', 'prior_lambda', default=1.0))
    ap.add_argument('--batch-size', type=int,
                    default=cfg_get(cfg, 'optim', 'batch_size', default=4096))
    ap.add_argument('--err-floor', type=float,
                    default=cfg_get(cfg, 'loss', 'err_floor', default=0.2))
    ap.add_argument('--huber-delta', type=float,
                    default=cfg_get(cfg, 'loss', 'huber_delta', default=10.0))
    ap.add_argument('--no-err-weight', action='store_true',
                    default=not cfg_get(cfg, 'loss', 'err_weight', default=True),
                    help='回退等权 Pseudo-Huber（默认由 config 的 loss.err_weight 决定）')
    ap.add_argument('--weight-decay', type=float,
                    default=cfg_get(cfg, 'optim', 'weight_decay', default=1e-4))
    ap.add_argument('--hidden', nargs='+', type=int,
                    default=list(cfg_get(cfg, 'model', 'hidden', default=[32, 32])),
                    help='MLP 隐层宽度，例：--hidden 64 64 32')
    ap.add_argument('--dropout', type=float,
                    default=cfg_get(cfg, 'model', 'dropout', default=0.0))
    ap.add_argument('--rebalance-by', choices=['none', 'qpah', 'co', 'joint'],
                    default=cfg_get(cfg, 'stage1', 'rebalance_by', default='qpah'),
                    help='Stage-1 采样重加权依据：none/qpah/co/joint（co 与 joint 用于加强高 CO 端）')
    ap.add_argument('--n-bins', type=int,
                    default=cfg_get(cfg, 'stage1', 'n_bins', default=10))
    args = ap.parse_args()

    # ---- 结构性参数 ----
    seed = int(cfg_get(cfg, 'split', 'random_state', default=0))
    n_sectors = int(cfg_get(cfg, 'split', 'n_sectors', default=10))
    train_frac = float(cfg_get(cfg, 'split', 'train_sector_frac', default=0.6))
    hidden = tuple(args.hidden)
    dropout = float(args.dropout)
    s1_step = int(cfg_get(cfg, 'stage1', 'sched_step', default=100))
    s1_gamma = float(cfg_get(cfg, 'stage1', 'sched_gamma', default=0.8))
    s1_rebalance_by = str(args.rebalance_by)
    s1_nbins = int(args.n_bins)
    cap_factor = float(cfg_get(cfg, 'stage1', 'weight_cap_factor', default=2.0))
    clamp_min = float(cfg_get(cfg, 'stage2', 'clamp_min', default=0.0))

    tag = args.tag or ('coalt_' + args.selection)
    outdir = os.path.join(RESULTS, tag)
    os.makedirs(outdir, exist_ok=True)

    print('device:', device, flush=True)
    log('config: %s' % (cfg_path if cfg_path and os.path.exists(cfg_path)
                        else '(未找到 %s，使用内置默认值)' % cfg_path))
    log('model: hidden=%s dropout=%.2f | split: %d sectors, train_frac=%.2f, seed=%d '
        '(无 val：训练扇区全部像素进迭代)'
        % (list(hidden), dropout, n_sectors, train_frac, seed))
    set_seed(seed)
    h5path = args.dataset or os.path.join(DATA, SEL_FILE[args.selection])
    ds = load_h5(h5path)
    log('valid rows (有 CO 观测):', ds['n_total'])

    # ---- 静态特征标准化（全部有效行，用测量 CO 的分布） ----
    Xraw = np.column_stack([ds[f] for f in FEATURES])
    mean = Xraw.mean(0, keepdims=True)
    std = Xraw.std(0, keepdims=True) + 1e-8
    Xn = (Xraw - mean) / std
    y = ds['qpah'].astype(np.float32)
    qpah_err = ds['qpah_err'].astype(np.float32)
    # sig_eval：评估用的 σ（始终基于 qpah_err，保证两版可比）
    # sig_eff ：loss 用的 σ（--no-err-weight 时退化为等权 =1）
    sig_eval = np.sqrt(qpah_err ** 2 + args.err_floor ** 2)
    sig_eff = sig_eval if not args.no_err_weight else np.ones_like(qpah_err)
    co_meas_raw = ds['CO'].astype(np.float32)
    if args.clip_co_neg:
        co_meas = np.maximum(co_meas_raw, 0.0)
        log('--clip-co-neg: CO 输入裁剪到 ≥0（负观测 %d px = %.1f%% 被置 0）'
            % (int((co_meas_raw < 0).sum()), 100.0 * (co_meas_raw < 0).mean()))
    else:
        co_meas = co_meas_raw
    co_sig = np.maximum(ds['CO_sigma'].astype(np.float32), 1e-6)
    det = (co_meas / co_sig) > args.snr_thresh
    log('detection(CO/σ>%.0f): %d / %d (%.1f%%)'
        % (args.snr_thresh, int(det.sum()), len(y), 100 * det.mean()))
    log('qPAH loss: %s (err-floor=%.2f, huber-delta=%.2f)'
        % ('heteroscedastic Pseudo-Huber' if not args.no_err_weight
           else 'equal-weight Pseudo-Huber', args.err_floor, args.huber_delta))
    log('sigma_eff: med=%.3f p16=%.3f p84=%.3f | qpah_err: med=%.3f p84=%.3f'
        % (np.median(sig_eval), np.percentile(sig_eval, 16), np.percentile(sig_eval, 84),
           np.median(qpah_err), np.percentile(qpah_err, 84)))

    json.dump({'features': FEATURES, 'mean': mean.flatten().tolist(),
               'std': std.flatten().tolist(), 'transform': None,
               'model_hidden': list(hidden), 'model_dropout': dropout},
              open(os.path.join(outdir, 'scale.json'), 'w'), indent=1)

    tr_s, te_s, train_idx, test_idx = sector_split(
        ds, n_sectors=n_sectors, train_frac=train_frac, seed=seed)
    log('train sectors %s | test sectors %s' % (tr_s, te_s))
    log('train/test:', len(train_idx), len(test_idx),
        '（空间测试扇区不参与训练，只在最后评估+画图）')
    log('CO/σ>%.0f 探测像素: 训练扇区 %d | 测试扇区 %d（后者不再进入 Stage-1）'
        % (args.snr_thresh, int(det[train_idx].sum()), int(det[test_idx].sum())))

    Xt = torch.tensor(Xn, dtype=torch.float32).to(device)
    yt = torch.tensor(y, dtype=torch.float32).to(device)
    co_m = torch.tensor(co_meas, dtype=torch.float32).to(device)
    co_s = torch.tensor(co_sig, dtype=torch.float32).to(device)
    se_t = torch.tensor(sig_eff, dtype=torch.float32).to(device)

    def build_X(co_values, idx=None):
        """co_values: 与全体行等长的 CO 真值(原单位) device tensor → 标准化特征。"""
        c = (co_values - mean[0][3]) / std[0][3]
        X = Xt.clone()
        X[:, 3] = c
        return X if idx is None else X[idx]

    # ============ Stage-1: CO>3σ 子集训练（固定轮数，无早停/无 val） ============
    log('Stage-1: train on CO/σ>%.0f pixels【仅训练扇区】(n=%d), 固定 %d epochs ...'
        % (args.snr_thresh, int(det[train_idx].sum()), args.stage1_epochs))
    model = MLPRegressor(hidden=hidden, dropout=dropout).to(device)
    log('  weight_decay=%.1e | ||W|| init=%.3f | hidden=%s'
        % (args.weight_decay, model_norm(model), list(hidden)))
    opt1 = optim.Adam(model.parameters(), lr=args.stage1_lr,
                      weight_decay=args.weight_decay)
    sch1 = optim.lr_scheduler.StepLR(opt1, step_size=s1_step, gamma=s1_gamma)
    # 只在【训练扇区】内取探测像素 —— 否则测试扇区像素会漏进 Stage-1 训练
    # （原实现是 idx1 = np.where(det)[0]，会把测试扇区的 3737 个探测像素也拿来训练）
    idx1 = train_idx[det[train_idx]]
    yy1 = y[idx1]
    co1 = co_meas[idx1]
    cnt = np.array([-1])
    if s1_rebalance_by == 'none':
        sw = np.ones(len(idx1))
    else:
        # 采样重加权：按 qPAH / CO / qPAH×CO 联合分箱，稀有格子权重更大。
        # 'co' 与 'joint' 用于加强高 CO 端的拟合（CO 与 PAH 争碳 -> 先升后降的拐点区）
        if s1_rebalance_by == 'qpah':
            cell = np.digitize(yy1, np.linspace(0, 10, s1_nbins + 1)) - 1
            ncell = s1_nbins
        elif s1_rebalance_by == 'co':
            edges = np.percentile(co1, np.linspace(0, 100, s1_nbins + 1))
            edges[-1] += 1e-9
            cell = np.digitize(co1, edges[:-1]) - 1
            ncell = s1_nbins
        elif s1_rebalance_by == 'joint':
            nb = max(2, int(round(np.sqrt(s1_nbins))))
            bq = np.digitize(yy1, np.linspace(0, 10, nb + 1)) - 1
            ce = np.percentile(co1, np.linspace(0, 100, nb + 1))
            ce[-1] += 1e-9
            bc = np.digitize(co1, ce[:-1]) - 1
            cell = bq * nb + bc
            ncell = nb * nb
        else:
            raise SystemExit('未知 rebalance_by=%r（可选 none/qpah/co/joint）' % s1_rebalance_by)
        cell = np.clip(cell, 0, ncell - 1)
        cnt = np.bincount(cell, minlength=ncell)
        sw = 1.0 / (cnt[cell] + 1)
        cap = np.percentile(sw, 95) * cap_factor
        sw = np.clip(sw, None, cap)
        sw = sw / sw.sum() * len(sw)
    log('  stage1 sampling: rebalance_by=%s n_bins=%d cap_factor=%.1f%s'
        % (s1_rebalance_by, s1_nbins, cap_factor,
           '' if s1_rebalance_by == 'none' else ' | 每格样本数 min=%d max=%d' % (cnt.min(), cnt.max())))
    sampler1 = WeightedRandomSampler(torch.tensor(sw, dtype=torch.float64),
                                     num_samples=len(idx1), replacement=True)
    loader1 = DataLoader(TensorDataset(Xt[idx1], yt[idx1], se_t[idx1]),
                         batch_size=args.batch_size, sampler=sampler1)
    s1_every = (args.stage1_ckpt_every if args.stage1_ckpt_every > 0
                else max(1, args.stage1_epochs // 10))
    ckpt_dir = os.path.join(outdir, 'stage1_ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)
    for ep in range(args.stage1_epochs):
        model.train()
        tot = 0.0
        for bx, by, bsig in loader1:
            opt1.zero_grad()
            loss = torch.mean(phuber((model(bx) - by) / bsig, args.huber_delta))
            loss.backward()
            opt1.step()
            tot += loss.item() * bx.size(0)
        sch1.step()
        # 无 val：不再选模型/早停，固定轮数跑完；只打印训练 loss 轨迹供事后检查
        if (ep + 1) % s1_every == 0 or (ep + 1) == args.stage1_epochs:
            log('  S1 ep %4d tr=%.4f' % (ep + 1, tot / len(idx1)))
            # 存 epoch 快照，供事后（含测试扇区）逐 epoch 诊断 —— 训练本身不看测试
            torch.save(model.state_dict(),
                       os.path.join(ckpt_dir, 'model_stage1_ep%04d.pth' % (ep + 1)))
    torch.save(model.state_dict(), os.path.join(outdir, 'model_stage1.pth'))
    log('Stage-1 done (%d epochs, 末态保存为 model_stage1.pth)  ||W||=%.3f'
        % (args.stage1_epochs, model_norm(model)))

    # ============ Stage-2: 交替优化（2a ↔ 2b），共 --alt-rounds 轮 ============
    # 交替坐标下降：每轮先冻结模型、只更新 CO_true（2a），再联合微调模型+CO_true（2b）。
    # 2b 改完模型后 2a 的最优性就失效了，所以要再交替回去 —— 这正是多轮的意义。
    # 第 1 轮 2a 的初值 = 测量 CO；第 r>1 轮以上一轮 2b 的 CO_true 为初值（warm start）。
    # CO_true 只对训练像素更新（val/test 一律用测量 CO，避免泄漏）。
    co_true = nn.Parameter(co_m[train_idx].clone())
    Xtr = Xt[train_idx]
    ytr = yt[train_idx]
    co_meas_tr = co_m[train_idx]
    co_sig_tr = co_s[train_idx]
    se_tr = se_t[train_idx]

    def co_loss_parts():
        """返回 (qPAH 数据项, CO 先验项)；先验项未乘 0.5·λ。"""
        Xb = Xtr.clone()
        Xb[:, 3] = (co_true - mean[0][3]) / std[0][3]
        q = torch.mean(phuber((model(Xb) - ytr) / se_tr, args.huber_delta))
        pr = torch.mean(((co_true - co_meas_tr) / co_sig_tr) ** 2)
        return q, pr

    def co_loss_full():
        q, pr = co_loss_parts()
        return q + 0.5 * args.prior_lambda * pr

    def co_stats(tag):
        """打印 CO_true 相对【测量值】的分布（衡量总共动了多少）。"""
        with torch.no_grad():
            c_np = co_true.detach().cpu().numpy().astype(np.float64)
            pull = (c_np - co_meas_tr.detach().cpu().numpy()) / \
                co_sig_tr.detach().cpu().numpy()
            n_at0 = int((c_np <= 1e-9).sum())
        log('  %s CO_true: med=%.3f p16=%.3f p84=%.3f | (CO_true-CO_meas)/sigma: '
            'med=%+.2f std=%.2f | clamp0=%.1f%%'
            % (tag, np.median(c_np), np.percentile(c_np, 16), np.percentile(c_np, 84),
               np.median(pull), np.std(pull), 100.0 * n_at0 / len(c_np)))

    def move_stats(a, b):
        """|a−b|/σ_CO 的中位/p84/max（a、b 为同长度 device tensor）。"""
        with torch.no_grad():
            d = ((a.detach() - b.detach()) / co_sig_tr).cpu().numpy()
        return (float(np.median(np.abs(d))), float(np.percentile(np.abs(d), 84)),
                float(np.abs(d).max()))

    n_tr = len(train_idx)
    s2a_hist, s2b_hist, round_hist = [], [], []   # 每轮 CO_true 快照 / 摘要
    prev_co = co_true.detach().clone()            # 上一轮结束时的 CO_true

    n_rounds_eff = 0 if args.no_co_update else args.alt_rounds
    if args.no_co_update:
        log('--no-co-update: 跳过 Stage-2（完全不允许 CO 更新），最终模型 = Stage-1 模型')
    for rnd in range(1, n_rounds_eff + 1):
        log('=' * 22 + ' alternate round %d/%d ' % (rnd, args.alt_rounds) + '=' * 22)

        # ---- 2a: 冻结模型，只更新 CO_true（warm start） ----
        opt_co = optim.Adam([co_true], lr=args.co_lr)
        every = max(1, args.co_only_steps // 10)
        log('Stage-2a: freeze model, update CO_true (%d steps)...' % args.co_only_steps)
        model.eval()
        s2a_loss = float('nan')
        for k in range(args.co_only_steps):
            opt_co.zero_grad()
            loss = co_loss_full()
            loss.backward()
            opt_co.step()
            with torch.no_grad():
                co_true.data.clamp_(min=clamp_min)
            s2a_loss = loss.item()
            if (k + 1) % every == 0 or (k + 1) == args.co_only_steps:
                with torch.no_grad():
                    q_now, pr_now = co_loss_parts()
                log('  R%d S2a step %4d loss=%.4f (data=%.4f prior=%.4f)  CO_true med=%.3f'
                    % (rnd, k + 1, loss.item(), q_now.item(),
                       0.5 * args.prior_lambda * pr_now.item(),
                       co_true.detach().median().item()))
        mv_prev = move_stats(co_true, prev_co)
        log('  R%d after-2a: |CO_true - 上一轮结束|/sigma  med=%.3f p84=%.3f max=%.2f'
            '   (R1 的"上一轮"= 测量 CO 初值)'
            % (rnd, mv_prev[0], mv_prev[1], mv_prev[2]))
        co_stats('R%d after-2a' % rnd)
        s2a_hist.append(co_true.detach().cpu().numpy().astype(np.float32).copy())

        # ---- 2b: 联合微调（模型 + CO_true） ----
        # 无 val：不做早停、不选模型，固定轮数跑完；只记录训练 loss 轨迹供事后检查
        log('Stage-2b: joint fine-tune (%d epochs, model_lr=%.0e co_lr=%.0e)...'
            % (args.joint_epochs, args.joint_model_lr, args.joint_co_lr))
        params = [{'params': model.parameters(), 'lr': args.joint_model_lr,
                   'weight_decay': args.weight_decay},
                  {'params': [co_true], 'lr': args.joint_co_lr, 'weight_decay': 0.0}]
        opt_j = optim.Adam(params)
        s2b_every = max(1, args.joint_epochs // 10)
        ep_tr_hist = []
        for ep in range(args.joint_epochs):
            model.train()
            order = np.random.permutation(n_tr)
            tot = 0.0
            for s in range(0, n_tr, args.batch_size):
                ib = order[s:s + args.batch_size]
                Xb = Xtr[ib].clone()
                Xb[:, 3] = (co_true[ib] - mean[0][3]) / std[0][3]
                opt_j.zero_grad()
                q = torch.mean(phuber((model(Xb) - ytr[ib]) / se_tr[ib], args.huber_delta))
                pr = torch.mean(((co_true[ib] - co_meas_tr[ib]) / co_sig_tr[ib]) ** 2)
                (q + 0.5 * args.prior_lambda * pr).backward()
                opt_j.step()
                with torch.no_grad():
                    co_true.data.clamp_(min=clamp_min)
                tot += q.item() * len(ib)
            ep_tr_hist.append(tot / n_tr)
            if (ep + 1) % s2b_every == 0 or (ep + 1) == args.joint_epochs:
                with torch.no_grad():
                    pr_ep = torch.mean(((co_true - co_meas_tr) / co_sig_tr) ** 2).item()
                log('  R%d S2b ep %4d tr=%.4f (CO prior=%.4f)'
                    % (rnd, ep + 1, tot / n_tr, 0.5 * args.prior_lambda * pr_ep))

        mv_ab = move_stats(co_true, torch.tensor(s2a_hist[-1], device=device))
        log('  R%d after-2b: |CO_true(2b末) - CO_true(2a)|/sigma  med=%.3f p84=%.3f max=%.2f'
            % (rnd, mv_ab[0], mv_ab[1], mv_ab[2]))
        co_stats('R%d after-2b' % rnd)
        # 每轮存一个【模型 + CO_true】配对快照，仅作事后诊断
        # （可用 eval_co_models.py 比较各轮；训练中不使用，测试扇区不参与）
        torch.save(model.state_dict(), os.path.join(outdir, 'model_round%d.pth' % rnd))
        np.save(os.path.join(outdir, 'co_true_round%d.npy' % rnd),
                co_true.detach().cpu().numpy())
        s2b_hist.append(co_true.detach().cpu().numpy().astype(np.float32).copy())
        round_hist.append(dict(round=rnd, s2a_loss=float(s2a_loss),
                               s2b_first_tr=float(ep_tr_hist[0]) if ep_tr_hist else None,
                               s2b_last_tr=float(ep_tr_hist[-1]) if ep_tr_hist else None,
                               move_2a_vs_prev_med=float(mv_prev[0]),
                               move_2b_vs_2a_med=float(mv_ab[0])))
        prev_co = co_true.detach().clone()

    # ---- 固定轮数跑完：最终模型 = 末轮末态（事先约定，不做任何选优） ----
    torch.save(model.state_dict(), os.path.join(outdir, 'model.pth'))
    np.save(os.path.join(outdir, 'co_true_final.npy'), co_true.detach().cpu().numpy())
    if args.no_co_update:
        final_note = ' (Stage-1 only, no CO update)'
        log('final model = Stage-1 末态（--no-co-update：CO 全程未参与更新）')
    else:
        final_note = ' (fixed schedule: end of round %d/%d)' % (args.alt_rounds, args.alt_rounds)
        log('alternate done: %d rounds (固定轮数，无 val 早停/选优) | final model = 末轮末态'
            % args.alt_rounds)

    # ============ CO_true 诊断（供 forward model 分析） ============
    with torch.no_grad():
        ct = co_true.detach().cpu().numpy().astype(np.float64)
    cm = co_meas_tr.detach().cpu().numpy().astype(np.float64)
    cs = co_sig_tr.detach().cpu().numpy().astype(np.float64)
    co_ratio = np.where(cm > 1e-9, ct / np.maximum(cm, 1e-9), np.nan)
    co_pull_all = (ct - cm) / cs
    log('CO_true/CO_meas (train): med=%.3f p16=%.3f p84=%.3f | (CO_true-CO_meas)/sigma: med=%.2f std=%.2f'
        % (np.nanmedian(co_ratio), np.nanpercentile(co_ratio, 16),
           np.nanpercentile(co_ratio, 84), np.median(co_pull_all), np.std(co_pull_all)))
    # 正性修正统计：负的 CO 观测不物理，被拉回 ≥0 是物理改进
    n_neg_meas = int((cm < 0).sum())
    n_at_zero = int((ct <= 1e-9).sum())
    log('CO positivity: CO_meas<0 = %d px (%.1f%%) -> 修正到 0 的 = %d px (%.1f%% of all)'
        % (n_neg_meas, 100 * n_neg_meas / len(cm), n_at_zero, 100 * n_at_zero / len(ct)))
    n_neg_after = int((ct < 0).sum())
    log('CO negativity removed: %d px 未修正的负值（应为 0）' % n_neg_after)

    # ============ 测试（测量 CO） ============
    # 注意：交替训练结束时内存里的模型是最后一轮（往往已过拟合），所以要跑最终评估
    # 前必须先回滚 —— 上面已重新加载 val 最优的【模型 + CO_true 配对】，否则指标失真。
    log('final eval uses%s' % final_note)
    log('MLP ||W|| (final) = %.3f | per-layer: %s'
        % (model_norm(model),
           ', '.join('%s=%.2f' % (k.replace('.weight', ''), v)
                     for k, v in layer_norms(model).items() if k.endswith('.weight'))))

    # ============ Fisher / 后验误差：qPAH 对 CO 的约束强度 ============
    # g = d(qPAH)/d(CO)（在最优模型、c_true 处求导）；一维高斯后验：
    #   I = g²/σ_eff² + λ/σ_CO² ,  σ_post = 1/√I
    #   σ_post/σ_CO = "qPAH 把 CO 误差压缩到原来的几分之几"（<1 即有额外约束）
    Xg = Xtr.clone().detach()
    Xg[:, 3] = (co_true.detach() - mean[0][3]) / std[0][3]
    Xg.requires_grad_(True)
    with torch.enable_grad():
        pred_g = model(Xg)
        grad_norm = torch.autograd.grad(pred_g.sum(), Xg)[0][:, 3]
    g_co = (grad_norm / std[0][3]).detach().cpu().numpy().astype(np.float64)
    se_np = se_tr.detach().cpu().numpy().astype(np.float64)
    fisher = g_co ** 2 / se_np ** 2 + args.prior_lambda / cs ** 2
    sigma_post = 1.0 / np.sqrt(fisher)
    sig_ratio = sigma_post / cs
    log('CO constraint by qPAH: sigma_post/sigma_CO med=%.3f p16=%.3f p84=%.3f | '
        '|d qPAH/d CO| med=%.3e'
        % (np.median(sig_ratio), np.percentile(sig_ratio, 16),
           np.percentile(sig_ratio, 84), np.median(np.abs(g_co))))

    # ============ CO 响应曲线诊断：碳竞争（先升后降）的强度 ============
    # 固定其它特征为中位，扫 CO，量出峰值位置与拐点之后的下降幅度。
    # 物理依据：CO 与 PAH 争夺碳原子 -> qPAH 随 CO 先升后降。
    co_lo, co_hi = np.percentile(co_meas, [1, 99])
    co_scan = np.linspace(co_lo, co_hi, 121)
    xmed = np.median(Xn, axis=0)          # 标准化空间的中位（≈0）
    Xs = np.tile(xmed, (len(co_scan), 1))
    Xs[:, 3] = (co_scan - mean[0][3]) / std[0][3]
    with torch.no_grad():
        ps = model(torch.tensor(Xs, dtype=torch.float32).to(device)).cpu().numpy()
    ipk = int(np.argmax(ps))
    co_peak, co_peak_val = float(co_scan[ipk]), float(ps[ipk])
    co_end_val = float(ps[-1])
    co_turnover = ((co_peak_val - co_end_val) / co_peak_val) if co_peak_val != 0 else 0.0
    log('CO response (others@median): peak CO=%.2f (qPAH=%.3f) -> CO=%.2f (qPAH=%.3f) | '
        'turnover=%.1f%%'
        % (co_peak, co_peak_val, co_scan[-1], co_end_val, 100 * co_turnover))
    np.savez(os.path.join(outdir, 'co_true_train.npz'),
             ra=ds['ra'][train_idx], dec=ds['dec'][train_idx],
             co_true=ct, co_meas=cm, co_sigma=cs, co_ratio=co_ratio,
             co_pull=co_pull_all, qpah=ytr.detach().cpu().numpy(),
             qpah_err=sig_eval[train_idx], co_det=det[train_idx],
             g_co=g_co, sigma_post=sigma_post, sigma_ratio=sig_ratio,
             sigma_eff=se_np,
             alt_rounds=n_rounds_eff,
             no_co_update=args.no_co_update, clip_co_neg=args.clip_co_neg,
             co_true_s2a=(np.asarray(s2a_hist, dtype=np.float32) if s2a_hist
                          else np.zeros((0, len(train_idx)), dtype=np.float32)),
             co_true_s2b=(np.asarray(s2b_hist, dtype=np.float32) if s2b_hist
                          else np.zeros((0, len(train_idx)), dtype=np.float32)))
    model.eval()
    with torch.no_grad():
        pt = model(build_X(co_m, test_idx)).cpu().numpy()
    ytst = y[test_idx]
    stst = sig_eval[test_idx]
    # 注意：不计算 MSE/RMSLE —— qPAH 观测误差是异方差的，等权平方误差没有物理意义。
    # 拟合质量只用误差加权的 χ² 与 pull 分布来评判。
    pull = (pt - ytst) / stst
    chi2_red = float(np.mean(pull ** 2))
    pull_med = float(np.median(pull))
    pull_std = float(np.std(pull))
    pull_p16 = float(np.percentile(pull, 16))
    pull_p84 = float(np.percentile(pull, 84))
    frac_neg = float(np.mean(pt < 0))
    log('spatial test (meas CO%s): chi2_red=%.3f (ideal 1) | '
        'pull med=%.3f std=%.3f [p16=%.3f p84=%.3f]'
        % (', clipped>=0' if args.clip_co_neg else '', chi2_red, pull_med, pull_std,
           pull_p16, pull_p84))
    json.dump(dict(selection=args.selection, tag=tag, h5=h5path, n=ds['n_total'],
                   config_file=cfg_path,
                   config=cfg,
                   model_hidden=list(hidden), model_dropout=dropout,
                   split=dict(n_sectors=n_sectors, train_sector_frac=train_frac,
                              random_state=seed, val=None,
                              note='训练扇区全部像素进迭代；空间测试扇区不参与训练'),
                   n_det=int(det.sum()), n_det_train=int(det[train_idx].sum()),
                   n_det_test=int(det[test_idx].sum()),
                   n_train=len(train_idx),
                   n_test=len(test_idx), snr_thresh=args.snr_thresh,
                   no_co_update=bool(args.no_co_update),
                   clip_co_neg=bool(args.clip_co_neg),
                   co_frac_raw_neg=float(100.0 * (co_meas_raw < 0).mean()),
                   err_weight=not args.no_err_weight, err_floor=args.err_floor,
                   huber_delta=args.huber_delta,
                   stage1_epochs=args.stage1_epochs,
                   alt_rounds=args.alt_rounds,
                   alt_rounds_hist=round_hist,
                   co_only_steps=args.co_only_steps, joint_epochs=args.joint_epochs,
                   prior_lambda=args.prior_lambda,
                   chi2_red=chi2_red, pull_med=pull_med, pull_std=pull_std,
                   pull_p16=pull_p16, pull_p84=pull_p84,
                   co_frac_meas_neg=float(100.0 * n_neg_meas / len(cm)),
                   co_frac_at_zero=float(100.0 * n_at_zero / len(ct)),
                   co_min=float(ct.min()),
                   frac_pred_neg=frac_neg, final_eval_model=final_note.strip(),
                   weight_decay=args.weight_decay,
                   sigma_eff_med=float(np.median(sig_eval)),
                   co_true_ratio_med=float(np.nanmedian(co_ratio)),
                   co_true_ratio_p16=float(np.nanpercentile(co_ratio, 16)),
                   co_true_ratio_p84=float(np.nanpercentile(co_ratio, 84)),
                   co_true_pull_med=float(np.median(co_pull_all)),
                   co_true_pull_std=float(np.std(co_pull_all)),
                   sigma_ratio_med=float(np.median(sig_ratio)),
                   sigma_ratio_p16=float(np.percentile(sig_ratio, 16)),
                   sigma_ratio_p84=float(np.percentile(sig_ratio, 84)),
                   g_co_abs_med=float(np.median(np.abs(g_co))),
                   co_peak=co_peak, co_peak_val=co_peak_val,
                   co_turnover=co_turnover,
                   model_file=os.path.join(outdir, 'model.pth')),
              open(os.path.join(outdir, 'train.json'), 'w'), indent=1)
    log('ALL DONE ->', outdir)


if __name__ == '__main__':
    main()
