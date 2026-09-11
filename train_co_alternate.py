#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_co_alternate.py — CO forward-modelling 三段式训练（Stage-1 / 2a / 2b）。

数据集要求：build_dataset.py 新版产物（data/M31_1d_array_full.h5 或 agb，
含 CO_sigma / CO_det 列）。

流程：
  Stage-1 : 只用 CO/CO_sigma > 3 的探测像素训练 MLP（异方差 Pseudo-Huber：
            σ_eff=sqrt(qpah_err²+floor²)，CO 固定为测量值；--no-err-weight 回退等权）；
  Stage-2a: 放开全部（含非探测）像素，冻结模型，只更新每像素 CO_true
            （全部像素都带高斯先验 (CO_true−CO_meas)²/σ²，CO_true≥0）；
  Stage-2b: 同时放开模型参数与 CO_true，用小学习率联合微调，
            CO 先验项始终保留在 loss 里防 CO 漂移。
验证/测试一律用测量 CO（不更新 CO），用于选模型与最终指标。

用法:
  python train_co_alternate.py --selection full
  python train_co_alternate.py --selection agb --tag coalt_agb

输出: results/<tag>/{model_stage1.pth, model.pth, scale.json, train.json}
"""
import os, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
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
    def __init__(self, input_dim=5, hidden=(32, 32)):
        super().__init__()
        layers, in_dim = [], input_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).flatten()


def phuber(r, delta=10.0):
    return delta ** 2 * (torch.sqrt(1 + (r / delta) ** 2) - 1)


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


def sector_split(ds):
    ang = np.arctan2(ds['dec'] - 41.25, ds['ra'] - 10.75)
    ang = (ang + 2 * np.pi) % (2 * np.pi)
    sec = np.digitize(ang, np.linspace(0, 2 * np.pi, 11)) - 1
    tr = np.random.choice(np.arange(10), size=6, replace=False)
    te = np.setdiff1d(np.arange(10), tr)
    train_full = np.where(np.isin(sec, tr))[0]
    test_idx = np.where(np.isin(sec, te))[0]
    train_idx, val_idx = train_test_split(train_full, test_size=0.2,
                                          random_state=random_state)
    return tr, te, train_idx, val_idx, test_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selection', choices=['full', 'agb'], default='full')
    ap.add_argument('--dataset', default=None)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--snr-thresh', type=float, default=3.0)
    ap.add_argument('--stage1-epochs', type=int, default=300)
    ap.add_argument('--co-only-steps', type=int, default=50,
                    help='Stage-2a 冻结模型时更新 CO_true 的步数')
    ap.add_argument('--co-lr', type=float, default=1e-3)
    ap.add_argument('--joint-epochs', type=int, default=200,
                    help='Stage-2b 联合微调 epoch 数')
    ap.add_argument('--joint-model-lr', type=float, default=1e-4)
    ap.add_argument('--joint-co-lr', type=float, default=1e-3)
    ap.add_argument('--prior-lambda', type=float, default=1.0)
    ap.add_argument('--batch-size', type=int, default=4096)
    ap.add_argument('--err-floor', type=float, default=0.2,
                    help='qpah_err 异方差 loss 的 floor（σ_eff=sqrt(err²+floor²)）')
    ap.add_argument('--huber-delta', type=float, default=10.0,
                    help='Pseudo-Huber δ（作用于残差/σ_eff 之后）')
    ap.add_argument('--no-err-weight', action='store_true',
                    help='回退等权 Pseudo-Huber（不除以 qpah_err）')
    args = ap.parse_args()
    tag = args.tag or ('coalt_' + args.selection)
    outdir = os.path.join(RESULTS, tag)
    os.makedirs(outdir, exist_ok=True)

    print('device:', device, flush=True)
    set_seed(random_state)
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
    sig_eff = (np.sqrt(qpah_err ** 2 + args.err_floor ** 2)
               if not args.no_err_weight else np.ones_like(qpah_err))
    co_meas = ds['CO'].astype(np.float32)
    co_sig = np.maximum(ds['CO_sigma'].astype(np.float32), 1e-6)
    det = (co_meas / co_sig) > args.snr_thresh
    log('detection(CO/σ>%.0f): %d / %d (%.1f%%)'
        % (args.snr_thresh, int(det.sum()), len(y), 100 * det.mean()))
    log('qPAH loss: %s (err-floor=%.2f, huber-delta=%.2f)'
        % ('heteroscedastic Pseudo-Huber' if not args.no_err_weight
           else 'equal-weight Pseudo-Huber', args.err_floor, args.huber_delta))

    json.dump({'features': FEATURES, 'mean': mean.flatten().tolist(),
               'std': std.flatten().tolist(), 'transform': None},
              open(os.path.join(outdir, 'scale.json'), 'w'), indent=1)

    tr_s, te_s, train_idx, val_idx, test_idx = sector_split(ds)
    log('train sectors %s | test sectors %s' % (tr_s, te_s))
    log('train/val/test:', len(train_idx), len(val_idx), len(test_idx))

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

    def val_loss(model):
        model.eval()
        with torch.no_grad():
            r = (model(build_X(co_m, val_idx)) - yt[val_idx]) / se_t[val_idx]
            return torch.mean(phuber(r, args.huber_delta)).item()

    # ============ Stage-1: CO>3σ 子集训练 ============
    log('Stage-1: train on CO/σ>%.0f pixels (n=%d) ...' % (args.snr_thresh, det.sum()))
    model = MLPRegressor().to(device)
    opt1 = optim.Adam(model.parameters(), lr=0.01)
    sch1 = optim.lr_scheduler.StepLR(opt1, step_size=100, gamma=0.8)
    idx1 = np.where(det)[0]
    yy1 = y[idx1]
    bi = np.digitize(yy1, np.linspace(0, 10, 11)) - 1
    cnt = np.bincount(bi, minlength=10)
    sw = 1.0 / (cnt[bi] + 1)
    cap = np.percentile(sw, 95) * 2
    sw = np.clip(sw, None, cap)
    sw = sw / sw.sum() * len(sw)
    sampler1 = WeightedRandomSampler(torch.tensor(sw, dtype=torch.float64),
                                     num_samples=len(idx1), replacement=True)
    loader1 = DataLoader(TensorDataset(Xt[idx1], yt[idx1], se_t[idx1]),
                         batch_size=args.batch_size, sampler=sampler1)
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
        if (ep + 1) % 100 == 0:
            log('  S1 ep %4d tr=%.4f' % (ep + 1, tot / len(idx1)))
    torch.save(model.state_dict(), os.path.join(outdir, 'model_stage1.pth'))
    v1 = val_loss(model)
    log('Stage-1 val (meas CO): %.4f' % v1)

    # ============ Stage-2a: 冻结模型，只更新 CO_true（全像素，带先验） ============
    # CO_true 只对训练像素优化（val/test 一律用测量 CO，避免泄漏）
    co_true = nn.Parameter(co_m[train_idx].clone())
    Xtr = Xt[train_idx]
    ytr = yt[train_idx]
    co_meas_tr = co_m[train_idx]
    co_sig_tr = co_s[train_idx]
    se_tr = se_t[train_idx]
    opt_co = optim.Adam([co_true], lr=args.co_lr)

    def co_loss_full():
        Xb = Xtr.clone()
        Xb[:, 3] = (co_true - mean[0][3]) / std[0][3]
        q = torch.mean(phuber((model(Xb) - ytr) / se_tr, args.huber_delta))
        pr = torch.mean(((co_true - co_meas_tr) / co_sig_tr) ** 2)
        return q + 0.5 * args.prior_lambda * pr

    log('Stage-2a: freeze model, update CO_true (all pixels) %d steps...'
        % args.co_only_steps)
    model.eval()
    for k in range(args.co_only_steps):
        opt_co.zero_grad()
        loss = co_loss_full()
        loss.backward()
        opt_co.step()
        with torch.no_grad():
            co_true.data.clamp_(min=0.0)
        if (k + 1) % 10 == 0:
            log('  S2a step %3d loss=%.4f  CO_true med=%.3f'
                % (k + 1, loss.item(), co_true.detach().median().item()))

    # ============ Stage-2b: 联合微调（模型 + CO_true 小步幅） ============
    log('Stage-2b: joint fine-tune (%d epochs, model_lr=%.0e co_lr=%.0e)...'
        % (args.joint_epochs, args.joint_model_lr, args.joint_co_lr))
    params = [{'params': model.parameters(), 'lr': args.joint_model_lr},
              {'params': [co_true], 'lr': args.joint_co_lr}]
    opt_j = optim.Adam(params)
    best = float('inf')
    best_ep = -1
    n_tr = len(train_idx)
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
                co_true.data.clamp_(min=0.0)
            tot += q.item() * len(ib)
        if (ep + 1) % 20 == 0 or ep == args.joint_epochs - 1:
            vl = val_loss(model)
            log('  S2b ep %3d tr=%.4f val(measCO)=%.4f best=%.4f'
                % (ep + 1, tot / n_tr, vl, best))
            if vl < best:
                best = vl
                best_ep = ep + 1
                torch.save(model.state_dict(), os.path.join(outdir, 'model.pth'))

    # ============ 测试（测量 CO） ============
    model.eval()
    with torch.no_grad():
        pt = model(build_X(co_m, test_idx)).cpu().numpy()
    ytst = y[test_idx]
    stst = sig_eff[test_idx]
    mse = float(np.mean((pt - ytst) ** 2))
    rmsle = float(np.sqrt(np.mean((np.log(np.clip(pt, 1e-12, None)) -
                                   np.log(ytst)) ** 2)))
    pull = (pt - ytst) / stst
    chi2_red = float(np.mean(pull ** 2))
    pull_med = float(np.median(pull))
    pull_std = float(np.std(pull))
    log('Stage-2 spatial test (meas CO): MSE=%.6f RMSLE=%.6f | chi2_red=%.3f pull med/std=%.3f/%.3f'
        % (mse, rmsle, chi2_red, pull_med, pull_std))
    json.dump(dict(selection=args.selection, tag=tag, h5=h5path, n=ds['n_total'],
                   n_det=int(det.sum()), n_train=len(train_idx), n_val=len(val_idx),
                   n_test=len(test_idx), snr_thresh=args.snr_thresh,
                   err_weight=not args.no_err_weight, err_floor=args.err_floor,
                   huber_delta=args.huber_delta,
                   stage1_val=float(v1), stage2_best_val=float(best), best_epoch=best_ep,
                   co_only_steps=args.co_only_steps, joint_epochs=args.joint_epochs,
                   prior_lambda=args.prior_lambda, mse=mse, rmsle=rmsle,
                   chi2_red=chi2_red, pull_med=pull_med, pull_std=pull_std,
                   model_file=os.path.join(outdir, 'model.pth')),
              open(os.path.join(outdir, 'train.json'), 'w'), indent=1)
    log('ALL DONE ->', outdir)


if __name__ == '__main__':
    main()
