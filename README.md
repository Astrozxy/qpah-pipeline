# qpah_pipeline

M31 qPAH(%) 逐像素预测的公开代码。
用 reproject_exact 把 CO/HI/HI_EW/SFR/dust 对齐到 qPAH 10″ 网格，
构建训练集，并用 **CO 三段式 forward model（交替优化）** 训练 MLP，
输出模型、指标、全网格预测图与解释图。

## 数据依赖（运行前需就位，位于本仓库上一级目录）

| 路径 | 内容 |
|---|---|
| `data/` | 各 FITS：qPAH/qpah_unc、SFR、HI mom0/EW、CO native 立方体、dust(NH2_sm24)（见 align_maps.py） |
| `sanity_cache/` | 对齐缓存（align_maps.py 生成；已存在则跳过） |
| `AGB/` | 仅 `--selection agb` 需要：PHAT/PHAT-S 星表 |

## 用法

```bash
# 整个 qpah_pipeline/ 放到含 data/ 的目录下（即 <repo>/qpah_pipeline）
cd <repo>
PYTHON=~/miniconda3/envs/torch/bin/python bash qpah_pipeline/run_pipeline_full.sh
```

分步：

```bash
python qpah_pipeline/make_co_maps.py
python qpah_pipeline/align_maps.py
python qpah_pipeline/build_dataset.py --selection full           # data/M31_1d_array_full.h5
python qpah_pipeline/train_co_alternate.py --selection full      # results/coalt_full/{model.pth,train.json,...}
python qpah_pipeline/predict_grid.py --tag coalt_full
python qpah_pipeline/interpret.py --selection full --tag coalt_full
```

- `--selection full`：全盘（主数据集）：coverage 齐全像素，无 AGB 阈值。
- `--selection agb`：AGB≥3 且剔除 M32（构建较慢，需 AGB 星表；交替训练 tag 默认 `coalt_agb`）。
- 交替训练三段式（`train_co_alternate.py`）：
  1. **Stage-1** 只用 CO/σ>3 探测像素训练 MLP（等权 Pseudo-Huber δ=10，CO=测量值）；
  2. **Stage-2a** 冻结模型，只更新全部像素的 CO_true（高斯先验 + CO_true≥0）；
  3. **Stage-2b** 联合小步微调模型参数与 CO_true（CO 先验始终保留）。
- 验证/测试一律用测量 CO，用于选模型与最终指标。

## 输出

```
results/coalt_full/
├── model_stage1.pth / model.pth / scale.json / train.json
├── co_true_train.npz                        (CO_true 诊断 + Fisher/sigma_post)
├── fig_galaxy_map.png/.pdf                  (obs / pred / chi 三面板)
├── fig_co_update_map.png/.pdf               (CO 更新前/后 + dCO + 显著性)
├── fig_pred_map_<tag>.png                   (全网格预测)
├── fig_chi_map_<tag>.png                    (全网格 chi)
└── fig_interpret_*.png                      (interpret.py)
data/qpah_pred_map_<tag>.fits / qpah_pred_resid_map_<tag>.fits / qpah_chi_map_<tag>.fits
```

**评估口径（重要）**：qPAH 观测误差是**逐像素异方差**的，因此本流程**不计算 MSE/RMSLE**。
拟合质量用 `chi2_red = mean(pull^2)`（理想值 1）与 pull 分布（理想 med 0 / std 1）判断，
其中 `pull = (pred - obs)/sigma_eff`，`sigma_eff = sqrt(qpah_err^2 + err_floor^2)`。
同理，所有残差图都画成相对误差 `chi`，而非绝对残差。

**CO 交替更新**：Stage-1 只用 CO/σ>3 像素训练（假设观测=真值）→ Stage-2a 冻结模型、
以 CO 观测为先验反推**全部训练扇区像素**的 `c_true` → Stage-2b 用 `c_true` 训练并与模型联合微调。
测试扇区不做反推（避免用 qPAH 标签泄漏），预测时用观测值输入。
`co_true_train.npz` 给出每像素的 `c_true`、`sigma_post/sigma_CO`（qPAH 对 CO 的约束强度，
=1 表示无额外约束）与 `g = d(qPAH)/d(CO)`。

## 附录分析（不进主流程）

`analysis/`：旧"中心取点" vs 新 exact-block 的对比（全网格 + 数据集级），
用于论文稳健性/方法论证，独立运行：

```bash
python qpah_pipeline/analysis/compare_map_level.py
python qpah_pipeline/analysis/compare_datasets.py   # 需要旧 data/M31_1d_array.h5
```

## 复现 notebook 对应关系

| 本仓库 | M31_power_law.ipynb / M31_plot.ipynb |
|---|---|
| align_maps.py | 统一投影部分（面积加权平均代替中心取点） |
| build_dataset.py / train_co_alternate.py | 数据筛选 + MLP 训练（扇区 60/40、均衡采样、Pseudo-Huber；CO forward model 交替优化） |
| interpret.py | 固定其它变量扫 CO/HI 的解释剖面 |
| analysis/ | 新旧取样对比（论文稳健性检验） |

## requirements

astropy, reproject, numpy, scipy, h5py, torch, scikit-learn, matplotlib
