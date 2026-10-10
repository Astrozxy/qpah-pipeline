# notebooks — 面向读者的公开复现文件

## `qPAH_model_M31.ipynb`

从公开数据出发，完整复现 **M31 的 qPAH 经验模型**，并证明 **AGB 恒星密度对模型没有贡献**。

### 内容

1. **数据来源**（含自动下载）
2. **完整筛选链**（每一步的通过数都打印出来）
3. **特征构造**：观测量取 log10、CO 未探测置 0、去投影盘面坐标
4. **训练/测试划分**（10 个 36° 方位扇区，6 训练 / 4 留出）
5. **训练两个模型**：有 AGB / 没有 AGB（**锁在同一批像素上**）
6. **★ χ² 分布对比图** —— 三条面板里两条曲线几乎完全重合，直观证明 AGB 无用
7. 4 面板图（观测 / 预测 / 相对误差 / 绝对误差）
8. 径向诊断
9. 结论与两个必须在论文里声明的注意事项（qPAH 下界截断、24.9″ 波束过采样）

### 运行需要的数据（放在 `notebooks/data/`）

| 文件 | 来源 | 大小 |
|---|---|---|
| `M31_1d_array_nodust.h5` | 本 pipeline 的逐像素数据集 | 11 MB |
| `M31_S350_110_SSS_110_Model_All_qpah.fits` | [Draine M31 dust page](https://www.astro.princeton.edu/~draine/m31dust/m31dust.html)（**notebook 会自动下载**） | 6 MB |
| `M31_S350_110_SSS_110_Model_All_SurfBr_Ldust.fits` | 同上（**自动下载**） | 6 MB |
| `AGB_count_map.npy` | 由 `make_agb_map.py` 从 PHAT / PHAT-S 星表生成 | 12 MB |

> 注意：两张 Draine FITS 与 qPAH 图**在同一个网格上**（SPIRE-350 分辨率 24.9″、10″ 像素），
> 因此**不需要 reproject**，可以直接按像素取值。

### 运行

```bash
pip install numpy h5py astropy matplotlib jupyter
jupyter nbconvert --to notebook --execute qPAH_model_M31.ipynb
# 或者直接 jupyter lab qPAH_model_M31.ipynb
```

### 已执行版本

本目录下的 `qPAH_model_M31.ipynb` **已带输出**（含所有图和数字），可直接阅读。

### 主要结果（n = 74,764，AGB 覆盖区，锁样本）

| | 没有 AGB | 有 AGB |
|---|---|---|
| χ²_red（全部） | 1.3593 | **1.3563** |
| χ²_red（留出） | 1.1387 | **1.1337** |
| 中位 \|pull\| | 0.5998 | 0.5995 |
| \|pull\|>3 占比 | 2.23% | 2.24% |

**AGB 标准化系数 = +0.0273**（H1 是 +0.47、Ldust 是 +0.49）；
**Δχ²_red = 0.0030，只有统计误差（0.0137）的 0.22σ** ⇒ **AGB 没有信息量**。
