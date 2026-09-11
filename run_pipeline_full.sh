#!/usr/bin/env bash
# run_pipeline_full.sh — 一次性从 0 跑到尾（full 数据集，只跑 CO 交替优化 forward model）
# 用法: 上传整个 qpah_pipeline/ 到 <repo>（含 data/ sanity_cache/ AGB/ 的上一级），然后:
#   cd <repo>
#   PYTHON=~/miniconda3/envs/torch/bin/python bash qpah_pipeline/run_pipeline_full.sh
# 顺序:
#   1 make_co_maps            (cube -> I_CO + sigma_I)
#   2 align_maps              (exact block 到 qPAH 网格; 重建 CO/CO_sigma/CO_det 缓存)
#   3 build_dataset full
#   4 train_co_alternate full (Stage-1 -> 2a -> 2b; 输出 results/coalt_full/)
#   5 predict_grid coalt_full (读 results/coalt_full/model.pth 预测全网格)
#   6 interpret coalt_full    (解释图，基于交替优化模型)
#   7 plot_galaxy_map coalt_full (全样本 RA-Dec obs/pred/resid 三面板)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/torch/bin/python}"
mkdir -p logs tmp results

echo "repo root : $ROOT"
echo "python    : $PYTHON"
echo "开始时间  : $(date '+%F %T')"

# 清理旧数据集（必须重建，因为 CO 产物已改变）
rm -f data/M31_1d_array_full.h5 data/M31_1d_array_agb.h5

step(){ echo; echo "===== [$1/7] $2 ====="; }

step 1 "make_co_maps.py (cube -> I_CO + sigma_I)"
"$PYTHON" "$HERE/make_co_maps.py" 2>&1 | tee logs/make_co_maps.log

step 2 "align_maps.py (对齐到 qPAH; 重建 CO/CO_sigma/CO_det 缓存)"
"$PYTHON" "$HERE/align_maps.py" 2>&1 | tee logs/align_maps.log

step 3 "build_dataset.py --selection full"
"$PYTHON" "$HERE/build_dataset.py" --selection full 2>&1 | tee logs/build_dataset_full.log

step 4 "train_co_alternate.py --selection full (CO forward model)"
"$PYTHON" "$HERE/train_co_alternate.py" --selection full 2>&1 | tee logs/train_co_alternate_full.log

step 5 "predict_grid.py --tag coalt_full"
"$PYTHON" "$HERE/predict_grid.py" --tag coalt_full 2>&1 | tee logs/predict_grid_coalt_full.log

step 6 "interpret.py --selection full --tag coalt_full"
"$PYTHON" "$HERE/interpret.py" --selection full --tag coalt_full 2>&1 | tee logs/interpret_coalt_full.log

step 7 "plot_galaxy_map.py --tag coalt_full (RA-Dec obs/pred/resid 三面板)"
"$PYTHON" "$HERE/plot_galaxy_map.py" --tag coalt_full 2>&1 | tee logs/plot_galaxy_map_coalt_full.log

echo
echo "全部完成: $(date '+%F %T')"
echo "关键产物:"
echo "  data/M31_1d_array_full.h5 (含 CO_sigma/CO_det)"
echo "  results/coalt_full/{model_stage1.pth,model.pth,scale.json,train.json,fig_*(含 fig_galaxy_map)}"
echo "  data/qpah_pred_map_coalt_full.fits (+ resid)"
echo "  logs/*.log"
