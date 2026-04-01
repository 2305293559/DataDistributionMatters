#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_batch_and_train.py

功能：
- 调用 10batch_sampling_processor.py 的 run_batch_sampling()
  进行采样（所有采样方法、所有数据量，完全使用脚本内部的设置）。
- 执行 50 次批量采样，每一次生成的所有 CSV 放入：
      auto_outputs/sampling_data/run_编号/
- 然后将每一个 CSV 都送入 new_model_XGBoost_30fold.py 的 process_single_file()
  并将模型结果输出到：
      auto_outputs/model_results/run_编号/

"""

import os
import traceback
from pathlib import Path
from importlib.machinery import SourceFileLoader

# =====================================================
# 配置
# =====================================================

# 你上传的两个主脚本文件路径（无需修改）
BATCH_SAMPLER_PATH = "10batch_sampling_processor_f3_Michalewicz.py"
XGB_MODULE_PATH = "new_model_XGBoost_30fold.py"

# 输出根目录
OUTPUT_ROOT = "auto_outputs_50_f3_Michalewicz_2"
SAMPLING_PARENT = os.path.join(OUTPUT_ROOT, "sampling_data")
MODEL_PARENT = os.path.join(OUTPUT_ROOT, "model_results")

# 要求：运行 50 次
N_RUNS = 50


# =====================================================
# 工具函数
# =====================================================

def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)

def load_module(module_name: str, path: str):
    loader = SourceFileLoader(module_name, path)
    return loader.load_module()


# =====================================================
# 主流程
# =====================================================

def main(n_runs=N_RUNS):

    safe_mkdir(SAMPLING_PARENT)
    safe_mkdir(MODEL_PARENT)

    print("[INFO] 加载采样模块...")
    batch_mod = load_module("batch_sampler_module", BATCH_SAMPLER_PATH)

    print("[INFO] 加载模型训练模块...")
    xgb_mod = load_module("xgb_module", XGB_MODULE_PATH)

    for run_idx in range(1, n_runs + 1):
        run_tag = f"run_{run_idx}"

        print("\n" + "="*80)
        print(f"[ RUN {run_idx}/{n_runs} ]   当前运行: {run_tag}")
        print("="*80)

        sampling_dir = os.path.join(SAMPLING_PARENT, run_tag)
        model_dir = os.path.join(MODEL_PARENT, run_tag)

        safe_mkdir(sampling_dir)
        safe_mkdir(model_dir)

        try:
            # ------------------------------------------------------------
            # 1. 采样 —— 覆盖模块内部默认输出目录为我们自己的目录
            # ------------------------------------------------------------
            if hasattr(batch_mod, "OUTPUT_DIR"):
                batch_mod.OUTPUT_DIR = sampling_dir
                print(f"[INFO] 设置采样输出目录: {sampling_dir}")
            else:
                print("[WARN] 未找到 OUTPUT_DIR，可能使用默认目录。")

            print("[INFO] 调用 run_batch_sampling()...")
            results = batch_mod.run_batch_sampling()

            # 从返回结果中提取 CSV 文件路径
            csv_list = []
            if isinstance(results, dict):
                for _, method_results in results.items():
                    if isinstance(method_results, dict):
                        for _, detail in method_results.items():
                            fp = detail.get("filepath") if isinstance(detail, dict) else None
                            if fp:
                                csv_list.append(fp)

            # 如果没有找到，直接扫描采样目录
            if not csv_list:
                csv_list = [str(p) for p in Path(sampling_dir).glob("*.csv")]

            print(f"[INFO] 本轮采样得到 {len(csv_list)} 个 CSV，将逐一训练。")

            # ------------------------------------------------------------
            # 2. 模型训练 —— 对每个 CSV 调用 process_single_file
            # ------------------------------------------------------------
            ok_count = 0
            for csv_path in csv_list:
                print("\n----------------------------------------")
                print(f"[TRAIN] 正在处理: {csv_path}")
                try:
                    ok = xgb_mod.process_single_file(
                        csv_path=csv_path,
                        output_base_dir=model_dir,
                        random_state=None  # 随机
                    )
                    if ok:
                        ok_count += 1
                        print("[TRAIN] ✔ 成功")
                    else:
                        print("[TRAIN] ✖ 失败（process_single_file 返回 False）")

                except Exception as e:
                    print(f"[ERROR] 模型训练异常: {e}")
                    traceback.print_exc()

            print("\n" + "-"*60)
            print(f"[RUN {run_idx}] 完成 {ok_count}/{len(csv_list)} 个模型训练")
            print(f"[RUN {run_idx}] 采样保存于：{sampling_dir}")
            print(f"[RUN {run_idx}] 模型结果位于：{model_dir}")
            print("-"*60)

        except Exception as e:
            print(f"[ERROR] run_{run_idx} 执行失败: {e}")
            traceback.print_exc()

    print("\n========== 全部运行完成！ ==========")
    print(f"采样数据目录：{os.path.abspath(SAMPLING_PARENT)}")
    print(f"模型结果目录：{os.path.abspath(MODEL_PARENT)}")


if __name__ == "__main__":
    main()

