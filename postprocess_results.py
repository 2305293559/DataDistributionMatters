# -*- coding: utf-8 -*-
"""
postprocess_results.py

功能
----
1. 将所有数值列中 < 0 的值裁剪为 0
2. 按 sample_size 分组，对多个 run 取平均
3. 输出处理后的汇总 CSV

输入
----
all_runs_evaluation_summary.csv

输出
----
all_runs_evaluation_summary_clipped.csv
all_runs_evaluation_summary_mean_by_sample_size.csv
"""

import pandas as pd
import numpy as np

# ============================================================
# 【用户参数区】
# ============================================================

INPUT_FILE = "all_runs_evaluation_summary.csv"

# 中间文件（小于 0 → 0）
CLIPPED_OUTPUT_FILE = "all_runs_evaluation_summary_clipped.csv"

# 最终文件（按 sample_size 求均值）
MEAN_OUTPUT_FILE = "all_runs_evaluation_summary_mean_by_sample_size.csv"

# ============================================================
# 主逻辑
# ============================================================

def main():
    # ---------- 1. 读取数据 ----------
    df = pd.read_csv(INPUT_FILE)

    print(f"读取数据: {df.shape[0]} 行, {df.shape[1]} 列")

    # ---------- 2. 找出数值列 ----------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # 不对 run_id / sample_size 做裁剪
    numeric_cols = [
        c for c in numeric_cols
        if c not in ["run_id", "sample_size"]
    ]

    print("数值列（将执行 <0 → 0）：")
    for c in numeric_cols:
        print(f"  - {c}")

    # ---------- 3. 小于 0 的数裁剪为 0 ----------
    df_clipped = df.copy()
    df_clipped[numeric_cols] = df_clipped[numeric_cols].clip(lower=0.0)

    df_clipped.to_csv(CLIPPED_OUTPUT_FILE, index=False)
    print(f"\n已保存裁剪后的文件: {CLIPPED_OUTPUT_FILE}")

    # ---------- 4. 按 sample_size 求平均 ----------
    df_mean = (
        df_clipped
        .groupby("sample_size")[numeric_cols]
        .mean()
        .reset_index()
        .sort_values("sample_size")
    )

    df_mean.to_csv(MEAN_OUTPUT_FILE, index=False)

    print(f"已保存按 sample_size 求均值后的文件: {MEAN_OUTPUT_FILE}")
    print("\n示例结果：")
    print(df_mean.head())


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":
    main()
