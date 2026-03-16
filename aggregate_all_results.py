# -*- coding: utf-8 -*-
"""
aggregate_all_results.py

功能
----
汇总 results/ 目录下：
    run_xx/xgb_xxx_samples/**/evaluation_summary.csv

特点
----
- 自动识别 run 编号
- 自动识别样本规模（20 / 50 / 100 / ...）
- 不依赖 evaluation_summary.csv 的具体层级
- 自动合并为一个总 CSV
"""

import re
from pathlib import Path
import pandas as pd


# ============================================================
# 【用户只需要改这里】
# ============================================================

RESULTS_DIR = "results"
OUTPUT_FILE = "all_runs_evaluation_summary.csv"

# ============================================================
# 内部逻辑
# ============================================================

def extract_run_id(path: Path) -> int:
    """
    从路径中提取 run_id，例如 run_01 -> 1
    """
    for part in path.parts:
        if part.startswith("run_"):
            return int(part.replace("run_", ""))
    return -1


def extract_sample_size(path: Path) -> int:
    """
    从路径中提取样本规模，例如 xgb_200_samples -> 200
    """
    for part in path.parts:
        m = re.match(r"xgb_(\d+)_samples", part)
        if m:
            return int(m.group(1))
    return -1


def aggregate_results():
    results_dir = Path(RESULTS_DIR)
    all_records = []

    # 递归查找所有 evaluation_summary.csv
    csv_files = results_dir.rglob("evaluation_summary.csv")

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)

            run_id = extract_run_id(csv_path)
            sample_size = extract_sample_size(csv_path)

            if run_id == -1 or sample_size == -1:
                print(f"⚠️ 跳过无法识别路径: {csv_path}")
                continue

            # evaluation_summary.csv 通常是一行（模型维度）
            for _, row in df.iterrows():
                record = row.to_dict()
                record["run_id"] = run_id
                record["sample_size"] = sample_size
                record["source_file"] = str(csv_path)
                all_records.append(record)

        except Exception as e:
            print(f"❌ 读取失败: {csv_path} | {e}")

    if not all_records:
        raise RuntimeError("❌ 未找到任何 evaluation_summary.csv")

    final_df = pd.DataFrame(all_records)

    # 排序更清晰
    final_df = final_df.sort_values(
        by=["sample_size", "run_id"]
    ).reset_index(drop=True)

    final_df.to_csv(OUTPUT_FILE, index=False)

    print("\n✅ 汇总完成")
    print(f"📄 输出文件: {OUTPUT_FILE}")
    print(f"📊 总记录数: {len(final_df)}")


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    aggregate_results()
