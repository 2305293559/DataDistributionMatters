# -*- coding: utf-8 -*-
"""
run_full_pipeline.py

功能：
1. 生成外部测试集
2. 按多个样本规模生成训练集（可配置采样策略）
3. 批量调用 XGBoost + external test 进行训练评估
4. 自动重复运行 N_RUNS 次（用于统计稳定性）
"""

import os
from pathlib import Path

# ============================================================
# 【用户参数区（你以后只需要改这里）】
# ============================================================

# 原始大规模 Rastrigin 数据
RAW_DATA_FILE = "rastrigin_random_1000000samples.csv"

# 样本规模列表
SAMPLE_SIZES = [20, 50, 80, 100, 200, 300, 500, 1000]

# 外部测试集参数
EXTERNAL_TEST_N = 1000
EXTERNAL_TEST_DIR = "data/external_test"
EXTERNAL_TEST_FILE = "external_test.csv"

# 采样策略
SAMPLING_CONFIG = {"high": 0.00, "low": 1.00, "random": 0.00}

# 中间数据 & 输出目录
SELECTED_BASE_DIR = "data/selected_samples"
RESULT_BASE_DIR = "results"

# ================= 多次运行控制 =================
N_RUNS = 10
BASE_RANDOM_SEED = 2025
# ============================================================


# ============================================================
# 1. 生成外部测试集（每个 run 都生成一个）
# ============================================================

def generate_external_test(run_id, random_seed):
    from gen_random_testset import generate_random_samples

    print(f"\n=== [Run {run_id}] Step 1: 生成外部测试集 ===")

    DOMAIN = {f"x{i}": (-5.12, 5.12) for i in range(1, 11)}

    run_dir = os.path.join(EXTERNAL_TEST_DIR, f"run_{run_id:02d}")
    os.makedirs(run_dir, exist_ok=True)

    generate_random_samples(
        n_samples=EXTERNAL_TEST_N,
        domain=DOMAIN,
        output_dir=run_dir,
        output_filename=EXTERNAL_TEST_FILE
    )

    return os.path.join(run_dir, EXTERNAL_TEST_FILE)


# ============================================================
# 2. 批量生成不同规模的训练集
# ============================================================

def generate_selected_datasets(run_id, random_seed):
    print(f"\n=== [Run {run_id}] Step 2: 生成训练集 ===")

    import numpy as np
    import pandas as pd
    from select_rastrigin_samples1 import find_objective_column

    df_all = pd.read_csv(RAW_DATA_FILE)
    obj_col = find_objective_column(df_all)

    # -------- 分位数从 25% / 75% 改为 15% / 85% --------
    q15 = df_all[obj_col].quantile(0.15)
    q85 = df_all[obj_col].quantile(0.85)

    df_low = df_all[df_all[obj_col] <= q15]     # 最小 15%
    df_high = df_all[df_all[obj_col] >= q85]    # 最大 15%
    # --------------------------------------------------

    rng = np.random.RandomState(random_seed)

    run_base_dir = os.path.join(SELECTED_BASE_DIR, f"run_{run_id:02d}")
    os.makedirs(run_base_dir, exist_ok=True)

    for n in SAMPLE_SIZES:
        print(f"  -> 生成 {n} 样本")

        n_high = int(n * SAMPLING_CONFIG["high"])
        n_low = int(n * SAMPLING_CONFIG["low"])
        n_random = n - n_high - n_low

        df_high_sel = df_high.sample(min(n_high, len(df_high)), random_state=rng)
        df_low_sel = df_low.sample(min(n_low, len(df_low)), random_state=rng)

        used_idx = set(df_high_sel.index) | set(df_low_sel.index)
        df_remain = df_all.drop(index=used_idx)

        df_random_sel = df_remain.sample(n_random, random_state=rng)

        df_final = (
            pd.concat([df_high_sel, df_low_sel, df_random_sel])
            .reset_index(drop=True)
        )

        out_dir = os.path.join(run_base_dir, f"{n}_samples")
        os.makedirs(out_dir, exist_ok=True)

        out_file = os.path.join(out_dir, f"train_{n}.csv")
        df_final.to_csv(out_file, index=False)


# ============================================================
# 3. 批量训练 XGBoost
# ============================================================

def run_xgboost_training(run_id, external_test_path, random_seed):
    print(f"\n=== [Run {run_id}] Step 3: XGBoost 训练 ===")

    from xgboost_externaltest import process_pipeline

    for n in SAMPLE_SIZES:
        input_folder = os.path.join(
            SELECTED_BASE_DIR, f"run_{run_id:02d}", f"{n}_samples"
        )

        output_dir = os.path.join(
            RESULT_BASE_DIR, f"run_{run_id:02d}", f"xgb_{n}_samples"
        )

        process_pipeline(
            input_folder=input_folder,
            external_test_file=external_test_path,
            output_base=output_dir,
            random_state=random_seed
        )


# ============================================================
# 主入口（自动运行 N_RUNS 次）
# ============================================================

if __name__ == "__main__":

    print("\n🚀 启动多次实验流水线")
    print(f"总运行次数: {N_RUNS}")

    for run_id in range(1, N_RUNS + 1):
        random_seed = BASE_RANDOM_SEED + run_id

        print("\n" + "=" * 60)
        print(f"🔥 开始 Run {run_id} | random_seed = {random_seed}")
        print("=" * 60)

        ext_test_path = generate_external_test(run_id, random_seed)
        generate_selected_datasets(run_id, random_seed)
        run_xgboost_training(run_id, ext_test_path, random_seed)

    print("\n🎉🎉🎉 所有 Runs 完成！")
