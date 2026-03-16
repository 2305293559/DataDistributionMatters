# ============================================================
# batch_run_modular.py
# 批量实验总控脚本（optimized_active_learning 稳定版）
# ============================================================

import os
import random
import importlib.util
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# =====================【实验参数区】=====================
EXTERNAL_TEST_FILENAME = "external_test_f8.csv"
SAMPLE_SIZES = [20, 50, 80, 100, 200, 300, 500, 1000]
N_RUNS = 10
N_EXTERNAL_TEST_SAMPLES = 1000
BASE_OUTPUT_DIR = "experiments_f8"
SEED_BASE = 2025

# 注意：Schwefel函数的典型定义域是 [-500, 500]
DOMAIN = {
    "x1": (-60, 60),  # Schwefel函数的典型定义域
    "x2": (-60, 60), 
    "x3": (-60, 60),
    "x4": (-60, 60),
    "x5": (-60, 60),
    "x6": (-60, 60),
    "x7": (-60, 60),
    "x8": (-60, 60),
    "x9": (-60, 60),
    "x10": (-60, 60)
}

# =====================【目标函数区】=====================
def default_objective_function(x: np.ndarray) -> float:
  """
  十维Schwefel函数 (F8函数)
  公式: f(x) = sum(-x_i * sin(sqrt(|x_i|)))
  这是一个多峰函数，具有许多局部最小值，全局最小值在x_i ≈ 420.9687
  """
  x_array = np.asarray(x)
  
  # 计算Schwefel函数值
  result = np.sum(-x_array * np.sin(np.sqrt(np.abs(x_array))))
  
  return float(result)

# =====================【工具函数】=====================
def load_module_from_path(path: str, module_name: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"模块文件不存在: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_all_seeds(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# =====================【主控制函数】=====================
def batch_experiment(
    objective_function,
    domain,
    sample_sizes,
    n_runs,
    n_external_test_samples,
    external_test_filename,
    base_output_dir,
    gen_module_path,
    al_module_path,
    xgb_module_path,
    seed_base=2025,
):
    """
    批量实验总控函数（optimized_active_learning 版本）
    设计原则：
      - 一个 run → 一个 external test
      - 一个 n → 一个 AL CSV → 一个 XGB 结果目录
      - process_pipeline 每次只看到一个 CSV
    """

    print("\n==== ENTER batch_experiment ====")
    print(f"DEBUG n_runs = {n_runs}")
    print(f"DEBUG sample_sizes = {sample_sizes}\n")

    # ---------- 加载模块 ----------
    gen_mod = load_module_from_path(gen_module_path, "gen")
    al_mod = load_module_from_path(al_module_path, "optimized_al")
    xgb_mod = load_module_from_path(xgb_module_path, "xgb")

    base_output = Path(base_output_dir)
    base_output.mkdir(exist_ok=True)

    all_results_records = []

    # ===================== RUN LOOP =====================
    for run_id in range(1, n_runs + 1):
        print(f"\nDEBUG entering run {run_id}/{n_runs}")

        run_name = f"run_{run_id:03d}"
        run_dir = base_output / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        seed = seed_base + run_id
        set_all_seeds(seed)

        print(f"================ {run_name} | seed={seed} ================")

        # ---------- External Test ----------
        external_dir = run_dir / "external_test"
        external_dir.mkdir(exist_ok=True)

        try:
            external_file_path = gen_mod.generate_random_samples(
                n_samples=n_external_test_samples,
                domain=domain,
                objective_function=objective_function,
                output_dir=str(external_dir),
                output_filename=external_test_filename,
            )

            if not external_file_path or not os.path.exists(external_file_path):
                raise RuntimeError("外部测试集生成失败")

            print(f"[External test] {external_file_path}")

        except Exception as e:
            print(f"❌ run={run_id}: 外部测试集失败，跳过该 run")
            traceback.print_exc()
            continue

        # ---------- Optimized Active Learning ----------
        al_root = run_dir / "optimized_al_samples"
        xgb_root = run_dir / "xgb_results"
        al_root.mkdir(exist_ok=True)
        xgb_root.mkdir(exist_ok=True)

        # ===================== N LOOP =====================
        for n in sample_sizes:
            print(f"  run={run_id}, n_samples={n}")

            try:
                # 1️⃣ 为每个 n 建立独立 AL 目录
                al_n_dir = al_root / f"n{n}"
                al_n_dir.mkdir(exist_ok=True)

                # 2️⃣ 主动学习采样
                df = al_mod.generate_slpa_samples(
                    domain=domain,
                    objective_function=objective_function,
                    n_samples=n,
                    population_size=None,  # 接口兼容
                    n_offspring=None,      # 接口兼容
                    seed=seed,
                    verbose=False,
                )

                if df is None or df.empty:
                    print(f"    ⚠️ AL 返回空数据，跳过 n={n}")
                    continue

                train_csv = al_n_dir / f"optimized_al_n{n}.csv"
                df.to_csv(train_csv, index=False)

                # 3️⃣ XGBoost（一次只看一个 CSV）
                xgb_n_dir = xgb_root / f"n{n}"
                xgb_n_dir.mkdir(exist_ok=True)

                xgb_mod.process_pipeline(
                    input_folder=str(al_n_dir),   # ⭐ 核心修复
                    external_test_file=str(external_file_path),
                    output_base=str(xgb_n_dir),
                    random_state=seed,
                )

                # 4️⃣ 汇总评估结果
                eval_csv = xgb_n_dir / train_csv.stem / "evaluation_summary.csv"
                if eval_csv.exists():
                    eval_df = pd.read_csv(eval_csv)
                    eval_df["run_id"] = run_id
                    eval_df["n_samples"] = n
                    eval_df["sampler"] = "optimized_active_learning"
                    all_results_records.append(eval_df)
                else:
                    print(f"    ⚠️ 未找到 evaluation_summary.csv")

            except Exception as e:
                print(f"    ❌ run={run_id}, n={n} 出错")
                traceback.print_exc()
                continue

    # ===================== 汇总输出 =====================
    if all_results_records:
        summary_dir = base_output / "summary"
        summary_dir.mkdir(exist_ok=True)

        final_summary = pd.concat(all_results_records, ignore_index=True)
        final_csv = summary_dir / "all_runs_evaluation_summary.csv"
        final_summary.to_csv(final_csv, index=False)

        print(f"\n✅ 所有实验完成，汇总结果已保存至:\n{final_csv}")
    else:
        print("\n⚠️ 未生成任何结果，请检查实验配置和模块。")


# =====================【CLI 入口】=====================
if __name__ == "__main__":
    BASE_DIR = Path.cwd()

    batch_experiment(
        objective_function=default_objective_function,
        domain=DOMAIN,
        sample_sizes=SAMPLE_SIZES,
        n_runs=N_RUNS,
        n_external_test_samples=N_EXTERNAL_TEST_SAMPLES,
        external_test_filename=EXTERNAL_TEST_FILENAME,
        base_output_dir=BASE_OUTPUT_DIR,
        gen_module_path=str(BASE_DIR / "1.1gen_random_testset.py"),
        al_module_path=str(BASE_DIR / "2.Bayesian sampling for maximization.py"),
        xgb_module_path=str(BASE_DIR / "5.xgboost_externaltest.py"),
        seed_base=SEED_BASE,
    )
