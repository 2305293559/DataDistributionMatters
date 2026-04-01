# -*- coding: utf-8 -*-
"""
Random Forest tabular trainer (改写版)
- 从 experiments_f9/run_001 ~ run_010/optimized_al_samples/... 读取 CSV 并训练
- 支持使用已有 external_test（多种常见命名），若不存在则在输出目录生成外部测试集
- 默认 F9 (Rastrigin) 对应 f_num=5（参见 objective_function 映射）
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
    RepeatedKFold,
    cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

# ===================== 目标函数定义 =====================
def objective_function(x: np.ndarray, f_num: int) -> float:
    if f_num == 1:
        coef = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        return float(np.sum(coef * x))
    elif f_num == 2:
        coef = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        return float(np.sum(coef * x**2))
    elif f_num == 3:
        res = 0.0
        for i in range(len(x) - 1):
            res += 100 * (x[i + 1] - x[i]**2)**2 + (x[i] - 1)**2
        return float(res)
    elif f_num == 4:
        return float(np.sum(-x * np.sin(np.sqrt(np.abs(x)))))
    elif f_num == 5:
        # Rastrigin (your F9 in the previous script)
        return float(np.sum(x**2 - 10 * np.cos(np.pi * x)) + 10 * len(x))
    elif f_num == 6:
        res = 0.0
        for i in range(len(x)):
            res += np.sin(x[i]) * (np.sin((i + 1) * x[i]**2 / np.pi))**20
        return float(-res)
    else:
        raise ValueError(f"Unknown function F{f_num}")


# ===================== External test 生成 =====================
DomainType = Dict[str, Tuple[float, float]]

def generate_external_test(
    n_samples: int,
    domain: DomainType,
    f_num: int,
    output_path: str
):
    feature_names = list(domain.keys())
    lows = np.array([domain[k][0] for k in feature_names])
    highs = np.array([domain[k][1] for k in feature_names])

    X = np.random.uniform(lows, highs, size=(n_samples, len(feature_names)))
    y = np.array([objective_function(x, f_num) for x in X])

    df = pd.DataFrame(X, columns=feature_names)
    df["y"] = y

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[External Test] 已生成: {output_path}")


# ===================== Trainer =====================
class TabularModelTrainer:
    def __init__(self, data_path, output_dir, random_state=None):
        self.data_path = str(data_path)
        self.output_dir = str(output_dir)
        self.random_state = random_state

        self.models = {}
        self.scalers = {}
        self.results = {}

        self.X_train = self.X_val = None
        self.y_train = self.y_val = None
        self.X_train_scaled = self.X_val_scaled = None
        self.y_train_scaled = self.y_val_scaled = None

        self.X_ext_scaled = None
        self.y_ext = None

        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        data = pd.read_csv(self.data_path)
        self.X = data[[f"x{i}" for i in range(1, 11)]]
        self.y = data["y"]

    def prepare_data(self):
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.random_state
        )

        self.X_scaler = StandardScaler().fit(self.X_train)
        self.y_scaler = StandardScaler().fit(self.y_train.values.reshape(-1, 1))

        self.X_train_scaled = self.X_scaler.transform(self.X_train)
        self.X_val_scaled = self.X_scaler.transform(self.X_val)
        self.y_train_scaled = self.y_scaler.transform(self.y_train.values.reshape(-1, 1)).ravel()

        self.scalers["X"] = self.X_scaler
        self.scalers["y"] = self.y_scaler

    def load_external_test(self, csv_path):
        ext = pd.read_csv(csv_path)
        X_ext = ext[[f"x{i}" for i in range(1, 11)]]
        self.y_ext = ext["y"]
        self.X_ext_scaled = self.scalers["X"].transform(X_ext)

    def train_rf(self):
        param_grid = {
            "n_estimators": [1000],
            "max_features": [0.2, 0.3, 0.4],
        }

        rf = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
        grid = GridSearchCV(rf, param_grid, cv=5, scoring="r2", n_jobs=-1)
        grid.fit(self.X_train_scaled, self.y_train_scaled)

        self.models["rf"] = grid.best_estimator_
        self.results["rf"] = {"best_params": grid.best_params_}

    def compute_metrics(self):
        model = self.models["rf"]

        def eval_r2(Xs, y):
            preds = model.predict(Xs)
            preds = self.scalers["y"].inverse_transform(preds.reshape(-1, 1)).ravel()
            return r2_score(y, preds)

        rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=self.random_state)
        cv = cross_val_score(model, self.X_train_scaled, self.y_train_scaled, cv=rkf)

        self.results["rf"].update({
            "train_r2": eval_r2(self.X_train_scaled, self.y_train),
            "val_r2": eval_r2(self.X_val_scaled, self.y_val),
            "external_test_r2": eval_r2(self.X_ext_scaled, self.y_ext),
            "cv_mean_r2": float(cv.mean()),
            "cv_std_r2": float(cv.std()),
        })

    def save(self):
        joblib.dump(self.models["rf"], os.path.join(self.output_dir, "rf_model.pkl"))
        pd.DataFrame(self.results).T.to_csv(os.path.join(self.output_dir, "evaluation_summary.csv"))
        print(f"[Saved] {self.output_dir}")


# ===================== Pipeline (改写) =====================
def process_pipeline(base_dir, output_base, f_num, random_state=42):
    """
    现在的行为:
    - 遍历 run_001 .. run_010
    - 在每个 run 下查找 optimized_al_samples 下的所有 CSV（递归）
    - 尝试找到该 run 的 external_test，如果不存在则在 output_base 下为该 run 生成一个
    - 对每个 CSV 进行训练、评估并存储到 output_base/run_xxx/<csv_stem>/
    """

    domain = {f"x{i}": (-5.12, 5.12) for i in range(1, 11)}
    base_path = Path(base_dir)
    out_base_path = Path(output_base)
    out_base_path.mkdir(parents=True, exist_ok=True)

    # runs 使用 run_001 ~ run_010（与 batch_experiment 输出一致）
    for i in range(1, 11):
        run_name = f"run_{i:03d}"
        run_dir = base_path / run_name
        if not run_dir.exists():
            print(f"[Skip] 找不到 {run_dir}, 跳过")
            continue

        print(f"\n=== Processing {run_name} ===")

        # 优先尝试使用 run 下已有的 external_test 文件（常见命名）
        possible_external_names = [
            f"external_test_F{f_num}.csv",
            "external_test_f9.csv",   # 兼容你最初的命名
            "external_test.csv"
        ]
        existing_ext_path = None
        run_ext_dir = run_dir / "external_test"
        for name in possible_external_names:
            candidate = run_ext_dir / name
            if candidate.exists():
                existing_ext_path = candidate
                break

        # 如果没有，就在输出目录下生成 external test（每个 run 只生成一次）
        run_output_dir = out_base_path / run_name
        run_output_dir.mkdir(parents=True, exist_ok=True)
        out_ext_dir = run_output_dir / "external_test"
        out_ext_dir.mkdir(parents=True, exist_ok=True)
        out_ext_path = out_ext_dir / f"external_test_F{f_num}.csv"

        if existing_ext_path:
            ext_path_to_use = str(existing_ext_path)
            print(f"[External Test] 使用已有外部测试集: {ext_path_to_use}")
        else:
            if not out_ext_path.exists():
                print(f"[External Test] 未找到现成外部测试集，生成: {out_ext_path}")
                generate_external_test(
                    n_samples=1000,
                    domain=domain,
                    f_num=f_num,
                    output_path=str(out_ext_path)
                )
            ext_path_to_use = str(out_ext_path)

        # 寻找 optimized_al_samples 下的所有 CSV
        optimized_dir = run_dir / "optimized_al_samples"
        if not optimized_dir.exists():
            print(f"[Skip] {optimized_dir} 不存在，跳过该 run")
            continue

        # 递归查找 CSV 文件（会找到 n20/*.csv, n50/*.csv 等）
        csv_list = sorted(optimized_dir.rglob("*.csv"))
        if not csv_list:
            print(f"[Skip] 在 {optimized_dir} 下未找到 CSV 文件")
            continue

        for csv_path in csv_list:
            try:
                print(f"  -> 处理 CSV: {csv_path}")
                out_dir = run_output_dir / csv_path.stem
                out_dir.mkdir(parents=True, exist_ok=True)

                trainer = TabularModelTrainer(csv_path, out_dir, random_state)

                trainer.load_data()
                trainer.prepare_data()
                trainer.load_external_test(ext_path_to_use)

                trainer.train_rf()
                trainer.compute_metrics()
                trainer.save()

            except Exception as e:
                print(f"  [Error] 处理 {csv_path} 出错: {e}")
                import traceback
                traceback.print_exc()
                continue


# ===================== Main =====================
if __name__ == "__main__":
    # 指定包含 run_001...run_010 的父目录（例如 experiments_f9）
    BASE_DIR = "f5/experiments_f5_2.0"               # <- 修改为你的根目录
    OUTPUT_BASE = "experiments_f5_2.0_rf_results" # <- 训练输出目录
    F_NUM = 1  # 5 对应上面 objective_function 中的 Rastrigin（F9）
    RANDOM_STATE = 42

    process_pipeline(BASE_DIR, OUTPUT_BASE, F_NUM, random_state=RANDOM_STATE)
