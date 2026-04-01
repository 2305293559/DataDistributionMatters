# -*- coding: utf-8 -*-
"""
Random Forest tabular trainer
- 支持 run_1 ~ run_10 批量处理
- 自动生成 external test
- 支持自定义目标函数 F1~F6
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import torch
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
        self.data_path = data_path
        self.output_dir = output_dir
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
            "cv_mean_r2": cv.mean(),
            "cv_std_r2": cv.std(),
        })

    def save(self):
        joblib.dump(self.models["rf"], f"{self.output_dir}/rf_model.pkl")
        pd.DataFrame(self.results).T.to_csv(f"{self.output_dir}/evaluation_summary.csv")


# ===================== Pipeline =====================
def process_pipeline(base_dir, output_base, f_num, random_state=42):

    domain = {f"x{i}": (-1, 1) for i in range(1, 11)}

    for i in range(1, 6):
        run_dir = Path(base_dir) / f"run_{i}"
        if not run_dir.exists():
            continue

        # ⭐ 每个 run 独立的 external_test 目录
        run_output_dir = Path(output_base) / run_dir.name
        ext_dir = run_output_dir / "external_test"
        ext_path = ext_dir / f"external_test_F{f_num}.csv"

        # ⭐ 每个 run 只生成一次 external test
        if not ext_path.exists():
            generate_external_test(
                n_samples=1000,
                domain=domain,
                f_num=f_num,
                output_path=str(ext_path)
            )

        for csv in run_dir.glob("*.csv"):
            out_dir = run_output_dir / csv.stem
            trainer = TabularModelTrainer(csv, out_dir, random_state)

            trainer.load_data()
            trainer.prepare_data()
            trainer.load_external_test(ext_path)

            trainer.train_rf()
            trainer.compute_metrics()
            trainer.save()


# ===================== Main =====================
if __name__ == "__main__":
    BASE_DIR = "auto_outputs_50_f1_Sum Squares Function/sampling_data"
    OUTPUT_BASE = "auto_outputs_50_f1_3.0_rf_results"
    F_NUM = 2

    process_pipeline(BASE_DIR, OUTPUT_BASE, F_NUM)
