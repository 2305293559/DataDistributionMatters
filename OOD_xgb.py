# -*- coding: utf-8 -*-
"""
XGBoost tabular trainer
- 支持 run_1 ~ run_5 批量处理
- 自动生成 external test
- 支持自定义目标函数 F1~F6
- External test 定义域: (-1.2, -1) U (1, 1.2)
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, Tuple, List
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
    RepeatedKFold,
    cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import xgboost as xgb

warnings.filterwarnings("ignore")

# ===================== 目标函数 =====================
def objective_function(x: np.ndarray, f_num: int) -> float:
    if f_num == 1:
        coef = np.arange(10, 110, 10)
        return float(np.sum(coef * x))
    elif f_num == 2:
        coef = np.arange(10, 110, 10)
        return float(np.sum(coef * x**2))
    elif f_num == 3:  # Rosenbrock
        return float(np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2))
    elif f_num == 4:  # Schwefel
        return float(np.sum(-x * np.sin(np.sqrt(np.abs(x)))))
    elif f_num == 5:  # Rastrigin
        return float(np.sum(x**2 - 10 * np.cos(np.pi * x)) + 10 * len(x))
    elif f_num == 6:
        return float(-np.sum(
            np.sin(x) * (np.sin((np.arange(1, len(x)+1) * x**2) / np.pi))**20
        ))
    else:
        raise ValueError(f"Unknown function F{f_num}")

# ===================== External Test =====================
DomainType = Dict[str, List[Tuple[float, float]]]

def generate_external_test(n_samples, domain, f_num, output_path):
    feature_names = list(domain.keys())
    dim = len(feature_names)

    X = np.zeros((n_samples, dim))

    for j, k in enumerate(feature_names):
        # 随机选择左区间 or 右区间
        choose_right = np.random.rand(n_samples) > 0.5

        lows = np.where(
            choose_right,
            domain[k][1][0],  # (1.0, 1.2)
            domain[k][0][0],  # (-1.2, -1.0)
        )
        highs = np.where(
            choose_right,
            domain[k][1][1],
            domain[k][0][1],
        )

        X[:, j] = np.random.uniform(lows, highs)

    y = np.array([objective_function(x, f_num) for x in X])

    df = pd.DataFrame(X, columns=feature_names)
    df["y"] = y

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

# ===================== Trainer =====================
class TabularModelTrainer:
    def __init__(self, data_path, output_dir, random_state=None):
        self.data_path = data_path
        self.output_dir = output_dir
        self.random_state = random_state
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.models = {}
        self.scalers = {}
        self.results = {}

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
        self.y_train_scaled = self.y_scaler.transform(
            self.y_train.values.reshape(-1, 1)
        ).ravel()

        self.scalers["X"] = self.X_scaler
        self.scalers["y"] = self.y_scaler

    def load_external_test(self, csv_path):
        ext = pd.read_csv(csv_path)
        X_ext = ext[[f"x{i}" for i in range(1, 11)]]
        self.y_ext = ext["y"]
        self.X_ext_scaled = self.scalers["X"].transform(X_ext)

    # ===================== XGBoost =====================
    def train_xgboost(self):
        param_grid = {
            "n_estimators": [200, 500],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
        }

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=self.random_state,
            n_jobs=-1,
            tree_method="gpu_hist" if self.device == "cuda" else "hist"
        )

        grid = GridSearchCV(
            model,
            param_grid,
            scoring="r2",
            cv=5,
            n_jobs=-1
        )

        grid.fit(self.X_train_scaled, self.y_train_scaled)

        self.models["xgboost"] = grid.best_estimator_
        self.results["xgboost"] = {"best_params": grid.best_params_}

    # ===================== Metrics =====================
    def compute_metrics(self):
        model = self.models["xgboost"]

        def eval_r2(Xs, y):
            preds = model.predict(Xs)
            preds = self.scalers["y"].inverse_transform(
                preds.reshape(-1, 1)
            ).ravel()
            return r2_score(y, preds)

        rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=self.random_state)
        cv = cross_val_score(
            model,
            self.X_train_scaled,
            self.y_train_scaled,
            cv=rkf,
            scoring="r2"
        )

        self.results["xgboost"].update({
            "train_r2": eval_r2(self.X_train_scaled, self.y_train),
            "val_r2": eval_r2(self.X_val_scaled, self.y_val),
            "external_test_r2": eval_r2(self.X_ext_scaled, self.y_ext),
            "cv_mean_r2": cv.mean(),
            "cv_std_r2": cv.std(),
        })

    def save(self):
        joblib.dump(
            self.models["xgboost"],
            os.path.join(self.output_dir, "xgboost_model.pkl")
        )
        pd.DataFrame(self.results).T.to_csv(
            os.path.join(self.output_dir, "evaluation_summary.csv")
        )

# ===================== Pipeline =====================
def process_pipeline(base_dir, output_base, f_num, random_state=42):
    domain = {
        f"x{i}": [(-6.144, -5.12), (5.12, 6.144)]
        for i in range(1, 11)
    }

    for i in range(1, 11):
        run_dir = Path(base_dir) / f"run_{i}"
        if not run_dir.exists():
            continue

        run_output_dir = Path(output_base) / run_dir.name
        ext_path = run_output_dir / "external_test" / f"external_test_F{f_num}.csv"

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
            trainer.train_xgboost()
            trainer.compute_metrics()
            trainer.save()
# ===================== Main =====================
if __name__ == "__main__":
    BASE_DIR = "auto_outputs_50_f9_Generalized Rastrigin's Function/sampling_data"
    OUTPUT_BASE = "auto_outputs_50_f9_OOD_xgb_results_2.0"
    F_NUM = 5

    process_pipeline(
        base_dir=BASE_DIR,
        output_base=OUTPUT_BASE,
        f_num=F_NUM,
        random_state=42
    )
