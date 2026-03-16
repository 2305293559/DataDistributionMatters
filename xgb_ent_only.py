# filename: tabular_trainer.py

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import torch
from pathlib import Path
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


class TabularModelTrainer:
    def __init__(self, data_path=None, output_dir="model_outputs_csv", random_state=None):
        self.data_path = data_path
        self.output_dir = output_dir
        self.random_state = random_state
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.models = {}
        self.scalers = {}
        self.results = {}

        self.X = None
        self.y = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.X_train_scaled = None
        self.X_val_scaled = None
        self.y_train_scaled = None
        self.y_val_scaled = None

        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self, data_path=None, target_col="y", feature_cols=None):
        if data_path is None:
            data_path = self.data_path
        try:
            data = pd.read_csv(data_path)
            if feature_cols is None:
                feature_cols = [c for c in data.columns if c != target_col]
            self.X = data[feature_cols].copy()
            self.y = data[target_col].copy()
            return True
        except Exception as e:
            print(f"[ERROR] 加载数据失败: {e}")
            return False

    def prepare_data(self, test_size=0.2, scale_features=True, scale_target=True):
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=self.random_state
        )

        if scale_features:
            self.X_scaler = StandardScaler().fit(self.X_train)
            self.X_train_scaled = self.X_scaler.transform(self.X_train)
            self.X_val_scaled = self.X_scaler.transform(self.X_val)
            self.scalers["X"] = self.X_scaler
        else:
            self.X_train_scaled = self.X_train
            self.X_val_scaled = self.X_val

        if scale_target:
            self.y_scaler = StandardScaler().fit(self.y_train.values.reshape(-1, 1))
            self.y_train_scaled = self.y_scaler.transform(
                self.y_train.values.reshape(-1, 1)
            ).ravel()
            self.y_val_scaled = self.y_scaler.transform(
                self.y_val.values.reshape(-1, 1)
            ).ravel()
            self.scalers["y"] = self.y_scaler
        else:
            self.y_train_scaled = self.y_train
            self.y_val_scaled = self.y_val

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

        print(f"[OK] Best params: {grid.best_params_}")

    def compute_metrics(self, model_name):
        model = self.models[model_name]

        def r2(X, y_true):
            preds = model.predict(X)
            if "y" in self.scalers:
                preds = self.scalers["y"].inverse_transform(
                    preds.reshape(-1, 1)
                ).ravel()
            return r2_score(y_true, preds)

        train_r2 = r2(self.X_train_scaled, self.y_train)
        val_r2 = r2(self.X_val_scaled, self.y_val)

        rkf = RepeatedKFold(
            n_splits=5,
            n_repeats=2,
            random_state=self.random_state
        )

        cv_scores = cross_val_score(
            model,
            self.X_train_scaled,
            self.y_train_scaled,
            cv=rkf,
            scoring="r2",
            n_jobs=-1
        )

        self.results[model_name].update({
            "train_r2": train_r2,
            "test_r2": val_r2,
            "cv_mean_r2": float(np.mean(cv_scores)),
            "cv_std_r2": float(np.std(cv_scores)),
        })

        print(
            f"[R2] train={train_r2:.4f} "
            f"test={val_r2:.4f} "
            f"cv={np.mean(cv_scores):.4f}±{np.std(cv_scores):.4f}"
        )

    def save_all_results(self):
        joblib.dump(
            self.models["xgboost"],
            os.path.join(self.output_dir, "xgboost_model.pkl")
        )
        pd.DataFrame(self.results).T.to_csv(
            os.path.join(self.output_dir, "evaluation_summary.csv")
        )


# ===================== 批量处理接口 =====================
def process_pipeline(base_dir: str, output_base: str, random_state: int = None):

    allowed_csv_names = [
        "10dim_active_learning_20_fun1.csv",
        "10dim_active_learning_50_fun1.csv",
        "10dim_active_learning_80_fun1.csv",
        "10dim_active_learning_100_fun1.csv",
        "10dim_active_learning_200_fun1.csv",
        "10dim_active_learning_300_fun1.csv",
        "10dim_active_learning_500_fun1.csv",
        "10dim_active_learning_1000_fun1.csv",
    ]

    base_dir = Path(base_dir)
    print(f"[INFO] base_dir = {base_dir.resolve()}")

    # ✅ 只遍历 run_1 ~ run_10
    for i in range(1, 11):
        run_dir = base_dir / f"run_{i}"
        if not run_dir.exists():
            print(f"[SKIP] run_{i} not found")
            continue

        print(f"\n========== run_{i} ==========")

        for csv_name in allowed_csv_names:
            csv_path = run_dir / csv_name
            if not csv_path.exists():
                print(f"[SKIP] {csv_name} not found")
                continue

            print(f"[TRAIN] {csv_path.name}")

            out_path = (
                Path(output_base)
                / run_dir.name
                / csv_path.stem
            )
            out_path.mkdir(parents=True, exist_ok=True)

            trainer = TabularModelTrainer(
                data_path=str(csv_path),
                output_dir=str(out_path),
                random_state=random_state
            )

            if not trainer.load_data():
                continue

            trainer.prepare_data()
            trainer.train_xgboost()
            trainer.compute_metrics("xgboost")
            trainer.save_all_results()

# ===================== 示例调用 =====================
if __name__ == "__main__":
    BASE_DIR = "auto_outputs_50_f9_Generalized Rastrigin's Function/sampling_data"
    OUTPUT_BASE = "auto_outputs_50_f9_Generalized Rastrigin's Function_xgb_results_ent_only"
    RANDOM_STATE = 42

    process_pipeline(
        base_dir=BASE_DIR,
        output_base=OUTPUT_BASE,
        random_state=RANDOM_STATE
    )
