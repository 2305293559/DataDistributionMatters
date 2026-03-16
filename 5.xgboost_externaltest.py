# filename: tabular_trainer.py

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import GridSearchCV, train_test_split, RepeatedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import xgboost as xgb

warnings.filterwarnings('ignore')


class TabularModelTrainer:
    def __init__(self, data_path=None, output_dir="model_outputs_csv", random_state=None):
        self.data_path = data_path
        self.output_dir = output_dir
        self.random_state = random_state
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 容器
        self.models = {}
        self.scalers = {}
        self.results = {}

        # 数据占位
        self.X_train, self.X_val = None, None
        self.y_train, self.y_val = None, None
        self.X_train_scaled, self.X_val_scaled = None, None
        self.y_train_scaled, self.y_val_scaled = None, None
        
        # 外部测试集
        self.X_ext_test_scaled = None
        self.y_ext_test = None

        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self, data_path=None, target_col='y', feature_cols=None):
        if data_path is None: data_path = self.data_path
        try:
            data = pd.read_csv(data_path)
            if feature_cols is None:
                feature_cols = [c for c in data.columns if c != target_col]
            self.X = data[feature_cols].copy()
            self.y = data[target_col].copy()
            return True
        except Exception as e:
            print(f"加载数据失败: {e}")
            return False

    def load_external_test(self, test_csv_path, target_col='y'):
        if not os.path.exists(test_csv_path):
            print(f"外部测试文件 {test_csv_path} 不存在")
            return False
        try:
            ext_data = pd.read_csv(test_csv_path)
            feature_cols = [c for c in ext_data.columns if c != target_col]
            X_ext = ext_data[feature_cols].copy()
            self.y_ext_test = ext_data[target_col].copy()

            if 'X' in self.scalers:
                self.X_ext_test_scaled = pd.DataFrame(
                    self.scalers['X'].transform(X_ext),
                    columns=X_ext.columns
                )
            else:
                self.X_ext_test_scaled = X_ext
            
            print(f"[External] 成功加载外部测试集: {test_csv_path}")
            return True
        except Exception as e:
            print(f"处理外部测试集失败: {e}")
            return False

    def prepare_data(self, test_size=0.2, scale_features=True, scale_target=True):
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y, test_size=test_size, random_state=self.random_state
        )
        if scale_features:
            self.X_scaler = StandardScaler().fit(self.X_train)
            self.X_train_scaled = pd.DataFrame(self.X_scaler.transform(self.X_train), columns=self.X_train.columns)
            self.X_val_scaled = pd.DataFrame(self.X_scaler.transform(self.X_val), columns=self.X_val.columns)
            self.scalers['X'] = self.X_scaler
        if scale_target:
            self.y_scaler = StandardScaler().fit(self.y_train.values.reshape(-1,1))
            self.y_train_scaled = pd.Series(self.y_scaler.transform(self.y_train.values.reshape(-1,1)).flatten())
            self.y_val_scaled = pd.Series(self.y_scaler.transform(self.y_val.values.reshape(-1,1)).flatten())
            self.scalers['y'] = self.y_scaler
        else:
            self.y_train_scaled, self.y_val_scaled = self.y_train, self.y_val

    def train_xgboost(self):
        param_grid = {
            'n_estimators': [200, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8],
            'colsample_bytree': [0.8]
        }
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=self.random_state, n_jobs=-1)
        if self.device == "cuda":
            xgb_model.set_params(tree_method='gpu_hist', gpu_id=0)
        grid = GridSearchCV(xgb_model, param_grid, scoring='r2', cv=5)
        grid.fit(self.X_train_scaled, self.y_train_scaled)
        self.models['xgboost'] = grid.best_estimator_
        self.results['xgboost'] = {'best_params': grid.best_params_}
        print(f"最佳参数: {grid.best_params_}")

    def compute_metrics(self, model_name):
        model = self.models[model_name]
        def get_r2(X_s, y_orig):
            if X_s is None: return np.nan
            preds_s = model.predict(X_s)
            if 'y' in self.scalers:
                preds = self.scalers['y'].inverse_transform(preds_s.reshape(-1,1)).flatten()
            else:
                preds = preds_s
            return r2_score(y_orig, preds)

        train_r2 = get_r2(self.X_train_scaled, self.y_train)
        val_r2 = get_r2(self.X_val_scaled, self.y_val)
        ext_r2 = get_r2(self.X_ext_test_scaled, self.y_ext_test)

        rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=self.random_state)
        cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train_scaled, cv=rkf, scoring='r2')

        self.results[model_name].update({
            'train_r2': train_r2,
            'test_r2': val_r2,
            'external_test_r2': ext_r2,
            'cv_mean_r2': np.mean(cv_scores)
        })
        print(f"\n--- {model_name} 结果 ---")
        print(f"Train R2: {train_r2:.4f}, Internal Test R2: {val_r2:.4f}, External Test R2: {ext_r2:.4f}, CV Mean R2: {np.mean(cv_scores):.4f}")

    def save_all_results(self):
        os.makedirs(self.output_dir, exist_ok=True)
        joblib.dump(self.models['xgboost'], os.path.join(self.output_dir, "xgboost_model.pkl"))
        res_df = pd.DataFrame(self.results).T
        res_df.to_csv(os.path.join(self.output_dir, "evaluation_summary.csv"))


# --------------------- 批量处理接口 ---------------------
def process_pipeline(
    input_folder: str,
    external_test_file: str,
    output_base: str,
    random_state: int = None
):
    """
    批量处理指定文件夹下所有 CSV 文件
    """
    csv_files = list(Path(input_folder).glob("*.csv"))
    for f in csv_files:
        print(f"\n处理文件: {f.name}")
        out_path = os.path.join(output_base, f.stem)
        os.makedirs(out_path, exist_ok=True)
        trainer = TabularModelTrainer(data_path=str(f), output_dir=out_path, random_state=random_state)
        if not trainer.load_data(): 
            continue
        trainer.prepare_data()
        trainer.load_external_test(external_test_file)
        trainer.train_xgboost()
        trainer.compute_metrics('xgboost')
        trainer.save_all_results()


# --------------------- 示例调用 ---------------------
if __name__ == "__main__":
    # 将硬编码部分替换为可配置参数
    INPUT_FOLDER = "func9"
    EXTERNAL_TEST_FILE = "external_test_fun9.csv"
    OUTPUT_BASE = "10dim_xgb_with_external_test"
    RANDOM_STATE = 42

    process_pipeline(INPUT_FOLDER, EXTERNAL_TEST_FILE, OUTPUT_BASE, random_state=RANDOM_STATE)

        
        