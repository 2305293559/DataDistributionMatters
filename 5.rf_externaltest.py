# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 15:15:26 2025

@author: 91278
"""
import os
import warnings
import joblib
import numpy as np
import pandas as pd
# import torch
from pathlib import Path
from sklearn.model_selection import GridSearchCV, train_test_split, RepeatedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
# 修改点1：引入随机森林，移除xgboost
from sklearn.ensemble import RandomForestRegressor


warnings.filterwarnings('ignore')


class TabularModelTrainer:
    def __init__(self, data_path=None, output_dir="model_outputs_csv", random_state=None):
        """
        初始化训练器，增加外部测试集路径支持
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.random_state = random_state
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 存储容器
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.cv_results = {}

        # 数据占位
        self.X_train, self.X_val = None, None
        self.y_train, self.y_val = None, None
        self.X_train_scaled, self.X_val_scaled = None, None
        self.y_train_scaled, self.y_val_scaled = None, None
        
        # 外部测试集占位
        self.X_ext_test_scaled = None
        self.y_ext_test = None

        os.makedirs(self.output_dir, exist_ok=True)


    def load_data(self, data_path=None, target_col='y', feature_cols=None):
        """
        加载初始训练数据集
        """
        if data_path is None: data_path = self.data_path
        try:
            data = pd.read_csv(data_path)
            if feature_cols is None:
                feature_cols = [f'x{i}' for i in range(1, 11)] if 'x1' in data.columns else [c for c in data.columns if c != target_col]
            
            self.X = data[feature_cols].copy()
            self.y = data[target_col].copy()
            return True
        except Exception as e:
            print(f"加载数据失败: {e}")
            return False


    def load_external_test(self, test_csv_path, target_col='y'):
        """
        加载外部 test.csv 文件并使用训练集的 scaler 进行转换
        """
        if not os.path.exists(test_csv_path):
            print(f"外部测试文件 {test_csv_path} 不存在")
            return False
        
        try:
            ext_data = pd.read_csv(test_csv_path)
            feature_cols = [f'x{i}' for i in range(1, 11)]
            X_ext = ext_data[feature_cols].copy()
            self.y_ext_test = ext_data[target_col].copy()

            # 使用训练集的 X_scaler 进行缩放
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
        """
        划分数据集并进行标准化
        """
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y, test_size=test_size, random_state=self.random_state
        )

        if scale_features:
            self.X_scaler = StandardScaler().fit(self.X_train)
            self.X_train_scaled = pd.DataFrame(self.X_scaler.transform(self.X_train), columns=self.X_train.columns)
            self.X_val_scaled = pd.DataFrame(self.X_scaler.transform(self.X_val), columns=self.X_val.columns)
            self.scalers['X'] = self.X_scaler

        if scale_target:
            self.y_scaler = StandardScaler().fit(self.y_train.values.reshape(-1, 1))
            self.y_train_scaled = pd.Series(self.y_scaler.transform(self.y_train.values.reshape(-1, 1)).flatten())
            self.y_val_scaled = pd.Series(self.y_scaler.transform(self.y_val.values.reshape(-1, 1)).flatten())
            self.scalers['y'] = self.y_scaler
        else:
            self.y_train_scaled, self.y_val_scaled = self.y_train, self.y_val

    
    # 修改点2：RF训练
    def train_rf(self):
        """
        训练 Random Forest，使用 GridSearchCV 寻找最优参数以保证收敛
        """
        param_grid = {
            'n_estimators': [1000],
            'max_depth': [None], #, 10, 20],
            'min_samples_split': [2], # 5],
            'min_samples_leaf': [1], # 2, 4],
            'max_features': [0.2, 0.3, 0.4]
        }

        # 初始化随机森林模型
        rf_model = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
        
        grid = GridSearchCV(rf_model, param_grid, scoring='r2', cv=5)
        grid.fit(self.X_train_scaled, self.y_train_scaled)
        
        self.models['rf'] = grid.best_estimator_
        self.results['rf'] = {'best_params': grid.best_params_}
        print(f"最佳参数: {grid.best_params_}")


    def compute_metrics(self, model_name):
        """
        计算训练集、测试集和外部测试集的 R2
        """
        model = self.models[model_name]
        
        # 定义计算 R2 的闭包
        def get_r2(X_s, y_orig):
            if X_s is None: return np.nan
            preds_s = model.predict(X_s)
            # 逆缩放
            if 'y' in self.scalers:
                preds = self.scalers['y'].inverse_transform(preds_s.reshape(-1, 1)).flatten()
            else:
                preds = preds_s
            return r2_score(y_orig, preds)

        train_r2 = get_r2(self.X_train_scaled, self.y_train)
        val_r2 = get_r2(self.X_val_scaled, self.y_val)
        ext_r2 = get_r2(self.X_ext_test_scaled, self.y_ext_test)

        # 交叉验证 R2 (在训练集上)
        rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=self.random_state)
        cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train_scaled, cv=rkf, scoring='r2')

        self.results[model_name].update({
            'train_r2': train_r2,
            'test_r2': val_r2,
            'external_test_r2': ext_r2,
            'cv_mean_r2': np.mean(cv_scores)
        })
        
        # 打印汇总
        print(f"\n--- {model_name} 结果 ---")
        print(f"CV Mean R2: {np.mean(cv_scores):.4f}")
        # print(f"Train R2: {train_r2:.4f}")
        # print(f"Internal Test R2: {val_r2:.4f}")
        # print(f"External Test R2: {ext_r2:.4f}")
        print("Train R2 | Valid R2 | Test R2")
        print(f"{train_r2:.4f} | {val_r2:.4f} | {ext_r2:.4f}")
        

    
    # 修改点3：保存rf模型
    def save_all_results(self):
        """
        保存结果到文件
        """
        joblib.dump(self.models['rf'], os.path.join(self.output_dir, "rf_model.pkl"))
        res_df = pd.DataFrame(self.results).T
        res_df.to_csv(os.path.join(self.output_dir, "evaluation_summary.csv"))


def process_pipeline(train_csv, test_csv, output_dir):
    """
    具体的执行pipeline
    """
    trainer = TabularModelTrainer(data_path=train_csv, output_dir=output_dir)
    
    # 加载主数据集
    if not trainer.load_data(): return
    
    # 准备数据并实例化 Scaler
    trainer.prepare_data()
    
    # 加载外部测试集（必须在 prepare_data 之后复用 Scaler）
    trainer.load_external_test(test_csv)
    
    # 修改点4：调用RF模型训练和评估
    trainer.train_rf()  # 训练
    trainer.compute_metrics('rf')  # 评估三端 R2
    
    # 保存
    trainer.save_all_results()


import time
if __name__ == "__main__":
    F_NUM = 1
    # 外部测试文件路径
    EXTERNAL_TEST_FILE = f"external_test_F{F_NUM}.csv" 
    # 初始数据集文件夹
    # INPUT_FOLDER = "test"
    INPUT_FOLDER = f"F{F_NUM}"
    # 修改点5：输入输出文件夹名字记得更新（如果有的话）
    OUTPUT_BASE = "10dim_rf_test_func{F_NUM}"

    csv_files = list(Path(INPUT_FOLDER).glob("*.csv"))
    for f in csv_files:
        print(f"\n处理文件: {f.name}")
        out_path = os.path.join(OUTPUT_BASE, f.stem)
        
        start_time = time.time()  # 记录程序开始时间
        
        process_pipeline(str(f), EXTERNAL_TEST_FILE, out_path)
        
        end_time = time.time()  # 记录程序结束时间
        elapsed_time = end_time - start_time  # 计算总耗时
        print(f"\n程序运行总耗时: {elapsed_time:.2f} 秒")

        