# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 21:36:50 2026

@author: 91278
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


# 1. 加载数据
TARGET = 'log10_shear_modulus'
print(f"正在加载数据，准备训练的目标属性是: {TARGET}...")
# file_path = r"G:\jarvis22_featurized_matminer.pkl"
file_path = rf"G:\mp21_{TARGET}_clean.pkl"
df = pd.read_pickle(file_path)

# 2. 根据论文要求进行数据清洗
df_clean = df[df['e_form'] <= 5.0].copy()  # 过滤形成能 > 5 eV/atom 的不稳定材料
print(f"原始数据量: {len(df)}, 过滤后数据量: {len(df_clean)}")

# 3. 指定特征 (X) 和 目标变量 (y)
# 注意：Python 索引从 0 开始
# 第15列到最后是特征 -> 索引为 14 往后
X = df_clean.iloc[:, -273:] 

# 预测性能 (e_form 还是 bandgap)
y = df_clean[TARGET]
print(f"特征矩阵形状: {X.shape}, 目标形状: {y.shape}")


# 4. 划分训练集和测试集 (80% 训练, 20% 测试)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 5. 定义并训练 XGBoost 模型
# 这里设置了一些常用的超参数，你可以根据需要进一步调优
model = xgb.XGBRegressor(
    n_estimators=1000,      # 树的数量
    learning_rate=0.05,     # 学习率
    max_depth=7,            # 树的最大深度
    subsample=0.8,          # 采样比例
    colsample_bytree=0.8,   # 特征采样比例
    n_jobs=4,               # 使用所有CPU核心
    # random_state=42
)
# model = xgb.XGBRegressor(
#     n_estimators=1000,           # estimators 数量
#     num_parallel_tree=4,         # [新增] 提升随机森林：4 parallel boosted trees
#     learning_rate=0.1,           # [修改] 学习率 0.1
#     reg_alpha=0.01,              # [新增] L1 regularization strength
#     reg_lambda=0.1,              # [新增/修改] L2 regularization strength
#     tree_method='hist',          # [新增] histogram tree grow method
#     subsample=0.85,              # [修改] 样本采样比例 0.85
#     colsample_bytree=0.3,        # [修改] 每棵树的特征采样比例 0.3
#     colsample_bylevel=0.5,       # [新增] 树每一次分裂（按层）时的特征采样比例 0.5
#     # max_depth=6,               # 文献未提及，建议注释掉使用默认值 6，或者保留你的尝试
#     n_jobs=1,                   # 使用所有CPU核心
#     random_state=42              # 保证结果可复现
# )

print("\n开始训练模型 (这可能需要几分钟)...")
model.fit(X_train, y_train)

# 6. 模型预测
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 7. 性能评估
def evaluate_model(y_true, y_pred, label="Dataset"):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    print(f"\n--- {label} 性能指标 ---")
    print(f"R² (决定系数): {r2:.4f}")
    print(f"RMSE (均方根误差): {rmse:.4f}")
    print(f"MAE (平均绝对误差): {mae:.4f}")
    return r2, rmse

train_r2, train_rmse = evaluate_model(y_train, y_train_pred, "训练集")
test_r2, test_rmse = evaluate_model(y_test, y_test_pred, "测试集")

# 8. 结果可视化 (预测值 vs 真实值)
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_test_pred, alpha=0.3, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r', linewidth=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title(f'Test Set: {TARGET}')

# 特征重要性排序 (展示前10个)
plt.subplot(1, 2, 2)
xgb.plot_importance(model, max_num_features=10, ax=plt.gca())
plt.title('Top 10 Feature Importance')

plt.tight_layout()
plt.show()
