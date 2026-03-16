# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 03:20:50 2026

@author: 91278
"""
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# 1. 加载数据
TARGET = 'log10_shear_modulus'
print(f"正在加载数据，准备训练的目标属性是: {TARGET}...")
file_path = rf"G:\mp21_{TARGET}_clean.pkl"
df = pd.read_pickle(file_path)

# 2. 根据论文要求进行数据清洗
df_clean = df[df['e_form'] <= 5.0].copy()  
print(f"原始数据量: {len(df)}, 过滤后数据量: {len(df_clean)}")

# 3. 指定特征 (X) 和 目标变量 (y)
X = df_clean.iloc[:, -273:] 
y = df_clean[TARGET]  # 目标依旧是对数值
print(f"特征矩阵形状: {X.shape}, 目标形状: {y.shape}")

# 4. 划分训练集和测试集 (80% 训练, 20% 测试)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 5. 定义并训练 XGBoost 模型
model = xgb.XGBRegressor(
    n_estimators=1000,      
    learning_rate=0.05,     
    max_depth=7,            
    subsample=0.8,          
    colsample_bytree=0.8,   
    n_jobs=4,              # 设置为 -1 可以自动使用所有 CPU 核心
    # random_state=42         # 锁定随机种子，保证每次运行结果一致
)

print("\n开始在 Log10 空间训练模型 (这可能需要几分钟)...")
model.fit(X_train, y_train)

# 6. 模型预测 (这里预测出来的还是 Log10 数值)
y_train_pred_log = model.predict(X_train)
y_test_pred_log = model.predict(X_test)

# ==========================================
# 7. 性能评估 (核心修改点：反变换回真实物理空间)
# ==========================================
def evaluate_model(y_true_log, y_pred_log, label="Dataset"):
    # 【核心操作】将对数值映射回真实的剪切模量 (GPa)
    y_true_real = 10 ** y_true_log
    y_pred_real = 10 ** y_pred_log
    
    # 在真实空间下计算指标
    r2_real = r2_score(y_true_real, y_pred_real)
    rmse_real = np.sqrt(mean_squared_error(y_true_real, y_pred_real))
    mae_real = mean_absolute_error(y_true_real, y_pred_real)
    
    # 作为对比，保留 Log 空间下的 R2
    r2_log = r2_score(y_true_log, y_pred_log)
    
    print(f"\n--- {label} 性能指标 ---")
    print(f"R² (Log10空间拟合度): {r2_log:.4f}  <-- 这是模型真实抓取规律的能力")
    print(f"R² (真实物理空间)    : {r2_real:.4f}  <-- 受极大异常值影响较大")
    print(f"RMSE (真实物理空间)  : {rmse_real:.4f} GPa")
    print(f"MAE  (真实物理空间)  : {mae_real:.4f} GPa")
    
    # 将真实值返回，给后续画图使用
    return y_true_real, y_pred_real

y_train_real, y_train_pred_real = evaluate_model(y_train, y_train_pred_log, "训练集")
y_test_real, y_test_pred_real = evaluate_model(y_test, y_test_pred_log, "测试集")

# ==========================================
# 8. 结果可视化 (在真实单位空间展示)
# ==========================================
plt.figure(figsize=(12, 5))

# 图1: 测试集真实值拟合图
plt.subplot(1, 2, 1)
# 我们用还原后的真实值来画图
plt.scatter(y_test_real, y_test_pred_real, alpha=0.3, color='blue', edgecolor='k')

# 动态获取当前数轴极值，画出 y=x 的标准虚线
min_val = min(y_test_real.min(), y_test_pred_real.min())
max_val = max(y_test_real.max(), y_test_pred_real.max())
plt.plot([min_val, max_val], [min_val, max_val], '--r', linewidth=2)

plt.xlabel('Actual Shear Modulus (GPa)')
plt.ylabel('Predicted Shear Modulus (GPa)')
plt.title('Test Set: Real Scale (GPa)')

# 【额外技巧】如果超硬材料把图表拉得太长，你可以取消下面两行的注释，将轴改为对数显示，可视化更好看
# plt.xscale('log')
# plt.yscale('log')

# 图2: 特征重要性排序
plt.subplot(1, 2, 2)
# importance_type 默认为 'weight'，你也可以按需改成 'gain' (增益)，通常 'gain' 更有物理意义
xgb.plot_importance(model, max_num_features=10, importance_type='gain', ax=plt.gca(), xlabel='Gain')
plt.title('Top 10 Feature Importance (Gain)')

plt.tight_layout()
plt.show()