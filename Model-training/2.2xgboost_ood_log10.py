# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 03:22:12 2026

@author: 91278
"""
# -*- coding: utf-8 -*-
"""
@author: Default
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os

# ==========================================
# 0. 参数配置区域 (在此修改抽样参数)
# ==========================================
# 🎯 指定目标为 log10 版本，保证数据读取文件路径正确并使用对数训练
TARGET = 'log10_shear_modulus'   
TARGET_REAL_NAME = 'shear_modulus'  # 用于图表和报告表头显示真实属性名

P_RATIO = 0.25           # Positive 样本比例 
N_RATIO = 0.25         # Negative 样本比例 
ITER = 5               # 统计平均数
Q_VALUE = 0.2         # 极值对应区域的百分比

DATA_PATH = rf"G:\mp21_{TARGET}_clean.pkl"
EXT_TEST_PATH = "external_test_jarvisN.csv"
N_EXTERNAL = 940               
SAVE_SAMPLED_FILE = False       
SAVE_DIST_FILE = True

# ============================
# 0. 定义统计分布函数 (完全在真实物理空间操作)
# ============================
def process_and_save_distribution(data_series, label_name, bins=100):
    """
    计算 PDF，保存 CSV，并返回绘图所需的 x, y
    注意：传入的 data_series 将确保已经是去除了 log 的真实数值。
    """
    hist, bin_edges = np.histogram(data_series, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    if SAVE_DIST_FILE:
        dist_df = pd.DataFrame({
            f'{TARGET_REAL_NAME}_Value': bin_centers,  # 确保 CSV 表头是真实名字
            'PDF_Probability': hist
        })
        filename = f"distribution_{label_name}.csv"
        dist_df.to_csv(filename, index=False)
        
    return bin_centers, hist

def merge_and_clean_distributions(base_name):
    output_filename = f"distribution_{base_name}.csv"
    temp_files = []
    pdf_sum = None
    first_df = None

    for icc in range(ITER):
        filename = f"distribution_{base_name}_{icc}.csv"
        if not os.path.exists(filename):
            print(f"错误: 找不到文件 {filename}，合并终止。")
            return
        try:
            df = pd.read_csv(filename)
            temp_files.append(filename)

            if pdf_sum is None:
                first_df = df.copy() 
                pdf_sum = df.iloc[:, 1].values   
            else:
                pdf_sum += df.iloc[:, 1].values
                
        except Exception as e:
            print(f"读取文件 {filename} 时发生错误: {e}")
            return
    
    pdf_avg = pdf_sum / len(temp_files)
    result_df = first_df.copy()
    result_df.iloc[:, 1] = pdf_avg
    result_df.to_csv(output_filename, index=False)
    print(f"✅ 平均分布文件已生成: {output_filename}")
    for f in temp_files:
        os.remove(f)  

# ==========================================
# 1. 加载与清洗原始数据
# ==========================================
print("Step 1: 正在加载原始数据...")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"找不到文件: {DATA_PATH}")

df = pd.read_pickle(DATA_PATH)
print(f"机器学习训练属性: {TARGET}, 报告还原属性: {TARGET_REAL_NAME}")

df_clean = df[df['e_form'] <= 5.0].copy()
print(f"原始数据: {len(df)} -> 清洗后数据: {len(df_clean)}")

# 🎯 将真实的 shear modulus 送入分布统计算法 (10 的对数次幂)
orig_x, orig_y = process_and_save_distribution(10 ** df_clean[TARGET], "Original_Full")

# ==========================================
# 1.5 构建独立的 External Test Set
# ==========================================
print(f"\nStep 1.5: 抽取并剔除 {N_EXTERNAL} 条数据作为独立外部测试集...")
if len(df_clean) < N_EXTERNAL:
    raise ValueError("数据量不足")

df_external = df_clean.sample(n=N_EXTERNAL, replace=False, random_state=42)
df_external.to_csv(EXT_TEST_PATH)
print(f"   -> 外部测试集已保存至: {EXT_TEST_PATH}")
df_clean = df_clean.drop(df_external.index)
print(f"   -> 剩余可用训练池 (df_clean): {len(df_clean)} 条")

# ==========================================
# 2. 实现有偏抽样算法 (Biased Sampling)
# ==========================================
df_sorted = df_clean.sort_values(by=TARGET, ascending=True)

threshold_count = int(len(df_sorted) * Q_VALUE)
pool_positive = df_sorted.iloc[:threshold_count]  
pool_negative = df_sorted.iloc[-threshold_count:] 

for N in [100, 200, 500, 1000, 2000, 4000, 5000, 6000]:
# for N in [6000]:
    print(f"\nStep 2: 执行有偏抽样 (N={N}, P={P_RATIO}, N={N_RATIO})...")
    
    count_pos = int(N * P_RATIO)
    count_neg = int(N * N_RATIO)
    count_rand = N - count_pos - count_neg
    
    print(" Train | Val | Test ")
    for icc in range(ITER): # 使用配置好的 ITER 循环次数
    
        sample_pos = pool_positive.sample(n=count_pos, replace=True)
        sample_neg = pool_negative.sample(n=count_neg, replace=True)
        sample_rand = df_clean.sample(n=count_rand, replace=False) 
        
        df_sampled = pd.concat([sample_pos, sample_neg, sample_rand])
        
        if SAVE_SAMPLED_FILE:
            csv_name = f"{N}_P{P_RATIO}+N{N_RATIO}.csv"
            df_sampled.to_csv(csv_name)
        
        # 🎯 保存分布数据集并画图 (同样将其转换为真实模量) 
        samp_x, samp_y = process_and_save_distribution(10 ** df_sampled[TARGET], f"{N}_P{P_RATIO}+N{N_RATIO}_{icc}")
    
        plt.figure(figsize=(10, 6))
        plt.plot(orig_x, orig_y, 'g--', linewidth=2, label='Original (Full Dataset)')
        plt.fill_between(orig_x, orig_y, alpha=0.1, color='green')
        plt.plot(samp_x, samp_y, 'r-', linewidth=2, label=f'Sampled (N={N}, P={P_RATIO}, N={N_RATIO})')
        plt.fill_between(samp_x, samp_y, alpha=0.1, color='red')
        
        plt.title(f'PDF Distribution Comparison: {TARGET_REAL_NAME} (Real Scale)')
        plt.xlabel(f'{TARGET_REAL_NAME} Value (GPa)')  # 明确标注单位是 GPa
        plt.ylabel('Probability Density Function (PDF)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # ==========================================
        # 3. 准备在 Log 空间训练特征序列
        # ==========================================
        X = df_sampled.iloc[:, -273:]
        y = df_sampled[TARGET] # 此时 y 是 Log10 值
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # ==========================================
        # 4. 定义并训练模型
        # ==========================================
        model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=4,
            random_state=42
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
        #     n_jobs=4,                   # 使用所有CPU核心
        #     random_state=42              # 保证结果可复现
        # )
        model.fit(X_train, y_train)
        
        # ==========================================
        # 5. 外部 OOD 测试集处理与真实空间还原
        # ==========================================
        external_r2 = float('nan') 
        external_rmse = float('nan')
        external_mae = float('nan')
        
        if os.path.exists(EXT_TEST_PATH):
            df_ext = pd.read_csv(EXT_TEST_PATH, index_col=0) 
            try:
                X_ext = df_ext.iloc[:, -273:]
                y_ext_log = df_ext[TARGET]
                
                # 预测输出依然是 Log 空间
                y_ext_pred_log = model.predict(X_ext)
                
                # 🎯 进行 10 的幂运算反推，还原为真实的剪切模量
                y_ext_real = 10 ** y_ext_log
                y_ext_pred_real = 10 ** y_ext_pred_log
                
                # 计算各种指标！
                external_r2 = r2_score(y_ext_real, y_ext_pred_real)                
                external_mae = mean_absolute_error(y_ext_real, y_ext_pred_real)
                external_rmse = np.sqrt(mean_squared_error(y_ext_real, y_ext_pred_real))
            except Exception as e:
                print(f"   -> 处理外部测试集时出错: {e}")
        
        # ==========================================
        # 6. 综合性能报告 (针对 Train 和 Test)
        # ==========================================
        y_train_pred_log = model.predict(X_train)
        y_test_pred_log = model.predict(X_test)
        
        # 🎯 同样，内部的数据还原真实特征计算
        y_train_real = 10 ** y_train
        y_test_real = 10 ** y_test
        y_train_pred_real = 10 ** y_train_pred_log
        y_test_pred_real = 10 ** y_test_pred_log
        
        train_r2 = r2_score(y_train_real, y_train_pred_real)
        test_r2 = r2_score(y_test_real, y_test_pred_real)
        train_mae = mean_absolute_error(y_train_real, y_train_pred_real)
        test_mae = mean_absolute_error(y_test_real, y_test_pred_real)
        train_rmse = np.sqrt(mean_squared_error(y_train_real, y_train_pred_real))
        test_rmse = np.sqrt(mean_squared_error(y_test_real, y_test_pred_real))

        # 打印综合表现表格！
        print(f"{external_r2:.4f} | {external_rmse:.4f} | {external_mae:.4f} | {test_r2:.4f} | {test_rmse:.4f} | {test_mae:.4f} | {train_r2:.4f} | {train_rmse:.4f} | {train_mae:.4f}")
        
        # # 🎯 画真实状态（Real Scale）的预测对比图
        # plt.figure(figsize=(8, 6))
        # plt.scatter(y_test_real, y_test_pred_real, alpha=0.3, label='Internal Test')
        # if not np.isnan(external_r2):
        #     plt.scatter(y_ext_real, y_ext_pred_real, alpha=0.3, color='red', label='External OOD')
            
        # # 根据动态极值铺设理想状态 y=x 的对齐基线
        # all_real = np.concatenate([y_test_real, y_test_pred_real, (y_ext_real if not np.isnan(external_r2) else np.array([]))])
        # min_val, max_val = all_real.min(), all_real.max()
        # plt.plot([min_val, max_val], [min_val, max_val], '--k')
        
        # plt.xlabel(f'True {TARGET_REAL_NAME} (GPa)')
        # plt.ylabel(f'Predicted {TARGET_REAL_NAME} (GPa)')
        # plt.legend()
        # plt.title(f'Prediction in Real Scale (Ext Real R2={external_r2:.3f})')
        # plt.show()
    
    # 按照 N 样本数量聚合取平均分布
    merge_and_clean_distributions(f"{N}_P{P_RATIO}+N{N_RATIO}")