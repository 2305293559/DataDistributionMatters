# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 04:56:10 2026

@author: 91278
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os
import glob

# ==========================================
# 0. 参数配置区域 (在此修改抽样参数)
# ==========================================
TARGET = 'bulk_modulus'   # 3: e_form, 4: bandgap
N = 200                  # 用于训练的（有偏）子数据集总大小
P_RATIO = 0.0           # Positive 样本比例 (数值偏大的前15%区域)
N_RATIO = 0.4          # Negative 样本比例 (数值偏小的后15%区域)
ITER = 30                  # 统计平均数
Q_VALUE = 0.25           # 极值对应区域的百分比
# 剩余比例 (1 - 0.2 - 0.3 = 0.5) 将用于全局随机抽样

DATA_PATH = r"G:\jarvis22_featurized_matminer.pkl"
DATA_PATH = rf"G:\mp21_{TARGET}_clean.pkl"
EXT_TEST_PATH = "external_test_jarvis.csv"
N_EXTERNAL = 1000               # 外部测试集的大小
SAVE_SAMPLED_FILE = False       # 是否保存构建的子数据集
SAVE_DIST_FILE = True
DIST_RANGE = [0, 400]   # 分布图范围设置 [0, 400]；如果设为 None，则沿用原始自动范围
DIST_BINS = 40


# ============================
# 0. 定义统计分布函数
# ============================
def process_and_save_distribution(data_series, label_name, bins=100, dist_range=None):
    """
    计算 PDF，保存 CSV，并返回绘图所需的 x, y
    
    参数
    ----
    data_series : pandas.Series 或 array-like
        要统计分布的数据
    label_name : str
        输出文件标签
    bins : int
        直方图分箱数
    dist_range : list/tuple/None
        例如 [0, 400] 或 (0, 400)
        - 若为 None，则使用数据本身范围（原逻辑）
        - 若指定范围，则所有统计统一使用该固定范围
    """
    # 转成 numpy array，避免潜在问题
    data_array = np.asarray(data_series.dropna())

    # 固定范围统计 / 自动范围统计
    if dist_range is None:
        hist, bin_edges = np.histogram(data_array, bins=bins, density=True)
    else:
        hist, bin_edges = np.histogram(data_array, bins=bins, range=tuple(dist_range), density=True)

    # 区间中心点
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    
    # 保存 CSV
    if SAVE_DIST_FILE:
        dist_df = pd.DataFrame({
            f'{TARGET}_Value': bin_centers,
            'PDF_Probability': hist
        })
        filename = f"distribution_{label_name}.csv"
        dist_df.to_csv(filename, index=False)
        # print(f"   -> 已保存分布文件: {filename}")
        
    return bin_centers, hist


def merge_and_clean_distributions(base_name):
    
    output_filename = f"distribution_{base_name}.csv"
    # print(f"\n--- 开始合并分布文件: {base_name} ---")
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
                first_df = df.copy() # 保留第一份数据的X轴结构
                pdf_sum = df.iloc[:, 1].values   # 第二列是 PDF_Probability
            else:
                pdf_sum += df.iloc[:, 1].values
                
        except Exception as e:
            print(f"读取文件 {filename} 时发生错误: {e}")
            return
    
    pdf_avg = pdf_sum / len(temp_files)

    # 使用第一个文件的结构，仅替换 PDF 列的值
    result_df = first_df.copy()
    result_df.iloc[:, 1] = pdf_avg

    result_df.to_csv(output_filename, index=False)
    print(f"✅ 平均分布文件已生成: {output_filename}")
    for f in temp_files:
        os.remove(f)  # 删除文件

# ==========================================
# 1. 加载与清洗原始数据
# ==========================================
print("Step 1: 正在加载原始数据...")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"找不到文件: {DATA_PATH}")

df = pd.read_pickle(DATA_PATH)

print(f"目标预测属性: {TARGET}")

# 数据清洗：过滤形成能 > 5 eV/atom
df_clean = df[df['e_form'] <= 5.0].copy()
print(f"原始数据: {len(df)} -> 清洗后数据: {len(df_clean)}")

# 保存原始数据分布
orig_x, orig_y = process_and_save_distribution(
    df_clean[TARGET],
    "Original_Full",
    bins=DIST_BINS,
    dist_range=DIST_RANGE
)


# ==========================================
# 1.5 构建独立的 External Test Set
# ==========================================

print(f"\nStep 1.5: 抽取并剔除 {N_EXTERNAL} 条数据作为独立外部测试集...")

if len(df_clean) < N_EXTERNAL + N:
    raise ValueError(f"数据量不足！可用 {len(df_clean)}，但需要 External({N_EXTERNAL}) + Train({N})")
# 随机抽取 External Set
# 注意：这里设定固定种子42，保证无论后续训练参数怎么改，外部测试集永远是这一批，方便横向对比
df_external = df_clean.sample(n=N_EXTERNAL, replace=False, random_state=42)
# 保存 External Set
df_external.to_csv(EXT_TEST_PATH)
print(f"   -> 外部测试集已保存至: {EXT_TEST_PATH}")
# 从总池中彻底剔除这些数据，生成真正的 df_clean (训练池)
df_clean = df_clean.drop(df_external.index)
print(f"   -> 剩余可用训练池 (df_clean): {len(df_clean)} 条")


# ==========================================
# 2. 实现有偏抽样算法 (Biased Sampling)
# ==========================================
    
# 2.1 排序
df_sorted = df_clean.sort_values(by=TARGET, ascending=True)

# 2.2 定义区域 (前15% 和 后15%)
threshold_count = int(len(df_sorted) * Q_VALUE)
pool_positive = df_sorted.iloc[-threshold_count:]  # 数值最大的前15% 
pool_negative = df_sorted.iloc[:threshold_count] # 数值最小的后15%
print("正样本池大小：", len(pool_positive), " 负样本池大小：", len(pool_negative))


# for N in [1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000, 60000]:
# for N in [100, 200, 500, 1000, 2000, 4000, 5000, 6000]:
for N in [1000]:
    print(f"\nStep 2: 执行有偏抽样 (N={N}, P={P_RATIO}, N={N_RATIO})...")
    
    # 2.3 计算各部分抽样数量
    count_pos = int(N * P_RATIO)
    count_neg = int(N * N_RATIO)
    count_rand = N - count_pos - count_neg
    
    print(f"   -> 计划抽取: Positive(前15%)={count_pos}, Negative(后15%)={count_neg}, Random(全局)={count_rand}")

    print(" Train | Val | Test ")
    for icc in range(ITER):
    
        # 2.4 执行抽样 (replace=False 表示不重复抽取，如果N很大可能需要改为True)
        # 注意：随机部分是从整个清洗后的池子中抽，可能会与Pos/Neg重叠，这是符合"全局随机"定义的
        sample_pos = pool_positive.sample(n=count_pos, replace=True)
        sample_neg = pool_negative.sample(n=count_neg, replace=True)
        sample_rand = df_clean.sample(n=count_rand, replace=False) # 不同的随机种子
        
        # 2.5 合并数据
        df_sampled = pd.concat([sample_pos, sample_neg, sample_rand])
        
        # 去重逻辑：如果不想让随机抽样抽到已有的Pos/Neg数据，可以执行去重。
        # 但通常混合抽样允许为了保持分布比例而存在少量重叠。这里我们简单保留所有行供训练。
        # 如果想严格去重索引：
        # df_sampled = df_sampled[~df_sampled.index.duplicated(keep='first')]
        # 如果去重后数量不足N，这是正常的，为了严谨这里不做额外填充。
        # print(f"   -> 实际构建数据集大小: {len(df_sampled)}")
        
        # 2.6 保存抽样数据集
        if SAVE_SAMPLED_FILE:
            csv_name = f"{N}_P{P_RATIO}+N{N_RATIO}.csv"
            df_sampled.to_csv(csv_name)
            print(f"   -> 数据集已保存至: {csv_name}")
        
        # 2.7 保存分布数据集并画图
        # 获取原始数据和采样数据的分布数据    
        samp_x, samp_y = process_and_save_distribution(
            df_sampled[TARGET],
            f"{N}_P{P_RATIO}+N{N_RATIO}_{icc}",
            bins=DIST_BINS,
            dist_range=DIST_RANGE
        )
    
        plt.figure(figsize=(10, 6))
        # 绘制原始分布
        plt.plot(orig_x, orig_y, 'g--', linewidth=2, label='Original (Full Dataset)')
        plt.fill_between(orig_x, orig_y, alpha=0.1, color='green')
        
        # 绘制采样分布
        plt.plot(samp_x, samp_y, 'r-', linewidth=2, label=f'Sampled (N={N}, P={P_RATIO}, N={N_RATIO})')
        plt.fill_between(samp_x, samp_y, alpha=0.1, color='red')
        
        plt.title(f'PDF Distribution Comparison: {TARGET}')
        plt.xlabel(f'{TARGET} Value')
        plt.ylabel('Probability Density Function (PDF)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.show()
        
        """
        # ==========================================
        # 3. 准备训练数据 (X, y)
        # ==========================================
        # 获取后273列作为特征
        X = df_sampled.iloc[:, -273:]
        y = df_sampled[TARGET]
        
        # 划分内部训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # ==========================================
        # 4. 定义并训练模型
        # ==========================================
        
        # model = xgb.XGBRegressor(
        #     n_estimators=1000,
        #     learning_rate=0.05,
        #     max_depth=7,
        #     subsample=0.8,
        #     colsample_bytree=0.8,
        #     n_jobs=4,
        #     random_state=42
        # )
        model = xgb.XGBRegressor(
            n_estimators=1000,           # estimators 数量
            num_parallel_tree=4,         # [新增] 提升随机森林：4 parallel boosted trees
            learning_rate=0.1,           # [修改] 学习率 0.1
            reg_alpha=0.01,              # [新增] L1 regularization strength
            reg_lambda=0.1,              # [新增/修改] L2 regularization strength
            tree_method='hist',          # [新增] histogram tree grow method
            subsample=0.85,              # [修改] 样本采样比例 0.85
            colsample_bytree=0.3,        # [修改] 每棵树的特征采样比例 0.3
            colsample_bylevel=0.5,       # [新增] 树每一次分裂（按层）时的特征采样比例 0.5
            # max_depth=6,               # 文献未提及，建议注释掉使用默认值 6，或者保留你的尝试
            n_jobs=4,                   # 使用所有CPU核心
            random_state=42              # 保证结果可复现
        )
        
        model.fit(X_train, y_train)
        
        # ==========================================
        # 5. 外部 OOD 测试集处理
        # ==========================================
        # print("\nStep 4: 加载外部测试集进行 OOD 测试...")
        external_r2 = float('nan') # 初始化
        
        if os.path.exists(EXT_TEST_PATH):
            # 读取外部 CSV
            # 假设 external csv 第一列是索引 (material_id)
            df_ext = pd.read_csv(EXT_TEST_PATH, index_col=0) 
            
            # 确保列名对齐，或者使用切片
            # 警告：CSV读取后列的顺序必须与 pickle 一致。
            # 这里假设 external csv 是用之前的脚本生成的，结构一致。
            
            try:
                # 获取特征 X_ext
                # 如果保存时包含了所有列，用 iloc 切片后273列
                X_ext = df_ext.iloc[:, -273:]
                
                # 获取目标 y_ext
                # 使用列名定位更安全，因为 CSV 保存后索引位置可能变化 (如果包含index列的话)
                if TARGET in df_ext.columns:
                    y_ext = df_ext[TARGET]
                else:
                    # 如果列名丢失，尝试用索引位置（假设结构与训练集一致）
                    print("列名不正确！")
                    y_ext = df_ext[TARGET]
                    
                # print(f"   -> 外部测试集大小: {len(df_ext)}")
                
                # 预测
                y_ext_pred = model.predict(X_ext)
                # -- 计算外部验证集的R2 MAE 和 RMSE --
                external_r2 = r2_score(y_ext, y_ext_pred)                
                external_mae = mean_absolute_error(y_ext, y_ext_pred)
                external_rmse = np.sqrt(mean_squared_error(y_ext, y_ext_pred))
                
            except Exception as e:
                print(f"   -> 处理外部测试集时出错: {e}")
        else:
            print(f"   -> 警告: 找不到外部测试文件 {EXT_TEST_PATH}")
        
        # ==========================================
        # 6. 最终综合性能报告
        # ==========================================
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # --- 计算 R² ---
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        # --- 计算 MAE ---
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        # --- 计算 RMSE ---
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        # 打印报告
        # print("\n" + "="*10)
        print(f"{external_r2:.4f} | {external_rmse:.4f} | {external_mae:.4f} | {test_r2:.4f} | {test_rmse:.4f} | {test_mae:.4f} | {train_r2:.4f} | {train_rmse:.4f} | {train_mae:.4f}")
        # 简单绘图
        # plt.figure(figsize=(8, 6))
        # plt.scatter(y_test, y_test_pred, alpha=0.3, label='Internal Test')
        # if not np.isnan(external_r2):
        #     plt.scatter(y_ext, y_ext_pred, alpha=0.3, color='red', label='External OOD')
        # plt.plot([y.min(), y.max()], [y.min(), y.max()], '--k')
        # plt.xlabel(f'True {TARGET}')
        # plt.ylabel(f'Predicted {TARGET}')
        # plt.legend()
        # plt.title(f'Prediction Performance (Ext R2={external_r2:.3f})')
        # plt.show()
    
    """
    
    # 分布文件统计平均
    merge_and_clean_distributions(f"{N}_P{P_RATIO}+N{N_RATIO}")


