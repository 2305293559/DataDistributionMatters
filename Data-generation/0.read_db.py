# -*- coding: utf-8 -*-
"""
Created on Sat Jan  3 18:03:52 2026

@author: 91278
"""
import pandas as pd

# 1. 读取 Pickle 文件
print("正在加载数据...")
# file_path = r"G:\mp21_featurized_matminer.pkl"
file_path = r"G:\mp21_bulk_modulus_clean.pkl"
df = pd.read_pickle(file_path)

# 2. 查看基本信息
print("\n--- 数据集形状 (行数, 列数) ---")
print(df.shape)

print("\n--- 数据集前 5 行预览 ---")
# 设置显示选项，防止列过多被折叠
pd.set_option('display.max_columns', 20)
print(df.head())

print("\n--- 列名信息 ---")
# 查看所有列的名称，了解特征和标签都在哪
print(df.columns.tolist())

print("\n--- 数据类型概览 ---")
print(df.info())

# 将前 10 行保存为 CSV，这样可以用 Excel 打开
df.head(10).to_csv('jarvis_sample.csv')
print("已导出 jarvis_sample.csv，请用 Excel 打开查看。")

# =====================
# 提取其中部分变量和性能
# =====================
# 1. 统计所有性能列的有效非空数据量
property_cols = df.columns[:-273]
print("\n--- 数据库中包含的非特征（性能/标签/元数据）列有哪些 ---")
print(property_cols.tolist())
print("\n--- 每列的有效数据量 (非 NaN 的行数) ---")
# 选取 property_cols 对应的子 dataframe，并用 .count() 统计有效数字
valid_counts = df[property_cols].count()
print(valid_counts)

# 2. 提取性质 (Y)
target_col = 'log10_shear_modulus' # 或者是 'bulk_modulus', 'e_form'
y = df[target_col]

# 3. 提取特征 (X)
# 最后 273 列是 matminer 特征
X = df.iloc[:, -273:]

print(f"特征矩阵 X 形状: {X.shape}")
print(f"目标向量 y 形状: {y.shape}")

# 检查一下特征列名，确保看起来像物理特征
print("部分特征列名:", X.columns[:5].tolist())

