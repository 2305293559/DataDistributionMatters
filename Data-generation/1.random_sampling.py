# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 04:46:00 2026
从数据库中抽取均匀独立测试集（作为OOD）
@author: 91278
"""
import pandas as pd

# 1. 加载原始数据
print("正在加载原始数据 (mp21_featurized_matminer.pkl)...")
# file_path = r'G:\mp21_featurized_matminer.pkl'
file_path = r"G:\mp21_bulk_modulus_clean.pkl"
df = pd.read_pickle(file_path)
print(f"数据量: {len(df)}")

# 2. 提取指定性能的非空条目
target_col = 'log10_shear_modulus' # 你想要学习的性能目标
print(f"\n正在提取 {target_col} 列非空的数据行...")

# 使用 dropna 剔除在该列包含 NaN 的所有行
# subset 参数指明只看 target_col 这一列是否为空
df = df.dropna(subset=[target_col]).copy() # 加 .copy() 避免由于切片引起的警告

print(f"清洗完成！{target_col} 有效数据剩余量: {len(df)}")

# 将非空的数据保存为一个全新的 pkl 文件，方便日后直接从这里读取，不破坏原库
clean_file_path = rf'G:\mp21_{target_col}_clean.pkl'
df.to_pickle(clean_file_path)
print(f"✅ 已将有效数据提取并另存为新文件: {clean_file_path}\n")


# 3. 均匀随机抽样
sample_size = 1000

# 检查数据量是否足够
if len(df) < sample_size:
    print(f"错误：可用数据量 ({len(df)}) 小于请求的样本量 ({sample_size})。")
else:
    print(f"正在随机抽取 {sample_size} 条数据...")
    
    # random_state 设置固定数字(如42)可保证每次运行代码抽出的数据是一样的
    # 如果希望每次都不一样，可以删掉 random_state=42
    sample_df = df.sample(n=sample_size, random_state=42)

    # 4. 保存为 CSV 文件
    output_filename = 'external_test_jarvis.csv'
    print(f"正在保存文件到 {output_filename} ...")
    
    # index=True 建议保留，因为索引通常是 material_id，方便后续追溯
    sample_df.to_csv(output_filename, index=True) 
    
    print("✅ 抽样完成！文件已保存。")
    
    # 打印前几行看看效果
    print("\n--- 抽样数据预览 ---")
    print(sample_df.iloc[:5, :8]) # 只打印前8列预览
    