import os
import pandas as pd
import numpy as np

# ===== 1. 设置你的 CSV 文件夹路径 =====
input_dir = "data_summary"   # 改成你的文件夹路径
output_dir = os.path.join(input_dir, "processed_test_r2")
os.makedirs(output_dir, exist_ok=True)

# ===== 2. 遍历文件夹中的所有 CSV =====
for file_name in os.listdir(input_dir):
    if not file_name.endswith(".csv"):
        continue

    file_path = os.path.join(input_dir, file_name)
    print(f"Processing: {file_name}")

    # ===== 3. 读取 CSV =====
    df = pd.read_csv(file_path)

    # ===== 4. 透视表：行 n_samples，列 run_id，值 external_test_r2 =====
    pivot_df = df.pivot(
        index="n_samples",
        columns="run_id",
        values="test_r2"
    )

    # ===== 5. 小于 0 的值置为 0 =====
    pivot_df = pivot_df.clip(lower=0)

    # ===== 6. 计算统计量 =====
    # 普通平均
    pivot_df["mean"] = pivot_df.mean(axis=1)

    # 去掉最大最小值后的平均
    def trimmed_mean(row):
        values = row.dropna().values
        if len(values) <= 2:
            return np.nan
        return np.mean(np.sort(values)[1:-1])

    pivot_df["trimmed_mean"] = pivot_df.iloc[:, :-1].apply(trimmed_mean, axis=1)

    # 标准差
    pivot_df["std"] = pivot_df.iloc[:, :-2].std(axis=1)

    # ===== 7. 保存结果 =====
    output_path = os.path.join(
        output_dir,
        file_name.replace(".csv", "_processed.csv")
    )
    pivot_df.to_csv(output_path)

print("All files processed.")
