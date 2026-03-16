import os
import pandas as pd
import numpy as np
from collections import defaultdict
import re


def parse_filename_from_content(filename_str):
    """
    从文件内容中的Filename字段解析采样方法和样本量
    格式示例：10dim_active_learning_1000_fun1.csv
    返回：('active_learning', 1000)
    """
    if not isinstance(filename_str, str):
        return None, None

    name = filename_str.replace('.csv', '')
    pattern = r'10dim_([a-zA-Z_]+)_(\d+)_fun1$'
    match = re.search(pattern, name)

    if match:
        return match.group(1), int(match.group(2))

    return None, None


def aggregate_metrics(directory_path, output_file='aggregated_metrics.csv'):
    """
    聚合指定目录下所有CSV文件的指标数据
    """

    data_dict = defaultdict(lambda: defaultdict(list))
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

    if not csv_files:
        print(f"目录 {directory_path} 中没有找到CSV文件")
        return None

    print(f"找到 {len(csv_files)} 个CSV文件")
    total_rows_processed = 0

    for i, csv_file in enumerate(csv_files, start=1):
        file_path = os.path.join(directory_path, csv_file)

        try:
            df = pd.read_csv(file_path)

            if 'Filename' not in df.columns:
                print(f"警告: 文件 {csv_file} 中缺少 Filename 列，已跳过")
                continue

            for _, row in df.iterrows():
                filename = row['Filename']
                if pd.isna(filename):
                    continue

                sampling_method, sample_size = parse_filename_from_content(filename)
                if sampling_method is None:
                    print(f"警告: 无法解析 Filename: {filename}")
                    continue

                # ✅ 补充：Sample_Size 一致性检查
                if 'Sample_Size' in df.columns and not pd.isna(row['Sample_Size']):
                    if int(row['Sample_Size']) != sample_size:
                        print(
                            f"警告: Sample_Size 不一致: "
                            f"{row['Sample_Size']} vs {sample_size} ({filename})"
                        )

                data_dict[sampling_method][sample_size].append(row)
                total_rows_processed += 1

            if i % 10 == 0:
                print(f"已处理 {i}/{len(csv_files)} 个文件")

        except Exception as e:
            print(f"处理文件 {csv_file} 时出错: {e}")

    print(f"\n数据处理完成，共处理 {total_rows_processed} 行数据，开始计算平均值...")

    aggregated_data = []

    for sampling_method, size_dict in data_dict.items():
        for sample_size, rows in size_dict.items():
            if not rows:
                continue

            df_group = pd.DataFrame(rows)
            row_count = len(df_group)

            print(f"{sampling_method}_{sample_size}: {row_count} 行数据")

            # ✅ 补充：明确排除 Sample_Size
            exclude_cols = ['Filename', 'Sample_Size']
            numeric_cols = [
                col for col in df_group.columns
                if col not in exclude_cols and
                pd.api.types.is_numeric_dtype(df_group[col])
            ]

            if not numeric_cols:
                print(f"警告: {sampling_method}_{sample_size} 没有数值列")
                continue

            avg_metrics = df_group[numeric_cols].mean()

            result_row = {
                'Sampling_Method': sampling_method,
                'Sample_Size': sample_size,
                'Row_Count': row_count
            }

            for col in numeric_cols:
                result_row[f'{col}_avg'] = avg_metrics[col]

            aggregated_data.append(result_row)

    if not aggregated_data:
        print("没有数据可以汇总")
        return None

    result_df = pd.DataFrame(aggregated_data)
    result_df = result_df.sort_values(['Sampling_Method', 'Sample_Size'])
    result_df.to_csv(output_file, index=False)

    print(f"\n结果已保存到: {output_file}")

    print("\n汇总统计:")
    print(f"共处理了 {len(csv_files)} 个文件")
    print(f"汇总了 {len(result_df)} 个采样方法-样本量组合")

    method_stats = result_df.groupby('Sampling_Method')['Row_Count'].sum()
    print("\n各采样方法数据行数统计:")
    for method, count in method_stats.items():
        print(f"  {method}: {count} 行")

    return result_df


def print_summary(result_df):
    if result_df is None:
        print("没有可显示的汇总结果")
        return

    print("\n汇总结果预览:")
    print(result_df.head())

    print("\n所有采样方法:")
    print(result_df['Sampling_Method'].unique())


if __name__ == "__main__":
    directory_path = "metrics_results_50_f3_Michalewicz"

    if not os.path.exists(directory_path):
        print(f"目录 {directory_path} 不存在，请检查路径")
    else:
        result_df = aggregate_metrics(
            directory_path,
            output_file='aggregated_metrics_50_files_f3.csv'
        )

        print_summary(result_df)
