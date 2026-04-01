# -*- coding: utf-8 -*-
"""
1.1gen_random_testset.py
功能：
1. 生成随机样本数据集
2. 支持自定义目标函数
3. 支持指定输出目录和输出文件名
"""

import os
import numpy as np
import pandas as pd
from typing import Callable, Optional, Dict, Tuple

DomainType = Dict[str, Tuple[float, float]]


def generate_random_samples(
    n_samples: int = 100,
    domain: Optional[DomainType] = None,
    output_dir: str = ".",
    output_filename: str = "external_test.csv",
    objective_function: Optional[Callable[[np.ndarray], float]] = None
) -> str:
    """
    在给定定义域内随机采样并生成数据集
    并保存到指定目录和文件名

    参数：
        n_samples: int, 样本数量
        domain: dict, 定义域，例如 {"x1": (-5,5), ...}
        output_dir: str, 保存目录
        output_filename: str, 输出 CSV 文件名
        objective_function: 可选自定义目标函数，signature f(x: np.ndarray) -> float

    返回：
        保存的 CSV 文件路径
    """
    if domain is None:
        domain = {f"x{i}": (-1, 1) for i in range(1, 11)}

    feature_names = list(domain.keys())
    lows = np.array([domain[k][0] for k in feature_names])
    highs = np.array([domain[k][1] for k in feature_names])

    # 生成随机样本矩阵
    samples_matrix = np.random.uniform(low=lows, high=highs, size=(n_samples, len(feature_names)))

    # 默认目标函数：Rastrigin（fun9）
    if objective_function is None:
        def objective_function(x: np.ndarray) -> float:
            x = np.asarray(x)
            return float(np.sum(x**2 - 10 * np.cos(np.pi * x)) + 10 * len(x))

    # 计算目标函数值
    y_values = np.array([objective_function(x) for x in samples_matrix])

    # 转为 DataFrame
    df = pd.DataFrame(samples_matrix, columns=feature_names)
    df["y"] = y_values

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    # 保存 CSV
    df.to_csv(output_path, index=False)
    print(f"✅ 随机样本已保存至: {output_path}")

    return output_path


# --------------------- CLI 测试入口 ---------------------
if __name__ == "__main__":
    # 示例：生成 1000 个样本，10维 Rastrigin
    DOMAIN_EXAMPLE = {f"x{i}": (-5.12, 5.12) for i in range(1, 11)}
    generate_random_samples(
        n_samples=1000,
        domain=DOMAIN_EXAMPLE,
        output_dir="external_test_samples",
        output_filename="external_test.csv"
    )


