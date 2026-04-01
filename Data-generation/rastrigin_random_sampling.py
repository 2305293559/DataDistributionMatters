import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union

# =========================
# 类型定义
# =========================
DomainType = Union[
    Dict[str, Tuple[float, float]],
    List[Tuple[float, float]],
    Tuple[Tuple[float, float], ...]
]

# =========================
# 域规范化（原逻辑保留）
# =========================
def _normalize_domain(domain: DomainType) -> Tuple[List[str], List[Tuple[float, float]]]:
    if isinstance(domain, dict):
        names = list(domain.keys())
        bounds = [tuple(domain[k]) for k in names]
    elif isinstance(domain, (list, tuple)):
        bounds = [tuple(b) for b in domain]
        names = [f"x{i+1}" for i in range(len(bounds))]
    else:
        raise ValueError("domain must be dict or list/tuple of (low, high) pairs.")

    for b in bounds:
        if len(b) != 2:
            raise ValueError("each bound must be a (low, high) pair.")
        if b[0] >= b[1]:
            raise ValueError(f"invalid bound: low >= high {b}")

    return names, bounds

# =========================
# 原随机采样函数（未改）
# =========================
def generate_random_samples(
    domain: DomainType,
    n_samples: int = 50,
    decimals: Optional[int] = 3,
    seed: Optional[int] = None,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    if seed is not None:
        np.random.seed(seed)

    names, bounds = _normalize_domain(domain)
    dim = len(bounds)

    if columns is not None:
        if len(columns) != dim:
            raise ValueError("length of columns must equal domain dimension.")
        col_names = columns
    else:
        col_names = names

    samples = np.empty((n_samples, dim), dtype=float)
    for j, (low, high) in enumerate(bounds):
        samples[:, j] = np.random.uniform(low=low, high=high, size=n_samples)

    if decimals is not None:
        samples = np.round(samples, decimals)

    return pd.DataFrame(samples, columns=col_names)

# =========================
# Rastrigin 目标函数（F9）
# =========================
def your_objective_function(x: np.ndarray) -> float:
    """
    十维 Rastrigin 函数 (F9)
    f(x) = sum(x_i^2 - 10*cos(pi*x_i)) + 10*dim
    """
    x_array = np.asarray(x)
    dim = len(x_array)
    sum_term = np.sum(x_array**2 - 10 * np.cos(np.pi * x_array))
    return float(sum_term + 10 * dim)

# =========================
# 在随机采样基础上计算函数值
# =========================
def generate_random_samples_with_objective(
    domain: DomainType,
    n_samples: int = 100,
    decimals: Optional[int] = 3,
    seed: Optional[int] = None
) -> pd.DataFrame:

    df = generate_random_samples(
        domain=domain,
        n_samples=n_samples,
        decimals=decimals,
        seed=seed
    )

    df["y"] = df.apply(
        lambda row: your_objective_function(row.values),
        axis=1
    )

    return df

# =========================
# 主程序入口
# =========================
if __name__ == "__main__":

    # --------- Rastrigin 定义域 ---------
    DOMAIN = {
        "x1": (-5.12, 5.12),
        "x2": (-5.12, 5.12),
        "x3": (-5.12, 5.12),
        "x4": (-5.12, 5.12),
        "x5": (-5.12, 5.12),
        "x6": (-5.12, 5.12),
        "x7": (-5.12, 5.12),
        "x8": (-5.12, 5.12),
        "x9": (-5.12, 5.12),
        "x10": (-5.12, 5.12)
    }

    # --------- 可自由修改的参数 ---------
    N_SAMPLES = 1000000    # 采样数量
    DECIMALS = 4        # 小数位数
    SEED = None           # 随机种子
    OUTPUT_FILE = "rastrigin_random_1000000samples.csv"

    # --------- 生成数据 ---------
    df = generate_random_samples_with_objective(
        domain=DOMAIN,
        n_samples=N_SAMPLES,
        decimals=DECIMALS,
        seed=SEED
    )

    # --------- 保存结果 ---------
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"生成完成：{N_SAMPLES} 个样本")
    print(f"已保存至：{OUTPUT_FILE}")
    print(df.head())
