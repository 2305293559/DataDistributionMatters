# possion_sampling_10d.py
import numpy as np
import pandas as pd
from typing import List, Optional, Union, Dict, Tuple

# --------------------------------------------------------------
# 泊松采样（核心）
# --------------------------------------------------------------
def poisson_sampling(n_samples: int,
                     n_dimensions: int,
                     lam: Union[float, List[float], np.ndarray]) -> np.ndarray:
    """
    使用泊松分布采样生成样本点
    返回 shape (n_samples, n_dimensions)
    """
    lam = np.array(lam) if np.ndim(lam) > 0 else np.full(n_dimensions, lam)
    if lam.shape[0] != n_dimensions:
        raise ValueError(f"lam 长度 ({lam.shape[0]}) 必须等于 n_dimensions ({n_dimensions})")
    result = np.random.poisson(lam=lam, size=(n_samples, n_dimensions))
    return result


# --------------------------------------------------------------
# 将 integer-Poisson 结果映射到连续定义域
# --------------------------------------------------------------
def map_to_domain(samples: np.ndarray, domain: Dict[str, Tuple[float, float]]) -> np.ndarray:
    """
    将整数采样值线性映射到连续的目标定义域（min-max 映射）。
    注意：这是尺度变换，会改变泊松分布的统计形状（仅保留排序结构）。
    """
    dimensions = list(domain.keys())
    bounds = [domain[dim] for dim in dimensions]

    if samples.shape[1] != len(bounds):
        raise ValueError("samples 的列数必须与 domain 维度数一致")

    mapped_samples = np.empty_like(samples, dtype=float)

    for i, (min_val, max_val) in enumerate(bounds):
        original_values = samples[:, i].astype(float)
        original_min = np.min(original_values)
        original_max = np.max(original_values)

        if original_max == original_min:
            mapped_samples[:, i] = (min_val + max_val) / 2.0
        else:
            mapped_samples[:, i] = min_val + (original_values - original_min) * \
                                   (max_val - min_val) / (original_max - original_min)

    return mapped_samples


# --------------------------------------------------------------
# 生成 Poisson 样本（可映射 domain）
# --------------------------------------------------------------
def generate_poisson_samples(
    n_samples: int,
    lam: Union[float, List[float], np.ndarray],
    domain: Optional[Dict[str, Tuple[float, float]]] = None,
    n_dimensions: Optional[int] = None,
    columns: Optional[List[str]] = None,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    生成泊松分布样本并返回 DataFrame，可选择映射到指定定义域。
    """
    if seed is not None:
        np.random.seed(seed)

    # ---------------------------
    # 确定维度数
    # ---------------------------
    lam_is_array = (np.ndim(lam) > 0)

    if domain is not None:
        domain_keys = list(domain.keys())
        domain_dim = len(domain_keys)
    else:
        domain_keys = None
        domain_dim = None

    if lam_is_array:
        lam_arr = np.array(lam)
        if n_dimensions is None and domain_dim is not None:
            if len(lam_arr) != domain_dim:
                raise ValueError(f"lam 长度 ({len(lam_arr)}) 与 domain 维度 ({domain_dim}) 不匹配")
            n_dim = domain_dim
        else:
            n_dim = len(lam_arr)
            if domain_dim is not None and n_dim != domain_dim:
                raise ValueError(f"lam 长度 ({n_dim}) 与 domain 维度 ({domain_dim}) 不匹配")
    else:
        if n_dimensions is not None:
            n_dim = n_dimensions
        elif domain_dim is not None:
            n_dim = domain_dim
        else:
            raise ValueError("当 lam 为标量时，必须提供 n_dimensions 或 domain")

    # ---------------------------
    # 泊松采样
    # ---------------------------
    samples = poisson_sampling(n_samples, n_dim, lam)

    # ---------------------------
    # 映射到定义域（若提供 domain）
    # ---------------------------
    if domain is not None:
        samples = map_to_domain(samples, domain)
        if columns is None:
            columns = list(domain.keys())

    # 列名
    if columns is None:
        columns = [f"x{i+1}" for i in range(n_dim)]
    elif len(columns) != n_dim:
        raise ValueError("columns 长度必须与维度相等")

    df = pd.DataFrame(samples, columns=columns)
    return df


# --------------------------------------------------------------
# 保存 CSV
# --------------------------------------------------------------
def save_samples_to_csv(df: pd.DataFrame, filename: str,
                        index: bool = False,
                        float_format: Optional[str] = "%.6f") -> None:
    if float_format:
        df.to_csv(filename, index=index, float_format=float_format)
    else:
        df.to_csv(filename, index=index)


# --------------------------------------------------------------
# CLI：10 维 Poisson 采样器
# --------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="Generate Poisson samples and save to CSV.")
    parser.add_argument("-n", "--n_samples", type=int, default=1000,
                        help="样本数 (default: 1000)")
    parser.add_argument("-d", "--n_dimensions", type=int, default=None,
                        help="维度（默认自动推断 lam 或 domain）")
    parser.add_argument("--lam", type=str, default="50",
                        help='λ，可为数字或 JSON 数组，例如 "50" 或 "[10,20,30]"')
    parser.add_argument("--domain", type=str, default=None,
                        help='定义域 JSON，例如 {"x1":[-1,1],"x2":[0,10]}')
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("-o", "--out", type=str,
                        default="poisson_samples_1000_10d.csv",
                        help="输出 CSV 文件名")

    args = parser.parse_args()

    # ----------------------------------------------------------
    # 默认 10 维定义域
    # ----------------------------------------------------------
    default_domain = {
       "x1": (-1, 1),
       "x2": (-1, 1),
       "x3": (-1, 1),
       "x4": (-1, 1),
       "x5": (-1, 1),
       "x6": (-1, 1),
       "x7": (-1, 1),
       "x8": (-1, 1),
       "x9": (-1, 1),
       "x10":(-1, 1)
    }

    # --------------------------
    # 解析 lam
    # --------------------------
    try:
        lam_parsed = json.loads(args.lam)
    except Exception:
        print(f"无法解析 lam 参数: {args.lam}", file=sys.stderr)
        sys.exit(1)

    if isinstance(lam_parsed, list):
        lam_val = lam_parsed
    else:
        try:
            lam_val = float(lam_parsed)
        except Exception:
            print(f"lam 必须为数字或 JSON 数组", file=sys.stderr)
            sys.exit(1)

    # --------------------------
    # 解析 domain
    # --------------------------
    if args.domain:
        try:
            domain_dict = json.loads(args.domain)
            if not isinstance(domain_dict, dict):
                raise ValueError("domain 必须为 JSON 字典")
            domain = {k: tuple(v) for k, v in domain_dict.items()}
        except Exception as e:
            print(f"解析 domain 时出错: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        domain = default_domain

    # 自动推断维度数
    if args.n_dimensions is None:
        inferred_dim = len(domain)
    else:
        inferred_dim = args.n_dimensions

    # lam 为数组时的维度检查
    if isinstance(lam_val, (list, tuple, np.ndarray)):
        if inferred_dim != len(lam_val):
            print(f"错误: lam 维度 {len(lam_val)} 与 domain/指定维度 {inferred_dim} 不匹配")
            sys.exit(1)

    # --------------------------
    # 生成样本
    # --------------------------
    df = generate_poisson_samples(
        n_samples=args.n_samples,
        lam=lam_val,
        domain=domain,
        n_dimensions=inferred_dim,
        seed=args.seed
    )

    save_samples_to_csv(df, args.out)
    print(f"已生成 {len(df)} 个 10 维 Poisson 样本并保存为 {args.out}")

