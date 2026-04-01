# poisson_sampling.py
import numpy as np
import pandas as pd
from typing import List, Optional, Union, Dict, Tuple

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

def map_to_domain(samples: np.ndarray, domain: Dict[str, Tuple[float, float]]) -> np.ndarray:
    """
    将整数采样值线性映射到连续的目标定义域（min-max 映射）。
    注意：
      - 这是按列做的线性映射，会改变原始离散分布的统计性质（只是尺度变换，非概率保持）。
      - 如果想保持离散性或概率结构，请使用不同的方法（例如截断/重采样/核密度再采样等）。
    """
    dimensions = list(domain.keys())
    bounds = [domain[dim] for dim in dimensions]

    if samples.shape[1] != len(bounds):
        raise ValueError("samples 的列数必须等于 domain 中维度的数目")

    mapped_samples = np.empty_like(samples, dtype=float)

    for i, (min_val, max_val) in enumerate(bounds):
        original_values = samples[:, i].astype(float)
        original_min = np.min(original_values)
        original_max = np.max(original_values)
        if original_max == original_min:
            mapped_samples[:, i] = (min_val + max_val) / 2.0
        else:
            mapped_samples[:, i] = min_val + (original_values - original_min) * (max_val - min_val) / (original_max - original_min)

    return mapped_samples

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
    - 当 lam 为标量且 domain 提供时，如果 n_dimensions 未给出，会从 domain 推断维度数。
    - 如果 lam 为数组，其长度必须与最终维度一致（否则抛错）。
    """
    # 设置随机种子（注意：这会设置全局 numpy RNG；若需更现代接口可改为 default_rng）
    if seed is not None:
        np.random.seed(seed)

    # 确定维度数（优先级：lam 数组 -> domain -> n_dimensions 参数）
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
            # 两者都存在，必须匹配
            if len(lam_arr) != domain_dim:
                raise ValueError(f"传入的 lam 长度 ({len(lam_arr)}) 与 domain 维度 ({domain_dim}) 不匹配")
            n_dim = domain_dim
        else:
            n_dim = len(lam_arr)
            if domain_dim is not None and n_dim != domain_dim:
                raise ValueError(f"传入的 lam 长度 ({n_dim}) 与 domain 维度 ({domain_dim}) 不匹配")
    else:
        # lam 是标量
        if n_dimensions is not None:
            n_dim = n_dimensions
        elif domain_dim is not None:
            n_dim = domain_dim
        else:
            raise ValueError("当 lam 为标量时，必须提供 n_dimensions 或 domain 以推断维度数")

    # 生成原始泊松样本（整数）
    samples = poisson_sampling(n_samples, n_dim, lam)

    # 如果提供 domain，则映射并使用 domain 的键作为列名（保持键的插入顺序）
    if domain is not None:
        samples = map_to_domain(samples, domain)
        if columns is None:
            columns = list(domain.keys())

    # 设置列名
    if columns is None:
        columns = [f"x{i+1}" for i in range(n_dim)]
    elif len(columns) != n_dim:
        raise ValueError("columns 长度必须与维度数匹配")

    df = pd.DataFrame(samples, columns=columns)
    return df

def save_samples_to_csv(df: pd.DataFrame, filename: str, index: bool = False, float_format: Optional[str] = "%.6f") -> None:
    """将 DataFrame 保存为 CSV 文件（默认保留 6 位小数）。"""
    if float_format:
        df.to_csv(filename, index=index, float_format=float_format)
    else:
        df.to_csv(filename, index=index)

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="Generate Poisson samples and save to CSV.")
    parser.add_argument("-n", "--n_samples", type=int, default=1000, help="样本数 (default: 1000)")
    parser.add_argument("-d", "--n_dimensions", type=int, default=None, help="维度数 (default: inferred from lam or domain)")
    parser.add_argument("--lam", type=str, default="50",
                        help='λ参数，可以是单个数字或JSON数组，例如: "50" 或 "[10,20,30,40]" (default: 50)')
    parser.add_argument("--domain", type=str, default=None,
                        help='定义域映射，JSON格式，例如: \'{"x1":[-1,1],"x2":[0,10]}\'')
    parser.add_argument("--seed", type=int, default=None, help="随机种子 (optional)")
    parser.add_argument("-o", "--out", type=str, default="poisson_samples_1000_10.csv",
                        help="输出 CSV 文件名 (default: poisson_samples.csv)")
    args = parser.parse_args()

    # 默认 domain（放在 __main__ 中）
    default_domain = {
        "x1": (-50.0, 50.0),
        "x2": (-25.0, 25.0),
        "x3": (-100.0, 100.0),
        "x4": (-75.0, 75.0)
    }

    # 解析 lam 参数（可能是 JSON 数组或标量）
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
            print(f"lam 参数必须为数字或数字列表: {args.lam}", file=sys.stderr)
            sys.exit(1)

    # 解析 domain（如果提供），否则使用默认 domain
    if args.domain:
        try:
            domain_dict = json.loads(args.domain)
            if not isinstance(domain_dict, dict):
                raise ValueError("domain 参数必须为 JSON 字典")
            domain = {k: tuple(v) for k, v in domain_dict.items()}
        except Exception as e:
            print(f"解析 domain 参数失败: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        domain = default_domain

    # 如果 domain 存在，则优先用 domain 的维度（若用户未显式传 n_dimensions）
    if args.n_dimensions is None and domain is not None:
        inferred_dim = len(domain)
    else:
        inferred_dim = args.n_dimensions

    # 校验 lam 与维度一致性（若 lam 为数组）
    if isinstance(lam_val, (list, tuple, np.ndarray)):
        if inferred_dim is not None and len(lam_val) != inferred_dim:
            print(f"错误: lam 长度 ({len(lam_val)}) 与维度 ({inferred_dim}) 不匹配", file=sys.stderr)
            sys.exit(1)

    # 生成样本
    df = generate_poisson_samples(
        n_samples=args.n_samples,
        lam=lam_val,
        domain=domain,
        n_dimensions=inferred_dim,
        seed=args.seed
    )

    # 保存 CSV
    save_samples_to_csv(df, args.out)
    print(f"已生成 {len(df)} 个样本并保存为 {args.out}")
