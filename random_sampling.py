# random_sampling.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union

DomainType = Union[Dict[str, Tuple[float, float]], List[Tuple[float, float]], Tuple[Tuple[float, float], ...]]

def _normalize_domain(domain: DomainType) -> Tuple[List[str], List[Tuple[float, float]]]:
    """
    将 domain 标准化为 (names, bounds_list) 形式。
    支持两种输入：
      1) dict: {"x1": (low, high), "x2": (low, high), ...}
      2) list/tuple: [(low, high), (low, high), ...] -> 自动命名为 x1, x2, ...
    返回：
      names: 列名列表
      bounds_list: 与 names 对应的 (low, high) 列表
    """
    if isinstance(domain, dict):
        names = list(domain.keys())
        bounds = [tuple(domain[k]) for k in names]
    elif isinstance(domain, (list, tuple)):
        bounds = [tuple(b) for b in domain]
        names = [f"x{i+1}" for i in range(len(bounds))]
    else:
        raise ValueError("domain 必须是 dict 或 list/tuple of (low, high) pairs.")
    # 验证每个 bound
    for b in bounds:
        if len(b) != 2:
            raise ValueError("每个 bound 必须是 (low, high) 元组/列表。")
        if b[0] >= b[1]:
            raise ValueError(f"bound 不合法：low >= high ({b})")
    return names, bounds

def generate_random_samples(
    domain: DomainType,
    n_samples: int = 50,
    decimals: Optional[int] = 3,
    seed: Optional[int] = None,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    在给定定义域内均匀生成随机样本（不计算目标函数）。
    参数:
      domain: 定义域，支持 dict 或 list/tuple，参见 _normalize_domain 注释
      n_samples: 生成样本数
      decimals: 保留小数位；若为 None 则不做四舍五入
      seed: 随机种子（可选）
      columns: 覆盖列名（如果提供，长度必须等于维度数）
    返回:
      pandas.DataFrame，列为样本维度
    """
    if seed is not None:
        np.random.seed(seed)

    names, bounds = _normalize_domain(domain)
    dim = len(bounds)

    if columns is not None:
        if len(columns) != dim:
            raise ValueError("columns 长度必须与域的维度相等。")
        col_names = columns
    else:
        col_names = names

    # 生成样本
    samples = np.empty((n_samples, dim), dtype=float)
    for j, (low, high) in enumerate(bounds):
        samples[:, j] = np.random.uniform(low=low, high=high, size=n_samples)

    # 保留小数（可选）
    if decimals is not None:
        samples = np.round(samples, decimals)

    df = pd.DataFrame(samples, columns=col_names)
    return df

def save_samples_to_csv(df: pd.DataFrame, filename: str, index: bool = False) -> None:
    """将 DataFrame 保存为 CSV 文件。"""
    df.to_csv(filename, index=index)


# ---------- CLI 支持：运行时生成并保存 CSV ----------
if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="Generate random samples within a domain and save to CSV.")
    parser.add_argument("-n", "--n_samples", type=int, default=1000, help="样本数 (default: 5)")
    parser.add_argument("--seed", type=int, default=None, help="随机种子 (default: None)")
    parser.add_argument("--decimals", type=int, default=3, help="保留小数位 (default: 3). Use -1 to disable rounding.")
    parser.add_argument("-o", "--out", type=str, default="random_samples_1000_10.csv", help="输出 CSV 文件名 (default: random_samples.csv)")
    parser.add_argument("--domain_json", type=str, default=None,
                        help='使用 JSON 字符串指定 domain，例如: \'{"x1":[-1,1],"x2":[0,10]}\'。若不提供将使用脚本内默认 domain。')
    args = parser.parse_args()

    # 默认 domain（如果想用你原始的例子）
    default_domain = {
        "x1": (-50.0, 50.0),
        "x2": (-25.0, 25.0),
        "x3": (-100.0, 100.0),
        "x4": (-75.0, 75.0)
    }

    # 解析 domain_json（若提供）
    if args.domain_json:
        try:
            parsed = json.loads(args.domain_json)
            # 将列表值转换为 tuple（normalize 接受 list/tuple，但为了字典存储转换方便）
            if isinstance(parsed, dict):
                domain = {k: tuple(v) for k, v in parsed.items()}
            elif isinstance(parsed, list):
                domain = [tuple(v) for v in parsed]
            else:
                raise ValueError("domain_json 必须是 dict 或 list.")
        except Exception as e:
            print(f"解析 domain_json 失败：{e}", file=sys.stderr)
            sys.exit(1)
    else:
        domain = default_domain

    decimals = None if args.decimals == -1 else args.decimals

    # 生成样本并保存
    df = generate_random_samples(domain,
                                 n_samples=args.n_samples,
                                 decimals=decimals,
                                 seed=args.seed
                                 )
    save_samples_to_csv(df, args.out)
    print(f"已生成 {len(df)} 个样本并保存为 {args.out}")
