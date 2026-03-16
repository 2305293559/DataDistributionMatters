# lhs_sampling.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from itertools import combinations
import matplotlib.pyplot as plt

DomainType = Union[Dict[str, Tuple[float, float]], List[Tuple[float, float]], Tuple[Tuple[float, float], ...]]

def _normalize_domain(domain: DomainType) -> Tuple[List[str], List[Tuple[float, float]]]:
    """
    将 domain 标准化为 (names, bounds_list) 形式。
    支持：
      1) dict: {"x1": (low, high), ...}
      2) list/tuple: [(low, high), ...] -> 自动命名 x1, x2, ...
    返回 names, bounds_list
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

def generate_lhs_samples(
    domain: DomainType,
    n_samples: int = 50,
    decimals: Optional[int] = 3,
    seed: Optional[int] = None,
    columns: Optional[List[str]] = None,
    center: bool = False
) -> pd.DataFrame:
    """
    使用拉丁超立方采样生成样本（不计算目标函数）。
    参数:
      domain: 定义域（dict 或 list/tuple）
      n_samples: 样本数
      decimals: 保留小数位；若为 None 则不做四舍五入
      seed: 随机种子（可选，传入整数以固定随机）
      columns: 自定义列名（长度必须等于维度）
      center: 若为 True，使用每格中心 (k+0.5)/n 采样；否则在格内随机采样
    返回:
      pandas.DataFrame
    """
    # 控制随机性（保持与 random_sampling.py 风格一致）
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

    # 归一化 [0,1) 的 LHS 采样：每维分 n_samples 个格子，每个格子恰有 1 个点
    samples = np.empty((n_samples, dim), dtype=float)
    for j in range(dim):
        perm = np.random.permutation(n_samples)  # 0..n_samples-1 的随机排列
        if center:
            u = (perm + 0.5) / n_samples
        else:
            u = (perm + np.random.rand(n_samples)) / n_samples
        samples[:, j] = u

    # 映射到指定区间
    low = np.array([b[0] for b in bounds], dtype=float)
    high = np.array([b[1] for b in bounds], dtype=float)
    samples = low + samples * (high - low)  # 广播映射

    # 可选四舍五入
    if decimals is not None:
        samples = np.round(samples, decimals)

    df = pd.DataFrame(samples, columns=col_names)
    return df

def save_samples_to_csv(df: pd.DataFrame, filename: str, index: bool = False) -> None:
    """将 DataFrame 保存为 CSV 文件。"""
    df.to_csv(filename, index=index)

def plot_2d_projections(df: pd.DataFrame, savefig: Optional[str] = None) -> None:
    """画出所有两两维度的 2D 投影散点图，保存（若指定）并显示。"""
    samples = df.to_numpy(dtype=float)
    dim = samples.shape[1]
    pairs = list(combinations(range(dim), 2))
    if not pairs:
        return
    n_plots = len(pairs)
    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    # 关闭多余子图
    for ax in axes_flat[n_plots:]:
        ax.axis("off")
    for ax, (i, j) in zip(axes_flat, pairs):
        ax.scatter(samples[:, i], samples[:, j], marker='o', alpha=0.7)
        ax.set_xlabel(df.columns[i])
        ax.set_ylabel(df.columns[j])
        ax.grid(True)
    plt.suptitle("Latin Hypercube Sampling: 2D projections")
    plt.tight_layout(rect=[0,0,1,0.96])
    if savefig:
        plt.savefig(savefig, bbox_inches="tight")
    plt.show()

# ---------- CLI 支持：运行时生成并保存 CSV ----------
if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="Generate Latin Hypercube Samples within a domain and save to CSV.")
    parser.add_argument("-n", "--n_samples", type=int, default=1000, help="样本数 (default: 5)")
    parser.add_argument("--seed", type=int, default=None, help="随机种子 (default: 42)")
    parser.add_argument("--decimals", type=int, default=3, help="保留小数位 (default: 3). Use -1 to disable rounding.")
    parser.add_argument("-o", "--out", type=str, default="lhs_samples_100_1000_10.csv", help="输出 CSV 文件名 (default: lhs_samples.csv)")
    parser.add_argument("--domain_json", type=str, default=None,
                        help='使用 JSON 字符串指定 domain，例如: \'{"x1":[-1,1],"x2":[0,10]}\'。若不提供将使用脚本内默认 domain。')
    parser.add_argument("--columns", type=str, default=None, help='逗号分隔的列名，例如 "a,b,c"')
    parser.add_argument("--center", action="store_true", help="是否在每格取中心点 (k+0.5)/n（默认在格内随机采样）")
    parser.add_argument("--plot", action="store_true", help="是否绘制两两维度的散点投影")
    parser.add_argument("--savefig", type=str, default=None, help="若指定且 --plot，则将图保存为该文件")
    args = parser.parse_args()

    # 默认 domain（示例，和你原始脚本中的上下界一致）
    default_domain = [(-25.0, 25.0), (-50.0, 50.0), (-75.0, 75.0), (-100.0, 100.0)]

    # 解析 domain_json（若提供）
    if args.domain_json:
        try:
            parsed = json.loads(args.domain_json)
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

    # 解析 columns（逗号分隔）
    cols = None
    if args.columns:
        cols = [c.strip() for c in args.columns.split(",") if c.strip()]

    # 生成并保存
    df = generate_lhs_samples(domain=domain,
                              n_samples=args.n_samples,
                              decimals=decimals,
                              seed=args.seed,
                              columns=cols,
                              center=args.center)
    save_samples_to_csv(df, args.out)
    print(f"已生成 {len(df)} 个样本并保存为 {args.out}")

