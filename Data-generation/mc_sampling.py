# mc_sampling.py
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
      - dict: {"x1": (low, high), ...}
      - list/tuple: [(low, high), ...] -> 自动命名 x1, x2, ...
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

def generate_mc_samples(
    domain: DomainType,
    n_samples: int = 50,
    decimals: Optional[int] = 3,
    seed: Optional[int] = None,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    在给定定义域内使用均匀 Monte Carlo 产生样本（不评估目标函数）。
    参数:
      domain: 定义域（dict 或 list/tuple）
      n_samples: 生成样本数
      decimals: 保留小数位（None 表示不做四舍五入）
      seed: 随机种子（可选）
      columns: 自定义列名（长度必须等于维度数）
    返回:
      pandas.DataFrame：每列为一个维度
    """
    # 设置随机种子（若提供）
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

    # 生成样本（每维独立均匀分布）
    samples = np.empty((n_samples, dim), dtype=float)
    for j, (low, high) in enumerate(bounds):
        samples[:, j] = np.random.uniform(low=low, high=high, size=n_samples)

    # 四舍五入（可选）
    if decimals is not None:
        samples = np.round(samples, decimals)

    df = pd.DataFrame(samples, columns=col_names)
    return df

def save_samples_to_csv(df: pd.DataFrame, filename: str, index: bool = False, float_format: Optional[str] = "%.6f") -> None:
    """保存 DataFrame 为 CSV"""
    if float_format:
        df.to_csv(filename, index=index, float_format=float_format)
    else:
        df.to_csv(filename, index=index)

def plot_2d_projections(df: pd.DataFrame, savefig: Optional[str] = None, marker: str = "o") -> None:
    """
    画出所有两两维度的 2D 投影散点图。若维度为 d，则共有 C(d,2) 张子图。
    可选保存为文件（savefig）。
    """
    samples = df.to_numpy(dtype=float)
    dim = samples.shape[1]
    pairs = list(combinations(range(dim), 2))
    n_plots = len(pairs)
    if n_plots == 0:
        return

    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
    # 统一为扁平列表便于索引
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax in axes_list[n_plots:]:
        ax.axis("off")

    for ax, (i, j) in zip(axes_list, pairs):
        ax.scatter(samples[:, i], samples[:, j], marker=marker, alpha=0.6)
        ax.set_xlabel(df.columns[i])
        ax.set_ylabel(df.columns[j])
        ax.grid(True)

    plt.suptitle("Monte Carlo Sampling: 2D projections")
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 给标题留点空间
    if savefig:
        plt.savefig(savefig, bbox_inches="tight")
    plt.show()

def main(
    n_samples: int = 5,
    seed: Optional[int] = 42,
    decimals: Optional[int] = 3,
    out: str = "mc_samples.csv",
    domain_json: Optional[str] = None,
    columns: Optional[str] = None,
    plot: bool = False,
    savefig: Optional[str] = None
) -> pd.DataFrame:
    """
    主函数：解析 domain_json（若给定），生成样本并保存 CSV。返回 DataFrame。
    参数:
      n_samples, seed, decimals, out: 同前
      domain_json: JSON 字符串形式的 domain（见说明）
      columns: 以逗号分隔的列名字符串（例如 "a,b,c"）
      plot: 是否绘制两两投影图（默认 False）
      savefig: 若指定，绘图时将图保存为该文件名
    """
    import json

    # 默认 domain（4 维示例）
    default_domain = {
        "x1": (-25.0, 25.0),
        "x2": (-50.0, 50.0),
        "x3": (-75.0, 75.0),
        "x4": (-100.0, 100.0)
    }

    # 解析 domain_json（若提供）
    if domain_json:
        parsed = json.loads(domain_json)
        if isinstance(parsed, dict):
            domain = {k: tuple(v) for k, v in parsed.items()}
        elif isinstance(parsed, list):
            domain = [tuple(v) for v in parsed]
        else:
            raise ValueError("domain_json 必须是 dict 或 list.")
    else:
        domain = default_domain

    # 处理 columns 参数（逗号分隔）
    cols = None
    if columns:
        cols = [c.strip() for c in columns.split(",") if c.strip()]

    # 生成并保存
    df = generate_mc_samples(domain=domain, n_samples=n_samples, decimals=decimals, seed=seed, columns=cols)
    save_samples_to_csv(df, out)
    print(f"已生成 {len(df)} 个样本并保存为 {out}")

    if plot:
        plot_2d_projections(df, savefig=savefig)

    return df

# ---------- CLI 支持 ----------
if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser(description="Monte Carlo sampling: generate uniform samples within domain and save to CSV.")
    parser.add_argument("-n", "--n_samples", type=int, default=1000, help="样本数 (default: 5)")
    parser.add_argument("--seed", type=int, default=None, help="随机种子 (default: 42). Use --seed None to disable.")
    parser.add_argument("--decimals", type=int, default=3, help="保留小数位 (default: 3). Use -1 to disable rounding.")
    parser.add_argument("-o", "--out", type=str, default="mc_samples_100_1000_10.csv", help="输出 CSV 文件名")
    parser.add_argument("--domain_json", type=str, default=None, help='JSON 字符串指定 domain，例如 \'{"x1":[-1,1],"x2":[0,10]}\'')
    parser.add_argument("--columns", type=str, default=None, help='逗号分隔列名，例如 "a,b,c"')
    parser.add_argument("--plot", action="store_true", help="是否绘制两两维度的散点投影")
    parser.add_argument("--savefig", type=str, default=None, help="若指定，绘图时保存为该文件")
    args = parser.parse_args()

    # 处理 decimals 映射（-1 -> None）
    decimals_arg = None if args.decimals == -1 else args.decimals

    # 允许通过命令行传入字符串 'None' 来表示不设 seed
    seed_arg = None if (args.seed is None or str(args.seed).lower() == "none") else args.seed

    try:
        main(
            n_samples=args.n_samples,
            seed=seed_arg,
            decimals=decimals_arg,
            out=args.out,
            domain_json=args.domain_json,
            columns=args.columns,
            plot=args.plot,
            savefig=args.savefig
        )
    except Exception as e:
        print(f"运行出错：{e}", file=sys.stderr)
        sys.exit(1)

