# sobel_sampling_10d.py
"""
Sobol sampling (10-dimensional, pure sampler version)
- No objective function
- Only generates Sobol low-discrepancy samples in a user-defined domain
- Saves CSV (with and without y) but y is removed in this version
"""

from scipy.stats.qmc import Sobol
import pandas as pd
import numpy as np
import argparse
import json
import sys


def sobol_sampling(bounds=None, n_samples=1024):
    """
    Pure Sobol sampling (no objective function)

    Parameters
    ----------
    bounds: list of (min,max) for each dimension
    n_samples: number of samples

    Returns
    -------
    X_samples: (n_samples, dim)
    """

    # Default 10D domain
    if bounds is None:
        print("使用默认 10 维定义域")
        bounds = [(-50, 50), (-25, 25), (-100, 100), (-75, 75),
                  (-10, 10), (-5, 5), (-20, 20), (-15, 15), (-30, 30), (-40, 40)]

    dim = len(bounds)

    sampler = Sobol(d=dim, scramble=True)

    # Generate samples in [0,1]^dim
    unit_samples = sampler.random(n=n_samples)

    # Map to domain
    X_samples = np.zeros_like(unit_samples)
    for i in range(dim):
        low, high = bounds[i]
        X_samples[:, i] = unit_samples[:, i] * (high - low) + low

    return X_samples


def save_sobol_csv(X, filename_base="sobol_10d"):
    dim = X.shape[1]

    df = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(dim)])

    fname = f"{filename_base}.csv"
    df.to_csv(fname, index=False)
    print(f"Sobol 数据已保存到 {fname}")

    return df


# ------------------------
# CLI
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="10D Sobol Sampler (pure sampler)")
    parser.add_argument("--n_samples", type=int, default=1024, help="采样点数量")
    parser.add_argument("--out", type=str, default="sobol_10d_1024", help="输出文件基础名")
    parser.add_argument("--domain_json", type=str, default=None,
                        help='定义域 JSON，如 {"x1":[-10,10], ... } 或 [[-10,10],[-5,5],...]')
    args = parser.parse_args()

    # Default 10D domain
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

    # Parse domain
    if args.domain_json:
        try:
            parsed = json.loads(args.domain_json)
            if isinstance(parsed, dict):
                bounds = [tuple(parsed[k]) for k in parsed]
            elif isinstance(parsed, list):
                bounds = [tuple(v) for v in parsed]
            else:
                raise ValueError("domain_json 必须为 dict 或 list")
        except Exception as e:
            print(f"解析 domain_json 失败: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        bounds = [default_domain[k] for k in default_domain]

    # Run Sobol sampling
    X = sobol_sampling(bounds=bounds, n_samples=args.n_samples)
    print(f"生成 {len(X)} 条 Sobol 采样数据 (10D)")

    # Save CSV
    save_sobol_csv(X, args.out)
