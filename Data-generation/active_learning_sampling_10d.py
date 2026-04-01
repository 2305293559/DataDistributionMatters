#active_learning_sampling_10dim.py
"""
new_active_learning_sampling.py

主动学习采样（接口暴露版本）
----------------------------------------
- 不再从外部模块自动导入黑箱函数（不再 import Xfunction_module）。
- 保留默认目标函数（x^2 求和）用于 CLI 测试/快速运行。
- 批量脚本可以直接 import generate_active_learning_samples 并传入自定义 objective_function。
- 默认维度已从 4 维扩展为 10 维。
- 对 None/NaN 做了鲁棒性处理以避免 GP 拟合失败。
"""

import numpy as np
import pandas as pd
import warnings
import argparse
import json
import sys
from typing import Dict, List, Tuple, Optional, Union, Callable, Any

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# --------------------- 类型定义 ---------------------
DomainType = Union[Dict[str, Tuple[float, float]], List[Tuple[float, float]]]


# --------------------- 工具函数 ---------------------
def _normalize_domain(domain: DomainType):
    """将 dict 或 list/tuple 形式的定义域标准化为 (names, bounds)"""
    if isinstance(domain, dict):
        names = list(domain.keys())
        bounds = [tuple(domain[k]) for k in names]
    else:
        bounds = [tuple(b) for b in domain]
        names = [f"x{i+1}" for i in range(len(bounds))]
    return names, bounds


def _force_scalar_y(y: Any) -> float:
    """确保 objective 输出为 float 标量"""
    if np.isscalar(y):
        return float(y)
    arr = np.asarray(y)
    if arr.size == 1:
        return float(arr.ravel()[0])
    raise ValueError("objective 返回多元素数组")


def safe_evaluate(objective: Callable, x: np.ndarray, reduce_methods=("sum", "mean", "first")) -> Optional[float]:
    """
    安全计算 objective(x)，确保返回标量或 None（当无法计算时）。
    reduce_methods 为尝试的降维顺序。
    """
    try:
        res = objective(x)
        try:
            return _force_scalar_y(res)
        except ValueError:
            arr = np.asarray(res)
            for m in reduce_methods:
                try:
                    if m == "sum":
                        return float(np.sum(arr))
                    elif m == "mean":
                        return float(np.mean(arr))
                    elif m == "first":
                        return float(arr.ravel()[0])
                except Exception:
                    continue
            return None
    except Exception:
        return None


def create_adaptive_kernel(bounds: List[Tuple[float, float]], iteration: int, dim: int):
    """构建自适应 Matern 核（用于 GP）"""
    x_ranges = [high - low for low, high in bounds]
    avg_range = np.mean(x_ranges) if len(x_ranges) > 0 else 1.0
    scale_factor = max(1.0, dim / 10.0)
    iteration_factor = min(5.0, 1.0 + iteration / 25.0)
    constant_bounds = (1e-5, 1e9 * iteration_factor)
    length_scale_bounds = (1e-4, 1e6 * scale_factor * iteration_factor)

    return (
        C(1.0, constant_bounds) *
        Matern(length_scale=max(1e-6, avg_range / 5.0),
               length_scale_bounds=length_scale_bounds, nu=2.5)
        + WhiteKernel(noise_level=1e-8, noise_level_bounds=(1e-10, 1e-2))
    )


# --------------------- 主动学习核心 ---------------------
def generate_active_learning_samples(
    domain: DomainType,
    objective_function: Callable[[np.ndarray], float],
    n_queries: int = 100,
    init_samples: int = 5,
    candidate_pool_size: int = 500,
    seed: Optional[int] = None,
    reduce_methods=("sum", "mean", "first"),
    verbose: bool = True,
) -> pd.DataFrame:
    """
    使用高斯过程不确定性驱动的主动学习采样。
    - domain: dict 或 list of (low,high)
    - objective_function: 可调用对象，signature f(x: np.ndarray) -> float
        （若从批量脚本调用，请直接传入你的函数）
    - 返回 DataFrame（被查询的 n_queries 个点及其 y）
    """

    if not callable(objective_function):
        raise ValueError("objective_function 必须是可调用的 (callable)")

    rng = np.random.default_rng(seed)
    names, bounds = _normalize_domain(domain)
    dim = len(bounds)

    if init_samples < 1:
        raise ValueError("init_samples 必须 >= 1")

    # ----- 初始化：生成 init_samples 个有效样本（确保 y 有效） -----
    X_train = []
    y_train = []
    attempts_limit = 5000
    for k in range(init_samples):
        attempts = 0
        while attempts < attempts_limit:
            x0 = np.array([rng.uniform(low, high) for (low, high) in bounds])
            y0 = safe_evaluate(objective_function, x0, reduce_methods=reduce_methods)
            attempts += 1
            if y0 is not None and np.isfinite(y0):
                X_train.append(x0)
                y_train.append(float(y0))
                break
        if attempts >= attempts_limit:
            raise RuntimeError("初始化阶段难以找到有效的初始样本，请检查 objective_function 的可评估性或放宽约束。")

    X_train = np.vstack(X_train)
    y_train = np.asarray(y_train, dtype=float)

    X_selected = []
    y_selected = []

    # 主循环：每次选取最不确定的 candidate（std 最大）
    for i in range(n_queries):
        kernel = create_adaptive_kernel(bounds, i, dim)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-3,
            normalize_y=True,
            n_restarts_optimizer=3,
            random_state=(None if seed is None else int(seed + i))
        )

        # 在 fit 之前确保没有 NaN/Inf
        valid_mask = np.isfinite(y_train)
        if np.sum(valid_mask) < 2:
            # 保证至少有 2 个样本用于 GP 拟合，否则随机挑选若干并继续
            while np.sum(valid_mask) < 2:
                xr = np.array([rng.uniform(low, high) for (low, high) in bounds])
                yr = safe_evaluate(objective_function, xr, reduce_methods=reduce_methods)
                if yr is not None and np.isfinite(yr):
                    X_train = np.vstack([X_train, xr])
                    y_train = np.append(y_train, float(yr))
                    valid_mask = np.isfinite(y_train)
            if verbose:
                print("补充随机样本以满足 GP 拟合的最小样本要求。")

        # 拟合 GP（只使用有限值）
        gp.fit(X_train[valid_mask], y_train[valid_mask])

        # 生成候选点池并计算预测不确定性
        candidate_pool = np.array([
            [rng.uniform(low, high) for (low, high) in bounds]
            for _ in range(candidate_pool_size)
        ])

        mu, std = gp.predict(candidate_pool, return_std=True)

        # 按 std 降序（越大越不确定），优先尝试最不确定点
        order = np.argsort(-std)

        chosen_x, chosen_y = None, None
        for idx in order:
            x_c = candidate_pool[idx]
            y_c = safe_evaluate(objective_function, x_c, reduce_methods=reduce_methods)
            if y_c is not None and np.isfinite(y_c):
                chosen_x, chosen_y = x_c, float(y_c)
                break

        # 若候选池内都不可评估（极少见），则随机采样直到找到有效样本
        if chosen_x is None:
            attempts = 0
            while attempts < 5000:
                xr = np.array([rng.uniform(low, high) for (low, high) in bounds])
                yr = safe_evaluate(objective_function, xr, reduce_methods=reduce_methods)
                attempts += 1
                if yr is not None and np.isfinite(yr):
                    chosen_x, chosen_y = xr, float(yr)
                    break
            if chosen_x is None:
                raise RuntimeError("无法在候选池或随机采样中找到有效的评估点，请检查 objective_function。")

        # 更新训练集
        X_train = np.vstack([X_train, chosen_x])
        y_train = np.append(y_train, chosen_y)

        X_selected.append(chosen_x)
        y_selected.append(chosen_y)

        if verbose and ((i + 1) % 10 == 0 or i == n_queries - 1):
            print(f"已完成 {i+1}/{n_queries} 次查询")

    df = pd.DataFrame(np.vstack(X_selected), columns=names)
    df["y"] = np.array(y_selected, dtype=float)
    return df


# --------------------- 默认目标函数（仅在 CLI 或未传入函数时使用） ---------------------
def default_objective(x: np.ndarray) -> float:
    """简单默认函数：向量元素平方和（用于 CLI 测试）"""
    return float(np.sum(np.asarray(x) ** 2))


# --------------------- 主程序入口（CLI） ---------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="主动学习采样（接口暴露版本）")
    parser.add_argument("--n_queries", type=int, default=100, help="主动学习采样次数")
    parser.add_argument("--init_samples", type=int, default=5, help="初始随机样本数")
    parser.add_argument("--candidate_pool", type=int, default=500, help="候选点数量")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--out", type=str, default=None, help="输出CSV文件名")
    parser.add_argument("--domain_json", type=str, default=None, help="定义域JSON字符串")
    args = parser.parse_args()

    # 默认定义域：10 维
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
            print(f"解析 domain_json 失败: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        domain = default_domain

    # CLI 使用默认 objective（如果你希望在 CLI 中使用自定义目标函数，请在批处理脚本中调用该模块）
    objective_func = default_objective

    # 执行主动学习采样
    df = generate_active_learning_samples(
        domain=domain,
        objective_function=objective_func,
        n_queries=args.n_queries,
        init_samples=args.init_samples,
        candidate_pool_size=args.candidate_pool,
        seed=args.seed,
        reduce_methods=("sum", "mean", "first"),
        verbose=True
    )

    # 生成输出文件名
    out_file = args.out or f"active_learning_{args.n_queries}_10d.csv"
    df.to_csv(out_file, index=False)
    print(f"\n✅ 已完成采样，输出文件: {out_file}")
