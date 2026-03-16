#!/usr/bin/env python3
"""
new_active_learning_sampling.py

主动学习采样（可自定义黑箱函数版本）
----------------------------------------
- 可通过命令行参数或函数调用指定 Xfunction_module 中的任意函数
- 在主动学习过程中始终使用同一个指定的黑箱函数
- 支持多维定义域与不同降维策略
"""

import numpy as np
import pandas as pd
import warnings
import argparse
import json
import sys
import os
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --------------------- 类型定义 ---------------------
DomainType = Union[Dict[str, Tuple[float, float]], List[Tuple[float, float]]]

# --------------------- 工具函数 ---------------------
def _normalize_domain(domain: DomainType):
    """将 dict 或 list/tuple 形式的定义域标准化"""
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
    """安全计算 objective(x)，确保返回标量"""
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
    """构建自适应 Matern 核"""
    x_ranges = [high - low for low, high in bounds]
    avg_range = np.mean(x_ranges)
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
    reduce_methods=("sum", "mean", "first")
) -> pd.DataFrame:
    """
    使用高斯过程不确定性驱动的主动学习采样。
    """
    rng = np.random.default_rng(seed)
    names, bounds = _normalize_domain(domain)
    dim = len(bounds)

    # 初始化随机样本
    X_train = np.array([
        [rng.uniform(low, high) for (low, high) in bounds]
        for _ in range(init_samples)
    ])
    y_train = np.array([
        safe_evaluate(objective_function, x, reduce_methods=reduce_methods)
        for x in X_train
    ], dtype=float)

    X_selected, y_selected = [], []

    for i in range(n_queries):
        kernel = create_adaptive_kernel(bounds, i, dim)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-3,
            normalize_y=True,
            n_restarts_optimizer=3,
            random_state=(None if seed is None else seed + i)
        )

        # 拟合 GP
        gp.fit(X_train, y_train)

        # 生成候选点
        candidate_pool = np.array([
            [rng.uniform(low, high) for (low, high) in bounds]
            for _ in range(candidate_pool_size)
        ])
        mu, std = gp.predict(candidate_pool, return_std=True)

        # 选最不确定的点
        order = np.argsort(-std)
        chosen_x, chosen_y = None, None
        for idx in order:
            x_c = candidate_pool[idx]
            y_c = safe_evaluate(objective_function, x_c, reduce_methods=reduce_methods)
            if y_c is not None:
                chosen_x, chosen_y = x_c, y_c
                break

        if chosen_x is None:
            # 如果全都无效，随机挑选一个
            while True:
                x_r = np.array([rng.uniform(low, high) for (low, high) in bounds])
                y_r = safe_evaluate(objective_function, x_r, reduce_methods=reduce_methods)
                if y_r is not None:
                    chosen_x, chosen_y = x_r, y_r
                    break

        # 更新数据集
        X_train = np.vstack([X_train, chosen_x])
        y_train = np.append(y_train, chosen_y)

        X_selected.append(chosen_x)
        y_selected.append(chosen_y)

        if (i + 1) % 10 == 0 or i == n_queries - 1:
            print(f"已完成 {i+1}/{n_queries} 次查询")

    df = pd.DataFrame(X_selected, columns=names)
    df["y"] = y_selected
    return df


# --------------------- 黑箱函数选择 ---------------------
def get_custom_objective(function_name: Optional[str] = None, reduce="sum", **kwargs):
    """
    通过 Xfunction_module 导入指定函数。
    例如：
        S_f8 或 8 或 f8 均可识别。
    """
    def default_func(x):
        return float(np.sum(x ** 2))

    if function_name is None:
        print("⚠️ 未指定函数，使用默认 x² 求和。")
        return default_func

    try:
        from Xfunction_module import get_high_dimensional_function
        # 自动解析输入
        name = str(function_name).strip()
        if name.startswith("S_f"):
            idx = int(name.replace("S_f", ""))
        elif name.startswith("f"):
            idx = int(name.replace("f", ""))
        else:
            idx = int(name)
        func = get_high_dimensional_function(idx)

        def wrapped(x):
            res = func(x, **kwargs) if kwargs else func(x)
            try:
                return _force_scalar_y(res)
            except ValueError:
                arr = np.asarray(res)
                if reduce == "sum":
                    return float(np.sum(arr))
                elif reduce == "mean":
                    return float(np.mean(arr))
                elif reduce == "first":
                    return float(arr.ravel()[0])
                else:
                    raise ValueError(f"未知 reduce 策略: {reduce}")
        return wrapped

    except Exception as e:
        print(f"⚠️ 加载函数 {function_name} 失败 ({e})，使用默认函数。")
        return default_func


# --------------------- 主程序入口 ---------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可自定义黑箱函数的主动学习采样")
    parser.add_argument("--function", type=str,default=" S_f1", help="指定目标函数，例如 S_f8 或 8")
    parser.add_argument("--reduce", type=str, default="sum", help="降维策略 sum|mean|first")
    parser.add_argument("--n_queries", type=int, default=1000, help="主动学习采样次数")
    parser.add_argument("--init_samples", type=int, default=5, help="初始随机样本数")
    parser.add_argument("--candidate_pool", type=int, default=500, help="候选点数量")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--out", type=str, default=None, help="输出CSV文件名")
    parser.add_argument("--domain_json", type=str, default=None, help="定义域JSON字符串")
    args = parser.parse_args()

    default_domain = {"x1": (-25.0, 25.0),
    "x2": (-50.0, 50.0),
    "x3": (-75.0, 75.0),
    "x4": (-100.0, 100.0)}
    
    if args.domain_json:
        try:
            domain = json.loads(args.domain_json)
            if isinstance(domain, dict):
                domain = {k: tuple(v) for k, v in domain.items()}
        except Exception as e:
            print(f"解析 domain_json 失败: {e}")
            sys.exit(1)
    else:
        domain = default_domain

    # 获取目标函数
    objective_func = get_custom_objective(args.function, reduce=args.reduce)

    # 执行主动学习采样
    df = generate_active_learning_samples(
        domain=domain,
        objective_function=objective_func,
        n_queries=args.n_queries,
        init_samples=args.init_samples,
        candidate_pool_size=args.candidate_pool,
        seed=args.seed,
        reduce_methods=(args.reduce, "mean", "first")
    )

    # 生成文件名
    func_label = args.function or "default"
    out_file = args.out or f"{func_label}_active_learning_{args.n_queries}_10.csv"

    df.to_csv(out_file, index=False)
    print(f"\n✅ 已完成采样，输出文件: {out_file}")
