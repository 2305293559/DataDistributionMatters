# entropic_sampling_modified.py
"""
基于KDE（核密度估计）的熵采样实现
----------------------------------------
使用熵采样方法进行探索性采样。

特点：
- 支持从 Xfunction_module 导入黑箱函数
- 支持任意维度和定义域
- 自动保存采样结果为CSV（包含y和不包含y两种格式）
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
import argparse
import json
import sys
from typing import Optional

def _force_scalar_y(y):
    """确保目标函数输出为标量"""
    if np.isscalar(y):
        return float(y)
    arr = np.asarray(y)
    if arr.size == 1:
        return float(arr.ravel()[0])
    raise ValueError(f"目标函数返回了多元素数组 (shape {arr.shape})，请指定 reduce 策略")

def get_objective_function(function_name: Optional[str] = None, reduce: str = "sum", **kwargs):
    """
    获取目标函数；支持使用索引号（int 或 "1"）或函数名（如 "S_f1"）调用。
    当外部函数返回向量时按 reduce 策略降维。
    reduce: 'sum' | 'mean' | 'first' | 'raise'
    """
    def default_objective(x: np.ndarray) -> float:
        return float(np.sum(x**2))

    if function_name is None:
        return default_objective

    try:
        from Xfunction_module import get_high_dimensional_function

        # --- 自动兼容 'S_f1'、'f1'、1、'1' 等输入 ---
        name = str(function_name).strip()
        if name.startswith("S_f"):
            index_str = name.replace("S_f", "")
        elif name.startswith("f"):
            index_str = name.replace("f", "")
        else:
            index_str = name

        try:
            index = int(index_str)
        except ValueError:
            raise KeyError(f"无法识别函数名或索引: {function_name}. 请输入 1-16 或 S_f1-S_f16")

        func = get_high_dimensional_function(index)

        # --- 包装函数，确保输出为标量 ---
        def wrapped(x):
            res = func(x, **kwargs) if kwargs else func(x)
            try:
                return _force_scalar_y(res)
            except ValueError:
                arr = np.asarray(res)
                if arr.size == 0:
                    raise ValueError("objective 返回空数组")
                if reduce == "sum":
                    return float(np.sum(arr))
                elif reduce == "mean":
                    return float(np.mean(arr))
                elif reduce == "first":
                    return float(arr.ravel()[0])
                elif reduce == "raise":
                    raise ValueError("objective 返回多元素数组且 reduce='raise'")
                else:
                    raise ValueError(f"unknown reduce: {reduce}")
        return wrapped

    except Exception as e:
        print(f" 警告：无法导入或找到外部函数 ({e})，使用默认目标函数")
        return default_objective


def entropic_sampling(function_name=None, bounds=None, n_samples=100, n_initial=20, reduce="sum", **kwargs):
    """
    使用KDE（核密度估计）的熵采样实现
    
    参数：
        function_name: 目标函数名称或索引（如 "S_f5" 或 5）
        bounds: 各变量的边界 [(min1, max1), (min2, max2), ...]
        n_samples: 总样本数
        n_initial: 初始随机样本数
        reduce: 当目标函数返回多元素数组时的降维策略
    """
    if bounds is None:
        # 默认定义域：4维 [0,1]
        bounds = [(0, 1)] * 4
        print("使用默认定义域: [(0,1)] * 4")
    
    dim = len(bounds)
    f = get_objective_function(function_name, reduce=reduce, **kwargs)
    
    X_sampled = np.zeros((n_samples, dim))
    y_sampled = np.zeros(n_samples)
    
    # 初始随机采样
    for i in range(n_samples):
        if i < n_initial:  # 前n_initial个点随机采样
            x = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
        else:
            # 采样数据标准化
            x_scaler = StandardScaler()
            X_scaled = x_scaler.fit_transform(X_sampled[:i])
            # 训练KDE模型: 使用KDE估计概率密度，在低密度区域采样
            kde = KernelDensity(bandwidth=0.1)
            kde.fit(X_scaled)
            
            # 生成候选点并选择概率最低的点
            n_candidates = 1000
            candidates = np.array([
                [np.random.uniform(b[0], b[1]) for b in bounds] 
                for _ in range(n_candidates)
            ])
            # 标准化候选点
            candidates_scaled = x_scaler.transform(candidates)
            log_densities = kde.score_samples(candidates_scaled)
            x = candidates[np.argmin(log_densities)]  # 选择密度最低的点
        
        X_sampled[i] = x
        y_sampled[i] = f(x)
    
    return X_sampled, y_sampled


def save_to_csv(X, y, filename="data_entropy.csv"):
    """
    将采样数据保存为CSV文件（包含y和不包含y两种格式）
    """
    dim = X.shape[1]
    
    # 创建DataFrame - 包含y
    df_with_y = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(dim)])
    df_with_y["y"] = y
    
    # 创建DataFrame - 不包含y
    df_X_only = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(dim)])
    
    # 生成不含y的文件名
    if filename.endswith('.csv'):
        filename_X_only = filename.replace('.csv', '_Xonly.csv')
    else:
        filename_X_only = filename + '_Xonly.csv'
    
    # 保存到CSV
    df_with_y.to_csv(filename, index=False)
    df_X_only.to_csv(filename_X_only, index=False)
    
    print(f"包含y的数据已保存到 {filename}")
    print(f"仅包含X的数据已保存到 {filename_X_only}")
    
    return df_with_y, df_X_only


# ------------------------
# 命令行接口
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entropic Sampling with KDE")
    parser.add_argument("--function", type=str, default="S_f1", help="目标函数名称或索引，如 'S_f5' 或 '5'")
    parser.add_argument("--n_samples", type=int, default=1000, help="采样点数量")
    parser.add_argument("--n_initial", type=int, default=20, help="初始随机样本数")
    parser.add_argument("--out", type=str, default="S_f1_data_entropy_1000_10.csv", help="输出文件名")
    parser.add_argument("--reduce", type=str, default="sum", choices=["sum", "mean", "first", "raise"], 
                       help="当目标函数返回多元素数组时的降维策略")
    parser.add_argument("--domain_json", type=str, default=None, 
                       help='定义域的 JSON 字符串，如 \'{"x1":[-10,10],"x2":[-5,5]}\'')
    args = parser.parse_args()

    # 默认定义域
    default_domain = {"x1": (-25.0, 25.0),
    "x2": (-50.0, 50.0),
    "x3": (-75.0, 75.0),
    "x4": (-100.0, 100.0)}

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

    # 运行熵采样
    X, y = entropic_sampling(
        function_name=args.function,
        bounds=[tuple(domain[k]) for k in domain],
        n_samples=args.n_samples,
        n_initial=args.n_initial,
        reduce=args.reduce
    )
    
    print(f"采样完成: {len(X)} 个点")
    print(f"函数值范围: min={min(y):.4f}, max={max(y):.4f}")
    
    # 保存数据到CSV文件
    df_with_y, df_X_only = save_to_csv(X, y, args.out)