#new_sobol_sampling
from scipy.stats.qmc import Sobol
import pandas as pd
import numpy as np
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


def sobol_sequencing(function_name=None, bounds=None, n_samples=128, reduce="sum", **kwargs):
    """
    Sobol低差异序列采样
    
    Parameters
    ----------
    function_name: 目标函数名称或索引（如 "S_f5" 或 5）
    bounds: 定义域 [(min1, max1), (min2, max2), ...]
    n_samples: 样本数量
    reduce: 当目标函数返回向量时的降维策略

    Returns
    -------
    X_samples: 采样点 (n_samples, dim)
    y_samples: 采样值 (n_samples,)
    """
    if bounds is None:
        bounds = [(0, 1)] * 4
        print("使用默认定义域: [(0,1)] * 4")
    
    dim = len(bounds)
    
    # 获取目标函数
    f = get_objective_function(function_name, reduce=reduce, **kwargs)
    
    # 创建Sobol采样器
    sobol_sampler = Sobol(d=dim, scramble=True)
    
    # 生成在 [0,1]^dim 的采样点
    unit_samples = sobol_sampler.random(n=n_samples)
    
    # 将单位超立方体中的点映射到实际定义域
    X_samples = np.zeros_like(unit_samples)
    for i in range(dim):
        low, high = bounds[i]
        X_samples[:, i] = unit_samples[:, i] * (high - low) + low
    
    # 计算函数值
    y_samples = []
    for x in X_samples:
        y_samples.append(f(x))
    
    return X_samples, np.array(y_samples)


def save_to_csv(X, y, filename_base="data_sobol"):
    """
    将采样数据保存为CSV文件
    
    参数:
        X: 采样点数组
        y: 函数值数组  
        filename_base: 文件名基础（不含扩展名）
    
    返回:
        df_with_y: 包含y的DataFrame
        df_without_y: 不包含y的DataFrame
    """
    dim = X.shape[1]
    
    # 创建包含y的DataFrame
    df_with_y = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(dim)])
    df_with_y["y"] = y
    
    # 创建不包含y的DataFrame
    df_without_y = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(dim)])
    
    # 保存文件
    filename_with_y = f"{filename_base}_with_y.csv"
    filename_without_y = f"{filename_base}.csv"
    
    df_with_y.to_csv(filename_with_y, index=False)
    df_without_y.to_csv(filename_without_y, index=False)
    
    print(f"包含y的数据已保存到 {filename_with_y}")
    print(f"不包含y的数据已保存到 {filename_without_y}")
    
    return df_with_y, df_without_y


# ------------------------
# 命令行接口
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sobol Sequence Sampler")
    parser.add_argument("--function", type=str, default="S_f1", help="目标函数名称或索引，如 'S_f5' 或 '5'")
    parser.add_argument("--n_samples", type=int, default=1024, help="采样点数量")
    parser.add_argument("--out", type=str, default="data_sobol_1024_10", help="输出文件基础名（不含扩展名）")
    parser.add_argument("--reduce", type=str, default="sum", choices=["sum", "mean", "first", "raise"], 
                       help="多输出函数的降维策略")
    parser.add_argument("--domain_json", type=str, default=None, 
                       help='定义域的 JSON 字符串，如 \'{"x1":[-10,10],"x2":[-5,5]}\'')
    args = parser.parse_args()

    # 默认定义域
    default_domain = {"x1": (-50.0, 50.0), "x2": (-25.0, 25.0), "x3": (-100.0, 100.0), "x4": (-75.0, 75.0)}

    if args.domain_json:
        try:
            parsed = json.loads(args.domain_json)
            if isinstance(parsed, dict):
                bounds = [tuple(parsed[k]) for k in parsed]
            elif isinstance(parsed, list):
                bounds = [tuple(v) for v in parsed]
            else:
                raise ValueError("domain_json 必须是 dict 或 list.")
        except Exception as e:
            print(f"解析 domain_json 失败：{e}", file=sys.stderr)
            sys.exit(1)
    else:
        bounds = [tuple(default_domain[k]) for k in default_domain]

    # 执行 Sobol 采样
    X, y = sobol_sequencing(
        function_name=args.function,
        bounds=bounds,
        n_samples=args.n_samples,
        reduce=args.reduce
    )
    
    print(f"Sampled {len(X)} points")
    print(f"Function range: min={min(y):.4f}, max={max(y):.4f}")
    
    # 保存数据到CSV文件
    df_with_y, df_without_y = save_to_csv(X, y, args.out)