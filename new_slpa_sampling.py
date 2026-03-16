# new_slpa_sampling.py
"""
Self-Learning Population Annealing (SLPA)
----------------------------------------
群体退火式的探索性采样方法。

特点：
- 支持从 Xfunction_module 导入黑箱函数
- 支持任意维度和定义域
- 自动保存采样结果为 CSV
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from Xfunction_module import get_high_dimensional_function
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

        # --- 新增：自动兼容 'S_f1'、'f1'、1、'1' 等输入 ---
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


class SelfLearningPopulationAnnealing:
    """
    自学习群体退火采样器
    """

    def __init__(self,
                 function_name=None,
                 bounds=None,
                 n_samples=100,
                 population_size=20,
                 n_offspring=10):
        """
        参数：
            function_name: 目标函数名称或索引（如 "S_f5" 或 5）
            bounds: [(min1, max1), (min2, max2), ...]
            n_samples: 总采样数
            population_size: 群体大小
            n_offspring: 每轮新增样本数
        """
        if bounds is None:
            # 默认定义域：4维 [0,1]
            self.bounds = [(0, 1)] * 4
            print("使用默认定义域: [(0,1)] * 4")
        else:
            self.bounds = bounds

        self.n_samples = n_samples
        self.population_size = population_size
        self.n_offspring = n_offspring
        self.dim = len(self.bounds)
        self.f = get_objective_function(function_name)

    # -------------------------------------------------------------------
    def run(self, verbose=True):
        """
        执行自学习群体退火采样
        返回：
            X_sampled, y_sampled
        """
        f = self.f
        bounds = self.bounds
        dim = self.dim
        n_samples = self.n_samples
        population_size = self.population_size
        n_offspring = self.n_offspring

        # 初始化群体
        population = np.array([
            [np.random.uniform(b[0], b[1]) for b in bounds]
            for _ in range(population_size)
        ])
        population_values = np.array([f(x) for x in population])

        X_sampled = population.copy()
        y_sampled = population_values.copy()
        samples_collected = population_size

        initial_temp, final_temp = 1.0, 0.1
        current_temp = initial_temp

        while samples_collected < n_samples:
            # 多样性指标：距离越大越好
            min_distances = []
            for i in range(population_size):
                distances = euclidean_distances([population[i]], X_sampled)[0]
                min_distances.append(np.min(distances))
            performance = np.array(min_distances)

            # Softmax 权重
            shifted = performance - np.max(performance)
            weights = np.exp(shifted / current_temp)
            weights /= np.sum(weights) if np.sum(weights) > 0 else population_size

            # 无重复重采样
            selected = []
            remaining_idx = list(range(population_size))
            remaining_w = weights.copy()
            for _ in range(population_size):
                if len(remaining_idx) == 0:
                    remaining_idx = list(range(population_size))
                    remaining_w = weights.copy()
                idx = np.random.choice(remaining_idx, p=remaining_w / remaining_w.sum())
                selected.append(idx)
                pos = remaining_idx.index(idx)
                remaining_idx.pop(pos)
                remaining_w = np.delete(remaining_w, pos)
            population = population[selected]

            # 变异
            mutation_strength = current_temp * 0.2
            for j in range(population_size):
                if np.random.random() < 0.5:
                    mutation = np.random.normal(0, mutation_strength, dim)
                    mutated = population[j] + mutation
                    for d in range(dim):
                        mutated[d] = np.clip(mutated[d], bounds[d][0], bounds[d][1])
                    population[j] = mutated

            # 评估
            population_values = np.array([f(x) for x in population])

            # 选择代表性样本
            n_to_collect = min(n_offspring, n_samples - samples_collected)
            diversities = []
            for i in range(population_size):
                distances = euclidean_distances([population[i]], X_sampled)[0]
                diversities.append(np.min(distances))
            best_idx = np.argsort(diversities)[-n_to_collect:]

            X_sampled = np.vstack([X_sampled, population[best_idx]])
            y_sampled = np.concatenate([y_sampled, population_values[best_idx]])
            samples_collected += n_to_collect

            # 降温
            cooling_rate = (final_temp / initial_temp) ** (1 / max(1, (n_samples - population_size) / 5))
            current_temp = max(final_temp, current_temp * cooling_rate)

            if verbose and samples_collected % 20 == 0:
                print(f"已收集 {samples_collected}/{n_samples} 个样本, 当前温度: {current_temp:.4f}")

        return X_sampled[:n_samples], y_sampled[:n_samples]

    # -------------------------------------------------------------------
    def save_to_csv(self, X, y, filename="data_SLPA.csv"):
        """
        保存采样结果到 CSV 文件
        """
        df = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(self.dim)])
        df["y"] = y
        df.to_csv(filename, index=False)
        print(f"采样结果已保存到 {filename}")
        return df


# ------------------------
# 命令行接口（可选）
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-Learning Population Annealing (SLPA) Sampler")
    parser.add_argument("--function", type=str, default="S_f1", help="目标函数名称或索引，如 'S_f5' 或 '5'")
    parser.add_argument("--n_samples", type=int, default=1000, help="采样点数量")
    parser.add_argument("--population_size", type=int, default=20, help="群体大小")
    parser.add_argument("--n_offspring", type=int, default=10, help="每轮新增样本数")
    parser.add_argument("--out", type=str, default="S_f1_data_SLPA_1000_10.csv", help="输出文件名")
    parser.add_argument("--domain_json", type=str, default=None, help='定义域的 JSON 字符串，如 \'{"x1":[-10,10],"x2":[-5,5]}\'')
    args = parser.parse_args()

    # 默认定义域
    default_domain = {"x1": (-50.0, 50.0), "x2": (-25.0, 25.0), "x3": (-100.0, 100.0), "x4": (-75.0, 75.0)}

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

    sampler = SelfLearningPopulationAnnealing(
        function_name=args.function,
        bounds=[tuple(domain[k]) for k in domain],  # 使用解析后的 domain
        n_samples=args.n_samples,
        population_size=args.population_size,
        n_offspring=args.n_offspring
    )
    X, y = sampler.run()
    sampler.save_to_csv(X, y, args.out)