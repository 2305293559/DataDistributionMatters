#slpa_sampling_10dim.py
"""
Self-Learning Population Annealing (SLPA) - 接口暴露版本
----------------------------------------
群体退火式的探索性采样方法。

特点：
- 不再从 Xfunction_module 导入黑箱函数
- 保留默认目标函数（x^2 求和）用于 CLI 测试/快速运行
- 批量脚本可以直接传入自定义 objective_function
- 默认维度已从 4 维扩展为 10 维
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import argparse
import json
import sys
from typing import Optional, Dict, List, Tuple, Union, Callable, Any

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
    """确保目标函数输出为标量"""
    if np.isscalar(y):
        return float(y)
    arr = np.asarray(y)
    if arr.size == 1:
        return float(arr.ravel()[0])
    raise ValueError(f"目标函数返回了多元素数组 (shape {arr.shape})，请指定 reduce 策略")


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


# --------------------- 默认目标函数（仅在 CLI 或未传入函数时使用） ---------------------
def default_objective(x: np.ndarray) -> float:
    """简单默认函数：向量元素平方和（用于 CLI 测试）"""
    return float(np.sum(np.asarray(x) ** 2))


# --------------------- SLPA 核心类 ---------------------
class SelfLearningPopulationAnnealing:
    """
    自学习群体退火采样器（接口暴露版本）
    """

    def __init__(self,
                 objective_function: Callable[[np.ndarray], float],
                 bounds: List[Tuple[float, float]],
                 n_samples: int = 100,
                 population_size: int = 20,
                 n_offspring: int = 10,
                 seed: Optional[int] = None):
        """
        参数：
            objective_function: 可调用对象，signature f(x: np.ndarray) -> float
            bounds: [(min1, max1), (min2, max2), ...]
            n_samples: 总采样数
            population_size: 群体大小
            n_offspring: 每轮新增样本数
            seed: 随机种子
        """
        if not callable(objective_function):
            raise ValueError("objective_function 必须是可调用的 (callable)")

        self.f = objective_function
        self.bounds = bounds
        self.n_samples = n_samples
        self.population_size = population_size
        self.n_offspring = n_offspring
        self.dim = len(self.bounds)
        self.rng = np.random.default_rng(seed)

    def run(self, verbose: bool = True):
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
        rng = self.rng

        # 初始化群体
        population = np.array([
            [rng.uniform(b[0], b[1]) for b in bounds]
            for _ in range(population_size)
        ])
        
        # 安全评估初始群体
        population_values = []
        for x in population:
            y_val = safe_evaluate(f, x, reduce_methods=("sum", "mean", "first"))
            if y_val is None or not np.isfinite(y_val):
                # 如果评估失败，使用默认值并警告
                y_val = 0.0
                if verbose:
                    print(f"警告：初始样本评估失败，使用默认值 0.0")
            population_values.append(y_val)
        population_values = np.array(population_values)

        X_sampled = population.copy()
        y_sampled = population_values.copy()
        samples_collected = population_size

        initial_temp, final_temp = 1.0, 0.1
        current_temp = initial_temp

        while samples_collected < n_samples:
            # 多样性指标：距离越大越好
            if len(X_sampled) > 0:
                min_distances = []
                for i in range(population_size):
                    distances = euclidean_distances([population[i]], X_sampled)[0]
                    min_distances.append(np.min(distances))
                performance = np.array(min_distances)
            else:
                performance = np.ones(population_size)

            # Softmax 权重
            shifted = performance - np.max(performance)
            weights = np.exp(shifted / current_temp)
            weights_sum = np.sum(weights)
            if weights_sum > 0:
                weights /= weights_sum
            else:
                weights = np.ones(population_size) / population_size

            # 无重复重采样
            selected = []
            remaining_idx = list(range(population_size))
            remaining_w = weights.copy()
            for _ in range(population_size):
                if len(remaining_idx) == 0:
                    remaining_idx = list(range(population_size))
                    remaining_w = weights.copy()
                
                if np.sum(remaining_w) > 0:
                    remaining_w_normalized = remaining_w / np.sum(remaining_w)
                    idx = rng.choice(remaining_idx, p=remaining_w_normalized)
                else:
                    idx = rng.choice(remaining_idx)
                    
                selected.append(idx)
                pos = remaining_idx.index(idx)
                remaining_idx.pop(pos)
                remaining_w = np.delete(remaining_w, pos)
            population = population[selected]

            # 变异
            mutation_strength = current_temp * 0.2
            for j in range(population_size):
                if rng.random() < 0.5:
                    mutation = rng.normal(0, mutation_strength, dim)
                    mutated = population[j] + mutation
                    for d in range(dim):
                        low, high = bounds[d]
                        mutated[d] = np.clip(mutated[d], low, high)
                    population[j] = mutated

            # 安全评估当前群体
            population_values = []
            for x in population:
                y_val = safe_evaluate(f, x, reduce_methods=("sum", "mean", "first"))
                if y_val is None or not np.isfinite(y_val):
                    y_val = 0.0  # 评估失败时使用默认值
                population_values.append(y_val)
            population_values = np.array(population_values)

            # 选择代表性样本
            n_to_collect = min(n_offspring, n_samples - samples_collected)
            
            if len(X_sampled) > 0:
                diversities = []
                for i in range(population_size):
                    distances = euclidean_distances([population[i]], X_sampled)[0]
                    diversities.append(np.min(distances))
                best_idx = np.argsort(diversities)[-n_to_collect:]
            else:
                best_idx = rng.choice(population_size, n_to_collect, replace=False)

            X_sampled = np.vstack([X_sampled, population[best_idx]])
            y_sampled = np.concatenate([y_sampled, population_values[best_idx]])
            samples_collected += n_to_collect

            # 降温
            cooling_rate = (final_temp / initial_temp) ** (1 / max(1, (n_samples - population_size) / 5))
            current_temp = max(final_temp, current_temp * cooling_rate)

            if verbose and samples_collected % 20 == 0:
                print(f"已收集 {samples_collected}/{n_samples} 个样本, 当前温度: {current_temp:.4f}")

        return X_sampled[:n_samples], y_sampled[:n_samples]

    def save_to_csv(self, X, y, filename="data_SLPA.csv"):
        """
        保存采样结果到 CSV 文件
        """
        names, _ = _normalize_domain(self.bounds)
        df = pd.DataFrame(X, columns=names)
        df["y"] = y
        df.to_csv(filename, index=False)
        print(f"采样结果已保存到 {filename}")
        return df


# --------------------- 批量脚本接口函数 ---------------------
def generate_slpa_samples(
    domain: DomainType,
    objective_function: Callable[[np.ndarray], float],
    n_samples: int = 100,
    population_size: int = 20,
    n_offspring: int = 10,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    使用自学习群体退火算法进行采样。
    
    参数：
        domain: dict 或 list of (low,high)
        objective_function: 可调用对象，signature f(x: np.ndarray) -> float
        n_samples: 总采样数
        population_size: 群体大小
        n_offspring: 每轮新增样本数
        seed: 随机种子
        verbose: 是否显示进度信息
        
    返回：
        DataFrame（包含采样点和对应的目标值）
    """
    names, bounds = _normalize_domain(domain)
    
    sampler = SelfLearningPopulationAnnealing(
        objective_function=objective_function,
        bounds=bounds,
        n_samples=n_samples,
        population_size=population_size,
        n_offspring=n_offspring,
        seed=seed
    )
    
    X, y = sampler.run(verbose=verbose)
    
    df = pd.DataFrame(X, columns=names)
    df["y"] = np.array(y, dtype=float)
    return df


# --------------------- 主程序入口（CLI） ---------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自学习群体退火采样（接口暴露版本）")
    parser.add_argument("--n_samples", type=int, default=100, help="总采样点数")
    parser.add_argument("--population_size", type=int, default=20, help="群体大小")
    parser.add_argument("--n_offspring", type=int, default=10, help="每轮新增样本数")
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

    # CLI 使用默认 objective
    objective_func = default_objective

    # 执行 SLPA 采样
    df = generate_slpa_samples(
        domain=domain,
        objective_function=objective_func,
        n_samples=args.n_samples,
        population_size=args.population_size,
        n_offspring=args.n_offspring,
        seed=args.seed,
        verbose=True
    )

    # 生成输出文件名
    out_file = args.out or f"slpa_samples_{args.n_samples}_10d.csv"
    df.to_csv(out_file, index=False)
    print(f"\n✅ 已完成 SLPA 采样，输出文件: {out_file}")
