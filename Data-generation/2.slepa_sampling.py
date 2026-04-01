# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 17:01:15 2025

@author: 91278
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
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

def safe_evaluate(objective: Callable, x: np.ndarray, f_num: int, reduce_methods=("sum", "mean", "first")) -> Optional[float]:
    """
    安全计算 objective(x, f_num)，确保返回标量或 None（当无法计算时）。
    reduce_methods 为尝试的降维顺序。
    """
    try:
        res = objective(x, f_num)
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

# --------------------- 默认目标函数 ---------------------
def default_objective(x, f_num):
    """默认函数（用于 CLI 测试）"""
    if f_num == 1:
        coefficients = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        return float(np.sum(coefficients * x))
    elif f_num == 2:
        coefficients = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        return float(np.sum(coefficients * (x ** 2)))
    elif f_num == 3:
        res = 0.0
        for i in range(len(x)-1):
            res += 100 * (x[i+1] - x[i]**2)**2 + (x[i] - 1)**2
        return float(res)
    elif f_num == 4:
        return float(np.sum(-x * np.sin(np.sqrt(np.abs(x)))))  # fun8
    elif f_num == 5:
        return float(np.sum(x**2 - 10 * np.cos(np.pi * x)) + 10 * len(x))  # fun9
    elif f_num == 6:
        res = 0.0
        for i in range(len(x)):
            res += np.sin(x[i]) * (np.sin((i+1) * x[i]**2 / np.pi))**20
        return float(-res)  # 取负号，因为原公式是最小化问题
    else:
        print(f"计算未知函数 F{f_num}")
        return None
    

# --------------------- SLPA 核心类 ---------------------
class SelfLearningPopulationAnnealing:
    """
    自学习群体退火采样器（基于 SLEPA 文献实现）
    核心机制：
    1. 使用 Gaussian Process 作为代理模型模拟能量函数。
    2. 群体根据代理模型的能量进行 MCMC 移动和重采样。
    3. 仅在更新代理模型时对部分样本调用真实目标函数。
    """

    def __init__(self,
                 objective_function: Callable[[np.ndarray], float],
                 bounds: List[Tuple[float, float]],
                 f_num: int,  # 函数编号
                 n_samples: int = 100,
                 population_size: int = 20,
                 n_offspring: int = 10,  # 此处语境下作为每轮迭代采集的真实样本数
                 seed: Optional[int] = None):
        """
        参数：
            objective_function: 可调用对象，真实的目标函数（昂贵）
            bounds: [(min1, max1), (min2, max2), ...]
            f_num: 函数编号
            n_samples: 总采样数（真实函数评估次数上限）
            population_size: 代理模型模拟所用的群体粒子数
            n_offspring: 每轮温度迭代中，选多少个新样本进行真实评估并加入训练集
            seed: 随机种子
        """
        if not callable(objective_function):
            raise ValueError("objective_function 必须是可调用的 (callable)")

        self.f = objective_function
        self.f_num = f_num
        self.bounds = bounds
        self.n_samples = n_samples
        self.population_size = population_size
        self.n_offspring = n_offspring
        self.dim = len(self.bounds)
        self.rng = np.random.default_rng(seed)
        
        # 初始化高斯过程作为代理模型
        # Matern 核函数适合模拟物理或平滑的景观，WhiteKernel 处理噪声 fixed 假设一个极小的数值稳定性噪声
        kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-6, noise_level_bounds="fixed")
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, normalize_y=True, random_state=seed)
        self.is_fitted = False


    def _predict_energy(self, X: np.ndarray) -> np.ndarray:
        """基于代理模型预测能量（如果未拟合则返回0或随机噪声）"""
        if not self.is_fitted:
            return np.zeros(len(X))
        
        # 使用均值作为能量返回 -> 找最小值
        mean, std = self.gp.predict(X, return_std=True)
        # return mean
        
        # 使用构造的均值与不确定性平衡探索和利用
        # High exploration: kappa = 3.0 ~ 5.0
        # Balanced: kappa = 1.96
        # Low exploration: kappa = 0.0 (Original code)
        kappa = 1.96
        # return mean - kappa * std
    
        # 构造平方反转势能：离中点越远，能量越低 -> 找最大和最小值
        midpoint = 0.5
        bi_polar_mean = -1.0 * (abs(mean) - midpoint)**2
        # print(mean, bi_polar_mean)
        return bi_polar_mean - kappa * std
        
        
    # def _predict_energy(self, X: np.ndarray) -> np.ndarray:
    #     if not self.is_fitted:
    #           # 初期没模型，随机探索，或者是基于距离的探索（尽量分散）
    #         return np.zeros(len(X))

    #     # 1. 获取预测值和不确定性
    #     mean, std = self.gp.predict(X, return_std=True)
        
    #     # 2. 构造特征项
        
    #     # A. 极值项：离数据中心的距离平方
    #     # 我们不仅要最小值，也要最大值。
    #     # 假设当前已知数据的平均值是 context_mean
    #     # context_mean = np.mean(self.gp.y_train_)  # 平均值
    #     context_mean = np.median(self.gp.y_train_)  # 中位数
    #     extreme_score = (mean - context_mean) ** 2
        
    #     # B. 不确定项：标准差
    #     uncertainty_score = std
        
    #     # C. 归一化 (非常重要，否则一项会压倒另一项)
    #     # 简单的 min-max 归一化到 [0, 1]
    #     def normalize(v):
    #         return (v - np.min(v)) / (np.max(v) - np.min(v) + 1e-9)
            
    #     score = 1.0 * normalize(extreme_score) + 2.0 * normalize(uncertainty_score)
        
    #     # 能量定义为负分数
    #     virtual_energy = -1.0 * score
        
    #     return virtual_energy


    def run(self, verbose: bool = True):
        """执行 SLEPA 采样"""
        bounds = np.array(self.bounds)
        dim = self.dim
        n_samples = self.n_samples
        population_size = self.population_size
        n_offspring = self.n_offspring
        rng = self.rng

        # ---------------- 1. 初始化数据收集 ----------------
        # 初始随机采样一批数据来训练第一个 GP
        n_init = min(population_size, n_samples)
        X_observed = []
        y_observed = []

        if verbose:
            print(f"初始化：生成 {n_init} 个随机样本进行初始评估...")

        init_pop = rng.uniform(bounds[:, 0], bounds[:, 1], (n_init, dim))
        for x in init_pop:
            y_val = safe_evaluate(self.f, x, self.f_num)
            if y_val is None or not np.isfinite(y_val):
                y_val = 0.0 # 异常处理
            X_observed.append(x)
            y_observed.append(y_val)

        X_observed = np.array(X_observed)
        y_observed = np.array(y_observed)
        samples_collected = len(X_observed)

        # ---------------- 2. 训练初始代理模型 ----------------
        y_mean = np.mean(y_observed)
        y_std = np.std(y_observed)
        y_observed_norm = (y_observed - y_mean) / y_std
        
        self.gp.fit(X_observed, y_observed_norm)
        self.is_fitted = True

        # ---------------- 3. 群体退火参数设置 ----------------
        # 逆温度 Beta 从 0 增加到 beta_max
        # 这里的 beta_max 和步数可以根据问题复杂度调整，文献常用 0 到 10
        beta_min = 0.0
        beta_max = 10.0
        # 迭代步数
        n_annealing_steps = max(5, int((n_samples - samples_collected) / max(1, n_offspring))) + 1
        n_warmup = n_annealing_steps // 2  # 前一半轮数用于纯探索（空间填充）

        # 构造 beta 列表, 前段为全0, 后段升温
        warmup_betas = np.zeros(n_warmup) 
        cooling_betas = np.linspace(beta_min, beta_max, n_annealing_steps - n_warmup)
        betas = np.concatenate([warmup_betas, cooling_betas])

        # 初始化群体 (M 个粒子)
        population = rng.uniform(bounds[:, 0], bounds[:, 1], (population_size, dim))
        
        # 计算初始群体在代理模型下的能量
        current_energies = self._predict_energy(population)

        # ---------------- 4. SLEPA 主循环 ----------------
        # 遍历温度阶梯
        for i in range(len(betas) - 1):
            if samples_collected >= n_samples:
                break

            beta_current = betas[i]
            beta_next = betas[i+1]
            d_beta = beta_next - beta_current

            if verbose:
                print(f"Iter {i+1}/{len(betas)-1}: Beta {beta_current:.2f} -> {beta_next:.2f}, 已收集 {samples_collected}/{n_samples}")

            # --- A. 重采样 (Resampling) ---
            # 根据 SLEPA/PA 理论，从 beta_i 到 beta_i+1 的重采样权重为 exp(-(beta_next - beta_current) * E)
            # 注意：如果是在求最大值，能量E取负值；假设目标函数越小越好（能量低），若目标是越大越好则需取反。
            # 为数值稳定性，减去最小能量
            log_weights = -d_beta * current_energies
            log_weights -= np.max(log_weights)
            weights = np.exp(log_weights)
            # 归一化
            if np.sum(weights) > 0:
                weights /= np.sum(weights)
            else:
                weights = np.ones(population_size) / population_size
            
            # 依据权重重采样群体
            indices = rng.choice(population_size, size=population_size, p=weights)
            population = population[indices]
            current_energies = current_energies[indices]

            # --- B. MCMC (Metropolis Sampling) on Surrogate ---
            # 在当前温度（beta_next）下，基于代理模型进行群体平衡
            # 增加 MCMC 步数以确保探索代理模型的地形
            n_mcmc_steps = 20 # 每个粒子尝试移动的次数
            accepted_count = 0
            
            # 确定步长 (随着 beta 增加，步长可以减小)
            step_scale = (bounds[:, 1] - bounds[:, 0]) * 0.1 * (1.0 / (1.0 + beta_next))

            for _ in range(n_mcmc_steps):
                # 提议新位置：高斯扰动
                perturbation = rng.normal(0, 1, (population_size, dim)) * step_scale
                proposals = population + perturbation
                # 边界处理
                proposals = np.clip(proposals, bounds[:, 0], bounds[:, 1])

                # 预测新位置的代理能量
                proposal_energies = self._predict_energy(proposals)

                # Metropolis 准则
                # delta_E = E_new - E_old
                # P_accept = min(1, exp(-beta * delta_E))
                delta_E = proposal_energies - current_energies
                
                # 计算接受概率 (beta_next)
                accept_prob = np.exp(-beta_next * delta_E)
                # 处理数值上溢/下溢
                accept_prob = np.minimum(1.0, accept_prob)
                
                random_vals = rng.random(population_size)
                accept_mask = random_vals < accept_prob
                
                # 更新被接受的粒子
                population[accept_mask] = proposals[accept_mask]
                current_energies[accept_mask] = proposal_energies[accept_mask]
                accepted_count += np.sum(accept_mask)

            # --- C. 自学习反馈 (Data Acquisition) ---
            # 从当前的群体中挑选最具代表性或未探索区域的样本进行真实评估
            # 策略：为了兼顾探索和挖掘，可以随机选取或基于距离选取
            # 为简单起见，这里从平衡后的群体中随机选取不重复的样本
            
            # 去除与已有观测点过于接近的样本 (简单去重)
            candidates_idx = rng.permutation(population_size)
            new_X = []
            new_y = []
            
            count_this_round = 0
            for idx in candidates_idx:
                if samples_collected + count_this_round >= n_samples:
                    break
                
                candidate = population[idx]
                
                # 简单检查是否与已有样本太近
                dists = np.sqrt(np.sum((X_observed - candidate)**2, axis=1))
                if np.min(dists) < 1e-4:
                    continue # 跳过重复点
                
                # 真实函数评估！
                y_true = safe_evaluate(self.f, candidate, self.f_num)
                if y_true is None: 
                    continue
                
                new_X.append(candidate)
                new_y.append(y_true)
                count_this_round += 1
                
                if count_this_round >= n_offspring:
                    break
            
            if len(new_X) > 0:
                new_X_arr = np.array(new_X)
                new_y_arr = np.array(new_y)
                
                X_observed = np.vstack([X_observed, new_X_arr])
                y_observed = np.concatenate([y_observed, new_y_arr])
                samples_collected += len(new_X)
                
                # --- D. 更新代理模型 ---
                y_mean = np.mean(y_observed)
                y_std = np.std(y_observed)
                y_observed_norm = (y_observed - y_mean) / y_std
                self.gp.fit(X_observed, y_observed_norm)
                # 更新当前群体的能量值（因为模型变了）
                current_energies = self._predict_energy(population)

        return X_observed[:n_samples], y_observed[:n_samples]

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


# -------------------------- 定义域 -------------------------
def def_domain(f_num):
    if f_num in [1, 2]:
        # F1-Linear & F2-Sphere
        DOMAIN = {
            "x1": (-1, 1),
            "x2": (-1, 1), 
            "x3": (-1, 1),
            "x4": (-1, 1),
            "x5": (-1, 1),
            "x6": (-1, 1),
            "x7": (-1, 1),
            "x8": (-1, 1),
            "x9": (-1, 1),
            "x10": (-1, 1)
        }
    elif f_num == 3:
        # F3-Rosenbrock (原F4)
        DOMAIN = {
            "x1": (-30, 30),  # Rosenbrock函数的典型定义域
            "x2": (-30, 30), 
            "x3": (-30, 30),
            "x4": (-30, 30),
            "x5": (-30, 30),
            "x6": (-30, 30),
            "x7": (-30, 30),
            "x8": (-30, 30),
            "x9": (-30, 30),
            "x10": (-30, 30)
        }
    elif f_num == 4:
        # F4-Schwefel (原F8)
        DOMAIN = {
            "x1": (-60, 60),  # Schwefel函数的典型定义域
            "x2": (-60, 60), 
            "x3": (-60, 60),
            "x4": (-60, 60),
            "x5": (-60, 60),
            "x6": (-60, 60),
            "x7": (-60, 60),
            "x8": (-60, 60),
            "x9": (-60, 60),
            "x10": (-60, 60)
        }
    elif f_num == 5:
        # F5-Rastrigin (原F9)
        DOMAIN = {
            "x1": (-5.12, 5.12),  # Rastrigin函数的典型定义域
            "x2": (-5.12, 5.12), 
            "x3": (-5.12, 5.12),
            "x4": (-5.12, 5.12),
            "x5": (-5.12, 5.12),
            "x6": (-5.12, 5.12),
            "x7": (-5.12, 5.12),
            "x8": (-5.12, 5.12),
            "x9": (-5.12, 5.12),
            "x10": (-5.12, 5.12)
        }
    elif f_num == 6:
        # F6-Michalewicz (原F3)
        DOMAIN = {
            "x1": (0, np.pi),
            "x2": (0, np.pi), 
            "x3": (0, np.pi),
            "x4": (0, np.pi),
            "x5": (0, np.pi),
            "x6": (0, np.pi),
            "x7": (0, np.pi),
            "x8": (0, np.pi),
            "x9": (0, np.pi),
            "x10": (0, np.pi)
        }
    else:
        print(f"没有这个函数 F{f_num}")
        DOMAIN = None
    return DOMAIN


# --------------------- 批量脚本接口函数 ---------------------
def generate_slpa_samples(
    domain: DomainType,
    objective_function: Callable[[np.ndarray], float],
    f_num: int,  # 函数编号
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
        f_num: 函数编号
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
        f_num=f_num,
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
    parser = argparse.ArgumentParser(description="自学习群体退火采样（SLEPA 实现版本）")
    parser.add_argument("--n_samples", type=int, default=100, help="总采样点数")
    parser.add_argument("--population_size", type=int, default=100, help="群体大小")
    parser.add_argument("--n_offspring", type=int, default=20, help="每轮退火新增的真实评估点数")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--out", type=str, default=None, help="输出CSV文件名")
    parser.add_argument("--domain_json", type=str, default=None, help="定义域JSON字符串")
    args = parser.parse_args()

    F_NUM = 6
    SAMPLE_NUM_LIST = [20, 50, 100, 200, 300, 500, 1000]  # 采样点数列表
    # SAMPLE_NUM_LIST = [20]
    # 默认定义域：10 维
    domain = def_domain(F_NUM)

    # CLI 使用默认 objective
    objective_func = default_objective
    
    n_samples = args.n_samples
    for n_samples in SAMPLE_NUM_LIST:        # 执行 SLPA 采样
        df = generate_slpa_samples(
            domain=domain,
            objective_function=objective_func,
            f_num=F_NUM,
            n_samples=n_samples,
            population_size=min(args.population_size, n_samples//2),
            n_offspring=min(args.n_offspring, args.n_samples//10),
            seed=args.seed,
            verbose=True
        )

        # 生成输出文件名
        out_file = args.out or f"./test/10dim_slpa_{n_samples}_fun{F_NUM}.csv"
        df.to_csv(out_file, index=False)
        print(f"\n✅ 已完成 SLPA 采样，输出文件: {out_file}")