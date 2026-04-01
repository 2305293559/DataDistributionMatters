import numpy as np
import pandas as pd
import warnings
from typing import Dict, Tuple, Callable, Union, List, Optional, Any
from scipy.stats import qmc, norm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

DomainType = Union[Dict[str, Tuple[float, float]], List[Tuple[float, float]]]

def _normalize_domain(domain: DomainType):
    if isinstance(domain, dict):
        names = list(domain.keys())
        bounds = [tuple(domain[k]) for k in names]
    else:
        bounds = [tuple(b) for b in domain]
        names = [f"x{i+1}" for i in range(len(bounds))]
    return names, bounds


def _safe_scalar(y: Any) -> Optional[float]:
    if np.isscalar(y):
        return float(y)
    arr = np.asarray(y)
    if arr.size == 1:
        return float(arr.ravel()[0])
    return None


def safe_evaluate(func: Callable, x: np.ndarray) -> Optional[float]:
    try:
        y = func(x)
        return _safe_scalar(y)
    except Exception:
        return None


def create_kernel(bounds, iteration, dim):
    # 类似原实现：平均区间尺度确定长度尺度
    avg_range = np.mean([h - l for l, h in bounds])
    return (
        C(1.0, (1e-5, 1e6))
        * Matern(length_scale=max(1e-6, avg_range / 5), nu=2.5)
        + WhiteKernel(1e-8, (1e-10, 1e-2))
    )


# ------------------------------
# 采集函数 — 仅用于"寻找最大值"：
# 1) Expected Improvement (EI)（默认）
# 2) 可选 Upper Confidence Bound (UCB)
# ------------------------------
class MaxAcquisition:
    @staticmethod
    def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float, xi: float = 1e-3):
        """
        Expected Improvement for maximization.
        EI(x) = (mu - best - xi) * Phi(z) + sigma * phi(z),  z = (mu - best - xi) / sigma
        numerical safe: when sigma ~ 0, EI ~ max(mu - best - xi, 0)
        """
        sigma = np.maximum(sigma, 1e-12)
        imp = mu - best - xi
        z = imp / sigma
        Phi = norm.cdf(z)
        phi = norm.pdf(z)
        ei = imp * Phi + sigma * phi
        ei[ sigma <= 1e-12 ] = np.maximum(imp[ sigma <= 1e-12 ], 0.0)
        # ensure non-negative
        ei = np.maximum(ei, 0.0)
        return ei

    @staticmethod
    def ucb(mu: np.ndarray, sigma: np.ndarray, kappa: float = 2.0):
        return mu + kappa * sigma


# =====================================================
# 贝叶斯采样主函数（只寻找最大值）
# =====================================================
def _bayesian_maximization(
    domain: DomainType,
    objective_function: Callable[[np.ndarray], float],
    n_queries: int,
    init_samples: int,
    seed: Optional[int],
    verbose: bool,
    acquisition: str = "ei",   # "ei" or "ucb"
    candidate_pool_size: int = 2000,
    xi: float = 1e-3,
    kappa: float = 2.0
) -> pd.DataFrame:

    rng = np.random.default_rng(seed)
    names, bounds = _normalize_domain(domain)
    dim = len(bounds)

    # 初始化样本（LHS）
    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    X = sampler.random(n=init_samples)
    X = np.array([[l + (h - l) * v for (l, h), v in zip(bounds, row)] for row in X])

    # 评估初始点，遇到不可用值用非常小的数代替（便于最大化问题）
    y_list = []
    for x in X:
        val = safe_evaluate(objective_function, x)
        if val is None or not np.isfinite(val):
            val = -1e9  # 非可行或异常点，标为极低值
        y_list.append(val)
    y = np.asarray(y_list)

    # 当前最优（最大化问题）
    best_y = np.max(y) if y.size > 0 else -np.inf

    selected_X, selected_y = [], []

    for i in range(n_queries):
        # 训练 GP
        gp = GaussianProcessRegressor(
            kernel=create_kernel(bounds, i, dim),
            normalize_y=True,
            random_state=None if seed is None else int(seed + i)
        )
        gp.fit(X, y)

        # 候选集合（均匀随机）
        candidates = np.array([
            [rng.uniform(l, h) for l, h in bounds]
            for _ in range(candidate_pool_size)
        ])

        mu, std = gp.predict(candidates, return_std=True)

        if acquisition == "ei":
            scores = MaxAcquisition.expected_improvement(mu, std, best_y, xi=xi)
        elif acquisition == "ucb":
            scores = MaxAcquisition.ucb(mu, std, kappa=kappa)
        else:
            raise ValueError("acquisition must be 'ei' or 'ucb'")

        # 选择分数最高的候选
        idx = int(np.argmax(scores))
        x_new = candidates[idx]
        y_new = safe_evaluate(objective_function, x_new)

        # 如果评估失败，跳过该次查询（不计入 selected），但仍计数 i
        if y_new is None or not np.isfinite(y_new):
            # 把这个点也记录为极低值，以避免重复选择同一点
            X = np.vstack([X, x_new])
            y = np.append(y, -1e9)
            if verbose:
                print(f"[BayesMax] iter {i+1}/{n_queries}: invalid evaluation, skipped (marked very low).")
            continue

        # 更新数据集
        X = np.vstack([X, x_new])
        y = np.append(y, y_new)

        selected_X.append(x_new)
        selected_y.append(y_new)

        # 更新最优
        if y_new > best_y:
            best_y = y_new

        if verbose and (i + 1) % 10 == 0:
            print(f"[BayesMax] {i+1}/{n_queries} | best={best_y:.6f}")

    # 结果 DataFrame（若没有选中任何点，返回空 df 但保留列）
    if len(selected_X) == 0:
        df = pd.DataFrame(columns=names + ["y"])
    else:
        df = pd.DataFrame(np.vstack(selected_X), columns=names)
        df["y"] = selected_y
    return df


# =====================================================
# 对外接口（保持原函数名与签名）
# =====================================================
def generate_slpa_samples(
    domain: DomainType,
    objective_function: Callable[[np.ndarray], float],
    n_samples: int,
    population_size: int = None,   # 兼容 batch_run，不使用
    n_offspring: int = None,        # 兼容 batch_run，不使用
    seed: Optional[int] = None,
    verbose: bool = False
) -> pd.DataFrame:
    """
    统一接口（与原接口兼容），内部使用面向最大化的贝叶斯采样。
    - n_samples: 期望额外查询次数（采样次数）
    - 返回 DataFrame，列为输入维度名和 y（目标值）
    """
    init_samples = max(5, min(20, n_samples // 5))
    return _bayesian_maximization(
        domain=domain,
        objective_function=objective_function,
        n_queries=n_samples,
        init_samples=init_samples,
        seed=seed,
        verbose=verbose,
        acquisition="ei",            # 默认使用 EI；如需 UCB，请修改为 "ucb"
        candidate_pool_size=2000,
        xi=1e-3,
        kappa=2.0
    )
