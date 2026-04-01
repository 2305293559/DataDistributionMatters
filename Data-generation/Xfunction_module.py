"""
Xfunction_module.py

封装了一组常用的高维测试函数（benchmark / synthetic objective functions），
供优化、采样、测试等代码调用。

主要特性：
- 所有函数均接受 numpy 数组作为输入，支持一维向量或二维批量输入。
- 返回值为标量或一维数组（与输入批量大小一致）。
- 提供 `get_high_dimensional_function(name_or_index)` 工厂函数快速获取函数对象。
- 额外暴露 `AVAILABLE_FUNCTIONS` 列表，方便自动发现。

使用示例：
>>> from Xfunction_module import get_high_dimensional_function
>>> f = get_high_dimensional_function(1)
>>> f(np.array([1.0, 2.0, 3.0]))

作者：自动封装（为原始 Xfunction.py 进行整理与修复）
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Optional, Sequence, Union, Dict, Any

ArrayLike = Union[Sequence[float], np.ndarray]


def _as_np_array(x: ArrayLike) -> np.ndarray:
    """把输入规范为 np.ndarray，并确保浮点类型与形状。(支持一维向量或二维批量输入)

    返回形状 (n_samples, n_dims) 或 (1, n_dims) 的数组
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    elif x.ndim == 0:
        x = x.reshape(1, 1)
    return x


def _squeeze_result(y: np.ndarray) -> Union[float, np.ndarray]:
    y = np.asarray(y)
    if y.size == 1:
        return float(y.ravel()[0])
    return y.ravel()


# ---------------------------
# 基本函数
# ---------------------------

def S_f1(x: ArrayLike) -> Union[float, np.ndarray]:
    """平方和函数：f(x)=sum(x_i^2)"""
    x = _as_np_array(x)
    return _squeeze_result(np.sum(x ** 2, axis=1))


def S_f2(x: ArrayLike) -> Union[float, np.ndarray]:
    """带乘积项的绝对值和：f(x)=sum(|x_i|)+prod(|x_i|)"""
    x = _as_np_array(x)
    return _squeeze_result(np.sum(np.abs(x), axis=1) + np.prod(np.abs(x), axis=1))


def S_f3(x: ArrayLike) -> Union[float, np.ndarray]:
    """累积和平方：f(x)=sum( (sum_{j<=i} x_j)^2 )"""
    x = _as_np_array(x)
    n = x.shape[1]
    # 使用向量化的上三角矩阵累加
    cumsum = np.cumsum(x, axis=1)
    return _squeeze_result(np.sum(cumsum ** 2, axis=1))


def S_f4(x: ArrayLike) -> Union[float, np.ndarray]:
    """无穷范数：f(x)=max(|x_i|)"""
    x = _as_np_array(x)
    return _squeeze_result(np.max(np.abs(x), axis=1))


def S_f5(x: ArrayLike) -> Union[float, np.ndarray]:
    """Rosenbrock（通用 n 维）"""
    x = _as_np_array(x)
    xi = x[:, :-1]
    xnext = x[:, 1:]
    return _squeeze_result(np.sum(100.0 * (xnext - xi ** 2) ** 2 + (xi - 1) ** 2, axis=1))


def S_f6(x: ArrayLike) -> Union[float, np.ndarray]:
    """偏移平方和：f(x)=sum((|x_i|+0.5)^2)"""
    x = _as_np_array(x)
    return _squeeze_result(np.sum(np.square(np.abs(x) + 0.5), axis=1))


def S_f7(x: ArrayLike, noise: bool = False, noise_scale: float = 1e-6, seed: Optional[int] = None) -> Union[float, np.ndarray]:
    """多项式与可选随机噪声。默认无噪声，使函数确定性。

    注意：如果需要确定的伪随机噪声，可传入 seed。
    """
    x = _as_np_array(x)
    base = np.sum(np.arange(1, x.shape[1] + 1) * (x ** 4), axis=1)
    if not noise:
        return _squeeze_result(base)
    rng = np.random.default_rng(seed)
    return _squeeze_result(base + rng.random(size=base.shape) * float(noise_scale))


def S_f8(x: ArrayLike) -> Union[float, np.ndarray]:
    """Schwefel-like: f(x)=sum(-x*sin(sqrt(|x|)))"""
    x = _as_np_array(x)
    return _squeeze_result(np.sum(-x * np.sin(np.sqrt(np.abs(x))), axis=1))


def S_f9(x: ArrayLike) -> Union[float, np.ndarray]:
    """Rastrigin 函数"""
    x = _as_np_array(x)
    n = x.shape[1]
    return _squeeze_result(np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x), axis=1) + 10 * n)


def S_f10(x: ArrayLike) -> Union[float, np.ndarray]:
    """Ackley 函数（向量化实现）"""
    x = _as_np_array(x)
    n = x.shape[1]
    sum_sq = np.sum(x ** 2, axis=1)
    term1 = -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / n))
    term2 = np.exp(np.sum(np.cos(2 * np.pi * x), axis=1) / n)
    return _squeeze_result(term1 - term2 + 20.0 + np.e)


def S_f11(x: ArrayLike) -> Union[float, np.ndarray]:
    """Griewank 函数（向量化）"""
    x = _as_np_array(x)
    n = x.shape[1]
    i = np.arange(1, n + 1)
    sum_term = np.sum(x ** 2, axis=1) / 4000.0
    prod_term = np.prod(np.cos(x / np.sqrt(i)), axis=1)
    return _squeeze_result(sum_term - prod_term + 1.0)


def S_f12(x: ArrayLike, a: float = 10, k: float = 100, m: int = 4) -> Union[float, np.ndarray]:
    """Penalized function (类似于 Penalized function #1)

    参考实现中对 y=(x+1)/4 做了变换并加入了惩罚项。
    """
    x = _as_np_array(x)
    n = x.shape[1]
    y = 1.0 + (x + 1.0) / 4.0
    term = np.sin(np.pi * y[:, 0]) ** 2
    for i in range(n - 1):
        term = term + (y[:, i] - 1.0) ** 2 * (1.0 + 10.0 * (np.sin(np.pi * y[:, i + 1]) ** 2))
    term = term + (y[:, -1] - 1.0) ** 2
    main = (np.pi / n) * term

    # 惩罚项
    abs_x = np.abs(x)
    penalty = np.zeros(x.shape[0])
    mask_high = abs_x > a
    if np.any(mask_high):
        penalty = np.sum(k * np.where(mask_high, (abs_x - a) ** m, 0.0), axis=1)

    return _squeeze_result(main + penalty)


def S_f13(x: ArrayLike, a: float = 5.0, k: float = 100.0, m: int = 4) -> Union[float, np.ndarray]:
    """Penalized function (类似于 Penalized function #2)"""
    x = _as_np_array(x)
    n = x.shape[1]
    if n < 2:
        # 需要至少 2 维
        raise ValueError("S_f13 需要至少 2 维输入")
    x_i = x[:, :-1]
    x_ip1 = x[:, 1:]
    middle_terms = np.sum((x_i - 1.0) ** 2 * (1.0 + np.sin(3.0 * np.pi * x_ip1) ** 2), axis=1)
    first_term = np.sin(3.0 * np.pi * x[:, 0]) ** 2
    last_term = (x[:, -1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * x[:, -1]) ** 2)
    main_function = 0.1 * (first_term + middle_terms + last_term)

    abs_x = np.abs(x)
    penalty = np.sum(k * np.where(abs_x > a, (abs_x - a) ** m, 0.0), axis=1)

    return _squeeze_result(main_function + penalty)


# S_f14: 自定义高阶距阵函数，默认要求 10 维
class _S_f14_Helper:
    def __init__(self, precomputed_points: Optional[np.ndarray] = None, low: float = -50.0, high: float = 50.0, rng_seed: Optional[int] = 42):
        self.low = float(low)
        self.high = float(high)
        self._rng = np.random.default_rng(rng_seed)
        if precomputed_points is not None:
            self.points = np.asarray(precomputed_points, dtype=float)
        else:
            self.points = None

    def __call__(self, x: ArrayLike, num_points: int = 1000) -> Union[float, np.ndarray]:
        x = _as_np_array(x)
        n_dim = x.shape[1]
        if n_dim != 10:
            raise ValueError("S_f14 目前实现为 10 维函数，输入维度需为 10")
        if self.points is None:
            self.points = self._rng.uniform(self.low, self.high, size=(num_points, n_dim))
        # 计算差的 6 次方距离
        diff = x[:, None, :] - self.points[None, :, :]
        dist6 = np.sum(diff ** 6, axis=-1)
        indices = np.arange(1, self.points.shape[0] + 1)[None, :]
        terms = 1.0 / (indices + dist6)
        result = 1.0 / (0.002 + np.sum(terms, axis=1))
        return _squeeze_result(result)


# 公开 S_f14 函数的工厂（使用缓存）
_S_F14_CACHE: Dict[str, _S_f14_Helper] = {}

def S_f14(x: ArrayLike, precomputed_points: Optional[np.ndarray] = None, num_points: int = 1000, low: float = -50.0, high: float = 50.0, rng_seed: Optional[int] = 42) -> Union[float, np.ndarray]:
    """高阶多峰函数（实现要求 10 维）。

    如果传入 precomputed_points，则使用该参考点集；否则在指定区间随机生成 num_points 个参考点并缓存。
    """
    key = f"seed={rng_seed}_low={low}_high={high}_npts={num_points}"
    if key not in _S_F14_CACHE:
        _S_F14_CACHE[key] = _S_f14_Helper(precomputed_points=precomputed_points, low=low, high=high, rng_seed=rng_seed)
        if precomputed_points is not None:
            _S_F14_CACHE[key].points = np.asarray(precomputed_points, dtype=float)
    helper = _S_F14_CACHE[key]
    return helper(x, num_points=num_points)


# Hartmann-10 (修正版)
def S_f15(x: ArrayLike) -> Union[float, np.ndarray]:
    """Hartmann 10 函数（4 项的加权指数和，10 维）"""
    alpha = np.array([
        [3.0, 10.0, 30.0, 3.0, 10.0, 30.0, 3.0, 10.0, 30.0, 3.0],
        [0.1, 10.0, 35.0, 0.1, 10.0, 35.0, 0.1, 10.0, 35.0, 0.1],
        [3.0, 10.0, 30.0, 3.0, 10.0, 30.0, 3.0, 10.0, 30.0, 3.0],
        [0.1, 10.0, 35.0, 0.1, 10.0, 35.0, 0.1, 10.0, 35.0, 0.1]
    ])
    c = np.array([1.0, 1.2, 3.0, 3.2])
    p = np.array([
        [0.131, 0.170, 0.557, 0.012, 0.828, 0.554, 0.373, 0.100, 0.665, 0.038],
        [0.232, 0.413, 0.831, 0.373, 0.100, 0.999, 0.234, 0.141, 0.352, 0.288],
        [0.234, 0.141, 0.352, 0.288, 0.304, 0.665, 0.404, 0.882, 0.873, 0.574],
        [0.109, 0.873, 0.554, 0.038, 0.574, 0.882, 0.131, 0.170, 0.557, 0.012]
    ])
    x = _as_np_array(x)
    if x.shape[1] != 10:
        raise ValueError("S_f15 (Hartmann-10) 要求 10 维输入")
    exponent_sum = np.zeros((x.shape[0], 4))
    for i in range(4):
        diff = x - p[i]
        exponent_sum[:, i] = np.sum(alpha[i] * (diff ** 2), axis=1)
    f = -np.sum(c * np.exp(-exponent_sum), axis=1)
    return _squeeze_result(f)


# Shekel 函数（随机位置版）
def S_f16(x: ArrayLike, m: int = 10, seed: int = 42) -> Union[float, np.ndarray]:
    """Shekel-like 函数：随机生成 m 个峰的位置 a_i 和权重 c_i，返回 -sum(1/(||x-a_i||^2 + c_i))"""
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.0, 10.0, size=(m, 10))
    c = rng.uniform(0.1, 0.7, size=m)
    x = _as_np_array(x)
    result = np.zeros(x.shape[0])
    for i in range(m):
        diff = x - a[i]
        sq_diff = np.sum(diff ** 2, axis=1)
        result -= 1.0 / (sq_diff + c[i])
    return _squeeze_result(result)


# ---------------------------
# 工厂与元数据
# ---------------------------

_AVAILABLE = {
    1: S_f1,
    2: S_f2,
    3: S_f3,
    4: S_f4,
    5: S_f5,
    6: S_f6,
    7: S_f7,
    8: S_f8,
    9: S_f9,
    10: S_f10,
    11: S_f11,
    12: S_f12,
    13: S_f13,
    14: S_f14,
    15: S_f15,
    16: S_f16,
}

# 同时支持字符串键
_AVAILABLE_STR: Dict[str, Callable[..., Any]] = {str(k): v for k, v in _AVAILABLE.items()}

AVAILABLE_FUNCTIONS = list(_AVAILABLE.keys())


def get_high_dimensional_function(name_or_index: Union[int, str]) -> Callable[..., Any]:
    """返回对应的目标函数对象。

    参数可以是整型索引（1..16）或对应的字符串形式（"1".."16"）。
    返回值是可调用对象，直接传入向量或批量向量计算。
    """
    if isinstance(name_or_index, int):
        if name_or_index in _AVAILABLE:
            return _AVAILABLE[name_or_index]
        raise KeyError(f"未找到函数索引: {name_or_index}")
    key = str(name_or_index)
    if key in _AVAILABLE_STR:
        return _AVAILABLE_STR[key]
    # 尝试忽略大小写匹配
    for k, v in _AVAILABLE_STR.items():
        if k.lower() == key.lower():
            return v
    raise KeyError(f"未知函数名或索引: {name_or_index}. 可用: {list(_AVAILABLE.keys())}")


__all__ = [
    "S_f1", "S_f2", "S_f3", "S_f4", "S_f5", "S_f6", "S_f7", "S_f8", "S_f9", "S_f10",
    "S_f11", "S_f12", "S_f13", "S_f14", "S_f15", "S_f16",
    "get_high_dimensional_function", "AVAILABLE_FUNCTIONS"
]


# 简单演示
if __name__ == "__main__":
    print("模块演示：")
    x = np.zeros(10)
    print("S_f1([0,..,0])=", S_f1(x))
    print("S_f10(zeros 10) =", S_f10(x))
    # 调用工厂
    f = get_high_dimensional_function(10)
    print("通过工厂获取 S_f10(zeros 10) =", f(x))
