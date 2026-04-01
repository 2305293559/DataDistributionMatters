# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 21:40:40 2025

@author: 91278
"""
import numpy as np
import pandas as pd

def function(x, F_num):
    """
    计算目标函数值: y = sum(x^2)
    :param x: 输入的特征向量 (numpy array)
    :return: 标量值 y
    """
    if F_num == 1:
        i_arr = 10 * np.arange(1, x.shape[-1]+1).reshape(1, -1)
        return np.sum(i_arr * x, axis=1)
    elif F_num == 2:
        i_arr = 10 * np.arange(1, x.shape[-1]+1).reshape(1, -1)
        return np.sum(i_arr * x ** 2, axis=1)
    elif F_num == 3:
        return np.sum(100 * (x[:, 1:] - x[:, :-1]**2)**2 + (x[:, :-1] - 1)**2, axis=1)
    elif F_num == 4:
        return np.sum(-x * np.sin(np.sqrt(np.abs(x))), axis=1)
    elif F_num == 5:
        return np.sum(x**2 - 10 * np.cos(np.pi * x), axis=1) + 10 * x.shape[-1]
    elif F_num == 6:
        i_arr = np.arange(1, x.shape[-1] + 1).reshape(1, -1)
        return -np.sum(np.sin(x) * (np.sin(i_arr * x ** 2 / np.pi))**20, axis=1)
    else:
        print(f"计算未知函数 F{F_num}")
        return None
    

def generate_random_samples(F_num, n_samples=100, domain=None):
    """
    在给定定义域内随机采样并生成数据集
    :param n_samples: 采样点数量
    :param domain: 定义域字典
    """
    # 默认 10 维定义域配置
    if domain is None:
        domain = {f"x{i}": (-1, 1) for i in range(1, 11)}

    # 提取特征名称及上下界
    feature_names = list(domain.keys())
    lows = np.array([domain[k][0] for k in feature_names])
    highs = np.array([domain[k][1] for k in feature_names])

    # 1. 使用高效的向量化操作生成随机点: 形状为 (n_samples, n_dims)
    samples_matrix = np.random.uniform(low=lows, high=highs, size=(n_samples, len(feature_names)))

    # 2. 计算每一行的 y 值
    y_values = function(samples_matrix, F_num)

    # 3. 整合数据并转换为 DataFrame
    df = pd.DataFrame(samples_matrix, columns=feature_names)
    df['y'] = y_values

    # 4. 导出为 CSV 文件
    output_filename = f"external_test_F{F_num}.csv"
    df.to_csv(output_filename, index=False)
    
    return output_filename


if __name__ == "__main__":
    # 执行生成任务
    N = 1000  # 设置采样点数量
    f_num = 6  # 控制输入函数
    DOMAIN = None
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
    
    path = generate_random_samples(F_num=f_num, n_samples=N, domain=DOMAIN)
    print(f"数据已成功生成并保存至: {path}")