# 10batch_sampling_processor.py
"""
批量采样处理脚本
----------------------------------------
统一调用所有采样方法，生成不同数据量的CSV文件
- 为主动学习和SLPA导入目标函数
- 为所有采样方法生成带目标函数值的CSV文件
- 生成不同数据量的文件（Sobol使用2的幂次方，其他使用原列表）
- 统一文件名格式：采样方法名_数据量.csv
"""

import numpy as np
import pandas as pd
import sys
import os
import time
from typing import Dict, List, Callable, Any

# 导入所有采样方法
from active_learning_sampling_10d import generate_active_learning_samples
from slpa_sampling_10dim import generate_slpa_samples
from lhs_sampling_10d import generate_lhs_samples
from mc_sampling_10d import generate_mc_samples
from random_sampling_10d import generate_random_samples
from sobel_sampling_10d import sobol_sampling
from possion_sampling_10d import generate_poisson_samples
from entropic_sampling_10d import pure_entropic_sampling

# --------------------- 目标函数定义 ---------------------
def your_objective_function(x: np.ndarray) -> float:
    """
    十维Michalewicz函数
    公式: f(x) = -Σ[sin(x_i) * sin^(2m)(i * x_i² / π)]
    其中 m 是调节参数，通常设为10
    这是一个多峰函数，具有许多局部最小值
    """
    x_array = np.asarray(x)
    m = 10  # 常用参数值
    n = len(x_array)
    
    # 计算Michalewicz函数值
    result = 0.0
    for i in range(n):
        result += np.sin(x_array[i]) * (np.sin((i+1) * x_array[i]**2 / np.pi))**(2*m)
    
    return float(-result)  # 取负号，因为原公式是最小化问题

def simple_objective(x: np.ndarray) -> float:
    """简单目标函数：平方和"""
    return float(np.sum(np.asarray(x) ** 2))

# 选择要使用的目标函数
OBJECTIVE_FUNCTION = your_objective_function  # 使用Michalewicz函数

# --------------------- 配置参数 ---------------------
# Michalewicz函数的典型定义域是 [0, π]
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
# 数据量列表 - 普通采样方法使用原列表，Sobol使用2的幂次方
SAMPLE_SIZES = [20, 50, 80, 100, 200, 300, 500, 1000, 2000, 5000,  10000]
SOBOL_SAMPLE_SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
# 输出目录
OUTPUT_DIR = "batch_sampling_results"

# --------------------- 工具函数 ---------------------
def safe_objective_evaluation(objective_func: Callable, x: np.ndarray) -> float:
    """安全评估目标函数，处理异常情况"""
    try:
        return float(objective_func(x))
    except Exception as e:
        print(f"目标函数评估失败: {e}, 使用默认值0.0")
        return 0.0

def add_objective_values(df: pd.DataFrame, objective_func: Callable) -> pd.DataFrame:
    """为DataFrame添加目标函数值列"""
    X = df[[col for col in df.columns if col.startswith('x')]].values
    y_values = [safe_objective_evaluation(objective_func, x) for x in X]
    df_result = df.copy()
    df_result['y'] = y_values
    return df_result

def ensure_output_dir():
    """确保输出目录存在"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建输出目录: {OUTPUT_DIR}")

# --------------------- 采样方法包装器 ---------------------
def run_active_learning_sampling(n_samples: int, domain: Dict, objective_func: Callable) -> pd.DataFrame:
    """运行主动学习采样"""
    print(f"  运行主动学习采样，数据量: {n_samples}")
    return generate_active_learning_samples(
        domain=domain,
        objective_function=objective_func,
        n_queries=n_samples,
        init_samples=min(5, n_samples // 4),  # 避免初始样本过多
        candidate_pool_size=500,
        seed=None,  # 修改为None
        verbose=False
    )

def run_slpa_sampling(n_samples: int, domain: Dict, objective_func: Callable) -> pd.DataFrame:
    """运行SLPA采样"""
    print(f"  运行SLPA采样，数据量: {n_samples}")
    return generate_slpa_samples(
        domain=domain,
        objective_function=objective_func,
        n_samples=n_samples,
        population_size=max(5, min(20, n_samples // 5)),  # 设置下限
        n_offspring=max(2, min(10, n_samples // 10)),     # 设置下限
        seed=None,  # 修改为None
        verbose=False
    )

def run_lhs_sampling(n_samples: int, domain: Dict, objective_func: Callable) -> pd.DataFrame:
    """运行LHS采样并添加目标函数值"""
    print(f"  运行LHS采样，数据量: {n_samples}")
    df = generate_lhs_samples(
        domain=domain,
        n_samples=n_samples,
        seed=None  # 修改为None
    )
    return add_objective_values(df, objective_func)

def run_mc_sampling(n_samples: int, domain: Dict, objective_func: Callable) -> pd.DataFrame:
    """运行蒙特卡洛采样并添加目标函数值"""
    print(f"  运行蒙特卡洛采样，数据量: {n_samples}")
    df = generate_mc_samples(
        domain=domain,
        n_samples=n_samples,
        seed=None  # 修改为None
    )
    return add_objective_values(df, objective_func)

def run_random_sampling(n_samples: int, domain: Dict, objective_func: Callable) -> pd.DataFrame:
    """运行随机采样并添加目标函数值"""
    print(f"  运行随机采样，数据量: {n_samples}")
    df = generate_random_samples(
        domain=domain,
        n_samples=n_samples,
        seed=None  # 修改为None
    )
    return add_objective_values(df, objective_func)

def run_sobol_sampling(n_samples: int, domain: Dict, objective_func: Callable) -> pd.DataFrame:
    """运行Sobol采样并添加目标函数值"""
    print(f"  运行Sobol采样，数据量: {n_samples}")
    # Sobol采样直接使用2的幂次方数据量
    bounds = [tuple(domain[k]) for k in domain]
    X = sobol_sampling(bounds=bounds, n_samples=n_samples)
    
    # 创建DataFrame
    df = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(len(domain))])
    
    return add_objective_values(df, objective_func)

def run_poisson_sampling(n_samples: int, domain: Dict, objective_func: Callable) -> pd.DataFrame:
    """运行泊松采样并添加目标函数值"""
    print(f"  运行泊松采样，数据量: {n_samples}")
    df = generate_poisson_samples(
        n_samples=n_samples,
        lam=5.0,  # 泊松分布的lambda参数
        domain=domain,
        seed=None  # 修改为None
    )
    return add_objective_values(df, objective_func)

def run_entropic_sampling(n_samples: int, domain: Dict, objective_func: Callable) -> pd.DataFrame:
    """运行熵采样并添加目标函数值"""
    print(f"  运行熵采样，数据量: {n_samples}")
    bounds = [tuple(domain[k]) for k in domain]
    X = pure_entropic_sampling(
        bounds=bounds,
        n_samples=n_samples,
        seed=None,
        n_initial=min(20, max(5, n_samples // 5))  # 设置合理的初始样本数
    )
    
    # 创建DataFrame
    df = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(len(domain))])
    return add_objective_values(df, objective_func)

# --------------------- 主处理函数 ---------------------
def run_batch_sampling():
    """运行批量采样"""
    print("开始批量采样处理...")
    print(f"目标函数: {OBJECTIVE_FUNCTION.__name__}")
    print(f"普通采样数据量: {SAMPLE_SIZES}")
    print(f"Sobol采样数据量: {SOBOL_SAMPLE_SIZES}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"随机种子: None (完全随机)")
    print("-" * 50)
    
    ensure_output_dir()
    
    # 采样方法配置
    sampling_methods = {
        "active_learning": (run_active_learning_sampling, SAMPLE_SIZES),
        "slpa": (run_slpa_sampling, SAMPLE_SIZES), 
        "lhs": (run_lhs_sampling, SAMPLE_SIZES),
        "mc": (run_mc_sampling, SAMPLE_SIZES),
        "random": (run_random_sampling, SAMPLE_SIZES),
        "sobol": (run_sobol_sampling, SOBOL_SAMPLE_SIZES),  # 使用专门的数据量列表
        "poisson": (run_poisson_sampling, SAMPLE_SIZES),
        "entropic": (run_entropic_sampling, SAMPLE_SIZES)
    }
    
    results = {}
    
    for method_name, (sampling_func, sample_sizes) in sampling_methods.items():
        print(f"\n处理采样方法: {method_name}")
        method_results = {}
        
        for n_samples in sample_sizes:
            try:
                # 执行采样
                df = sampling_func(n_samples, DOMAIN, OBJECTIVE_FUNCTION)
                
                # 生成文件名
                filename = f"10dim_{method_name}_{n_samples}_fun1.csv"
                filepath = os.path.join(OUTPUT_DIR, filename)
                
                # 保存文件
                df.to_csv(filepath, index=False)
                
                # 记录结果
                method_results[n_samples] = {
                    'filepath': filepath,
                    'dataframe': df,
                    'success': True
                }
                
                print(f"  ✅ 生成: {filename}")
                
            except Exception as e:
                print(f"  ❌ 失败: 10dim_{method_name}_{n_samples}_fun1.csv - {e}")
                import traceback
                error_details = traceback.format_exc()
                print(f"  详细错误: {error_details}")  # 添加详细错误输出
                method_results[n_samples] = {
                    'filepath': None,
                    'dataframe': None, 
                    'success': False,
                    'error': str(e),
                    'traceback': error_details  # 保存详细错误信息
                }
        
        results[method_name] = method_results
    
    # 生成汇总报告
    generate_summary_report(results)
    
    return results

def generate_summary_report(results: Dict):
    """生成采样结果汇总报告"""
    report_file = os.path.join(OUTPUT_DIR, "sampling_summary.txt")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("批量采样结果汇总报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"目标函数: {OBJECTIVE_FUNCTION.__name__}\n")
        f.write(f"定义域: {DOMAIN}\n")
        f.write(f"普通采样数据量: {SAMPLE_SIZES}\n")
        f.write(f"Sobol采样数据量: {SOBOL_SAMPLE_SIZES}\n")
        f.write(f"随机种子: None (完全随机)\n\n")
        f.write("各方法生成情况:\n")
        f.write("-" * 30 + "\n")
        
        for method_name, method_results in results.items():
            f.write(f"\n{method_name}:\n")
            success_count = sum(1 for r in method_results.values() if r['success'])
            total_count = len(method_results)
            f.write(f"  成功: {success_count}/{total_count}\n")
            
            for n_samples, result in method_results.items():
                status = "✅" if result['success'] else "❌"
                f.write(f"  {n_samples}: {status}\n")
    
    print(f"\n汇总报告已保存: {report_file}")

# --------------------- 主程序入口 ---------------------
if __name__ == "__main__":
    print("批量采样处理器")
    print("=" * 50)
    
    try:
        results = run_batch_sampling()
        
        print("\n" + "=" * 50)
        print("批量采样完成！")
        print(f"所有文件已保存到: {OUTPUT_DIR}/")
        
        # 显示成功统计
        total_success = 0
        total_files = 0
        
        for method_name, method_results in results.items():
            success_count = sum(1 for r in method_results.values() if r['success'])
            total_success += success_count
            total_files += len(method_results)
            print(f"{method_name}: {success_count}/{len(method_results)}")
        
        print(f"\n总体成功率: {total_success}/{total_files} ({total_success/total_files*100:.1f}%)")
        
    except Exception as e:
        print(f"批量采样过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)