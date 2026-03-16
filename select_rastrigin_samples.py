"""
select_rastrigin_samples.py

功能说明
--------
从已有的大规模 Rastrigin 随机采样数据中，按如下规则抽取一个子集：

1. 用户自定义最终抽取的数据总量 DESIRED_N
2. 在全部数据中：
   - 从 y 值最大的 25% 中，随机抽取 DESIRED_N 的 25%
   - 从 y 值最小的 25% 中，随机抽取 DESIRED_N 的 25%
   - 从剩余数据中，随机抽取 DESIRED_N 的 50%
3. 保证：
   - 不重复抽样
   - 最终样本数严格等于 DESIRED_N
   - 结果可复现（通过随机种子）

适用于：
- 优化算法训练集 / 测试集构造
- 代理模型数据筛选
- 高值 / 低值 / 随机混合采样实验
"""

import os
import numpy as np
import pandas as pd

# ============================================================
# 【用户需要修改的参数（重点）】
# ============================================================

# 输入数据文件（你的原始大规模采样数据）
INPUT_FILE = "rastrigin_random_1000000samples.csv"

# 输出数据文件（抽样后的结果）
OUTPUT_FILE = "rastrigin_selected_1000samples.csv"

# 最终你希望抽取的样本总数（★只需要改这里★）
DESIRED_N = 1000

# 随机种子（用于保证结果可复现）
RANDOM_SEED = None

# ============================================================
# 内部工具函数（一般不需要改）
# ============================================================

def find_objective_column(df: pd.DataFrame) -> str:
    """
    自动识别目标函数列（y / f(x) 等）

    优先顺序：
    1. 常见命名：'y', 'f(x)', 'fx', 'objective', 'value'
    2. 如果都不存在，则默认最后一列为目标值（前提是数值型）
    """
    candidates = ["y", "f(x)", "fx", "objective", "value"]

    for col in candidates:
        if col in df.columns:
            return col

    # 如果没有找到常见名字，尝试使用最后一列
    last_col = df.columns[-1]
    if pd.api.types.is_numeric_dtype(df[last_col]):
        return last_col

    raise ValueError(
        "无法自动识别目标函数列，请检查 CSV 列名。"
    )

# ============================================================
# 主逻辑
# ============================================================

def main():
    # ---------- 1. 读取数据 ----------
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"找不到输入文件: {INPUT_FILE}")

    df_all = pd.read_csv(INPUT_FILE)
    total_rows = len(df_all)

    if DESIRED_N <= 0:
        raise ValueError("DESIRED_N 必须是正整数")

    if DESIRED_N > total_rows:
        raise ValueError(
            f"DESIRED_N={DESIRED_N} 大于数据总量 {total_rows}"
        )

    print(f"已读取数据: {total_rows} 行")

    # ---------- 2. 找到目标函数列 ----------
    obj_col = find_objective_column(df_all)
    print(f"使用目标列: {obj_col}")

    # ---------- 3. 计算 25% / 75% 分位点 ----------
    q25 = df_all[obj_col].quantile(0.25)
    q75 = df_all[obj_col].quantile(0.75)

    # 根据 y 值划分数据区间
    df_low = df_all[df_all[obj_col] <= q25]      # 最小 25%
    df_high = df_all[df_all[obj_col] >= q75]     # 最大 25%
    df_middle = df_all[
        (df_all[obj_col] > q25) & (df_all[obj_col] < q75)
    ]

    # ---------- 4. 计算各部分目标抽样数量 ----------
    n_high = int(DESIRED_N * 0.25)     # y 最大部分
    n_low = int(DESIRED_N * 0.25)      # y 最小部分
    n_random = DESIRED_N - n_high - n_low  # 随机部分（保证总数）

    rng = np.random.RandomState(RANDOM_SEED)

    # ---------- 5. 从 y 最大的 25% 中抽样 ----------
    n_high_actual = min(n_high, len(df_high))
    df_high_sel = df_high.sample(
        n=n_high_actual,
        replace=False,
        random_state=rng
    )

    # ---------- 6. 从 y 最小的 25% 中抽样 ----------
    n_low_actual = min(n_low, len(df_low))
    df_low_sel = df_low.sample(
        n=n_low_actual,
        replace=False,
        random_state=rng
    )

    # ---------- 7. 剩余数据池（避免重复） ----------
    selected_indices = set(df_high_sel.index) | set(df_low_sel.index)
    df_remaining = df_all.drop(index=selected_indices)

    # 若高/低区间样本不足，从剩余池补齐
    need_extra = DESIRED_N - (n_high_actual + n_low_actual)

    if need_extra > len(df_remaining):
        raise ValueError("剩余数据不足以补齐所需样本")

    df_random_sel = df_remaining.sample(
        n=need_extra,
        replace=False,
        random_state=rng
    )

    # ---------- 8. 合并最终数据 ----------
    df_final = pd.concat(
        [df_high_sel, df_low_sel, df_random_sel],
        axis=0
    ).reset_index(drop=True)

    # ---------- 9. 保存结果 ----------
    df_final.to_csv(OUTPUT_FILE, index=False)

    # ---------- 10. 输出统计信息 ----------
    print("\n抽样完成！")
    print(f"最终样本数: {len(df_final)}")
    print(f"  - y 最大 25% 区间: {len(df_high_sel)}")
    print(f"  - y 最小 25% 区间: {len(df_low_sel)}")
    print(f"  - 随机抽样部分:   {len(df_random_sel)}")
    print(f"结果已保存至: {OUTPUT_FILE}")

# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":
    main()
