import os
import numpy as np
import pandas as pd

# ============================================================
# 【用户需要修改的参数（★重点★）】
# ============================================================

# 原始大规模采样数据
INPUT_FILE = "rastrigin_random_1000000samples.csv"

# 输出抽样结果
OUTPUT_FILE = "rastrigin_selected_1000samples.csv"

# 最终抽取的样本总数
DESIRED_N = 1000

# 随机种子（None 表示不固定）
RANDOM_SEED = None

# ------------------ 采样策略配置（只改这里） ------------------
SAMPLING_CONFIG = {"high": 0.00, "low": 1.00, "random": 0.00}
# ------------------------------------------------------------

# ============================================================
# 工具函数
# ============================================================

def find_objective_column(df: pd.DataFrame) -> str:
    """
    自动识别目标函数列（y / f(x) 等）
    """
    candidates = ["y", "f(x)", "fx", "objective", "value"]
    for col in candidates:
        if col in df.columns:
            return col

    # 如果都没找到，默认最后一列是目标值
    last_col = df.columns[-1]
    if pd.api.types.is_numeric_dtype(df[last_col]):
        return last_col

    raise ValueError("无法识别目标函数列，请检查 CSV 文件。")

# ============================================================
# 主逻辑
# ============================================================

def main():

    # ---------- 1. 基本检查 ----------
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"找不到输入文件: {INPUT_FILE}")

    assert abs(sum(SAMPLING_CONFIG.values()) - 1.0) < 1e-8, \
        "SAMPLING_CONFIG 中的比例之和必须等于 1.0"

    # ---------- 2. 读取数据 ----------
    df_all = pd.read_csv(INPUT_FILE)
    total_rows = len(df_all)

    if DESIRED_N <= 0 or DESIRED_N > total_rows:
        raise ValueError("DESIRED_N 设置不合法")

    print(f"已读取数据: {total_rows} 行")

    # ---------- 3. 目标列 ----------
    obj_col = find_objective_column(df_all)
    print(f"使用目标列: {obj_col}")

    # ---------- 4. 分位点划分（15% / 85%） ----------
    q15 = df_all[obj_col].quantile(0.15)
    q85 = df_all[obj_col].quantile(0.85)

    df_low = df_all[df_all[obj_col] <= q15]     # 最小 15%
    df_high = df_all[df_all[obj_col] >= q85]    # 最大 15%

    # ---------- 5. 计算各部分样本数量 ----------
    n_high = int(DESIRED_N * SAMPLING_CONFIG.get("high", 0.0))
    n_low = int(DESIRED_N * SAMPLING_CONFIG.get("low", 0.0))
    n_random = DESIRED_N - n_high - n_low  # 保证总数严格一致

    rng = np.random.RandomState(RANDOM_SEED)

    # ---------- 6. 抽取最大 y 部分 ----------
    if n_high > 0:
        n_high_actual = min(n_high, len(df_high))
        df_high_sel = df_high.sample(
            n=n_high_actual,
            replace=False,
            random_state=rng
        )
    else:
        df_high_sel = df_all.iloc[0:0]
        n_high_actual = 0

    # ---------- 7. 抽取最小 y 部分 ----------
    if n_low > 0:
        n_low_actual = min(n_low, len(df_low))
        df_low_sel = df_low.sample(
            n=n_low_actual,
            replace=False,
            random_state=rng
        )
    else:
        df_low_sel = df_all.iloc[0:0]
        n_low_actual = 0

    # ---------- 8. 剩余数据池（防止重复） ----------
    selected_indices = set(df_high_sel.index) | set(df_low_sel.index)
    df_remaining = df_all.drop(index=selected_indices)

    if n_random > len(df_remaining):
        raise ValueError("剩余数据不足以完成随机抽样")

    df_random_sel = df_remaining.sample(
        n=n_random,
        replace=False,
        random_state=rng
    )

    # ---------- 9. 合并最终结果 ----------
    df_final = pd.concat(
        [df_high_sel, df_low_sel, df_random_sel],
        axis=0
    ).reset_index(drop=True)

    # ---------- 10. 保存 ----------
    df_final.to_csv(OUTPUT_FILE, index=False)

    # ---------- 11. 输出信息 ----------
    print("\n抽样完成 ✔")
    print(f"最终样本数: {len(df_final)}")
    print(f"  最大 y 样本数: {n_high_actual}")
    print(f"  最小 y 样本数: {n_low_actual}")
    print(f"  随机样本数:   {n_random}")
    print(f"结果已保存至: {OUTPUT_FILE}")

# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":
    main()
