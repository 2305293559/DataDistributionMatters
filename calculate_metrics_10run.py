import pandas as pd
import numpy as np
import os
import glob
from scipy.spatial.distance import pdist, squareform
from scipy.special import gamma, digamma
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse.csgraph import minimum_spanning_tree


# ==========================================
# 核心计算函数（完全保持不变）
# ==========================================

def calculate_l2_star_discrepancy(X):
    N, d = X.shape
    term1 = (1 / 3) ** d
    prod_term2 = np.prod((1 - X ** 2) / 2, axis=1)
    term2 = (2 / N) * np.sum(prod_term2)
    max_matrix = np.maximum(X[:, np.newaxis, :], X[np.newaxis, :, :])
    prod_term3 = np.prod(1 - max_matrix, axis=2)
    term3 = (1 / (N ** 2)) * np.sum(prod_term3)
    discrepancy_sq = term1 - term2 + term3
    return np.sqrt(max(0, discrepancy_sq))


def calculate_mst_metrics(X):
    n_samples = X.shape[0]
    if n_samples < 2:
        return 0.0, 0.0

    dist_matrix = squareform(pdist(X, metric='euclidean'))
    mst_csr = minimum_spanning_tree(dist_matrix)
    edges = mst_csr.data
    return np.mean(edges), np.std(edges)


def calculate_avg_nn_distance(X, k=5):
    if X.shape[0] < 2:
        return 0.0
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    return np.mean(distances[:, 1])


def calculate_shannon_entropy_histogram(X, bins=5):
    hist, _ = np.histogramdd(X, bins=bins)
    n_samples = X.shape[0]
    if n_samples == 0:
        return 0.0
    probs = hist[hist > 0] / n_samples
    return -np.sum(probs * np.log2(probs))


def calculate_differential_entropy_knn(X, k=5):
    N, d = X.shape
    if N <= k:
        return 0.0

    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    rho = distances[:, k - 1]
    rho = np.where(rho == 0, 1e-10, rho)

    cd = (np.pi ** (d / 2)) / gamma(d / 2 + 1)
    return -digamma(k) + digamma(N) + np.log(cd) + d * np.mean(np.log(rho))


def calculate_log_generalized_variance(X):
    N, d = X.shape
    if N < d:
        return -np.inf

    cov = np.cov(X, rowvar=False)
    sign, logdet = np.linalg.slogdet(cov)
    return logdet if sign > 0 else -999.0


def estimate_lipschitz(X, Y):
    if len(X) < 2:
        return 0.0
    dx = pdist(X)
    dy = pdist(Y.reshape(-1, 1))
    mask = dx > 1e-9
    return np.max(dy[mask] / dx[mask]) if np.any(mask) else 0.0


def calculate_covariance_metrics(X):
    n, d = X.shape
    if d < 2:
        return 0.0, 0.0

    cov = np.cov(X, rowvar=False)
    corr = np.corrcoef(X, rowvar=False)

    mask = ~np.eye(d, dtype=bool)
    eps = np.max(np.abs(cov[mask]))

    corr = np.nan_to_num(corr, nan=0.0)
    rho = np.max(np.abs(corr[mask]))

    return eps, rho


# ==========================================
# 单文件分析（保持不变）
# ==========================================

def analyze_single_file(file_path):
    try:
        data = pd.read_csv(file_path)
        if data.shape[1] < 11:
            return None

        X = data.iloc[:, :10].values
        Y = data.iloc[:, -1].values

        X_norm = MinMaxScaler().fit_transform(X)

        results = {
            "L2_Star_Discrepancy": calculate_l2_star_discrepancy(X_norm),
            "MST_mean": calculate_mst_metrics(X_norm)[0],
            "MST_std": calculate_mst_metrics(X_norm)[1],
            "Shannon_Entropy": calculate_shannon_entropy_histogram(X_norm),
            "Avg_NN_Dist": calculate_avg_nn_distance(X_norm),
            "Differential_Entropy": calculate_differential_entropy_knn(X_norm),
            "Generalized_Variance": calculate_log_generalized_variance(X_norm),
            "Covariance_Error_Metric": calculate_covariance_metrics(X_norm)[0],
            "Max_Pearson_Corr": calculate_covariance_metrics(X_norm)[1],
            "Lipschitz_Est": estimate_lipschitz(X, Y),
            "Sample_Size": X.shape[0],
        }

        return results

    except Exception as e:
        print(f"[Error] {file_path}: {e}")
        return None


# ==========================================
# 处理单个 run
# ==========================================

def process_single_run(run_folder, output_folder):
    run_name = os.path.basename(run_folder)
    run_id = run_name.split("_")[-1]

    optimized_root = os.path.join(run_folder, "optimized_al_samples")
    if not os.path.exists(optimized_root):
        print(f"[Skip] {run_name}: no optimized_al_samples")
        return None

    results = []

    n_folders = sorted(
        [d for d in os.listdir(optimized_root) if d.startswith("n")],
        key=lambda x: int(x[1:])
    )

    for n_folder in n_folders:
        n_value = int(n_folder[1:])
        csv_files = glob.glob(os.path.join(optimized_root, n_folder, "*.csv"))

        for csv_file in csv_files:
            metrics = analyze_single_file(csv_file)
            if metrics:
                row = {
                    "Run": run_name,
                    "Sample_Size_Tag": n_value,
                    "Filename": os.path.basename(csv_file)
                }
                row.update(metrics)
                results.append(row)

    if results:
        df = pd.DataFrame(results)
        out_file = os.path.join(output_folder, f"metrics_result_{run_id}.csv")
        df.to_csv(out_file, index=False)
        print(f"[OK] {run_name} -> {out_file}")
        return True

    return False


# ==========================================
# 批量处理所有 run
# ==========================================

def batch_process_all_runs(experiments_root, output_folder_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, output_folder_name)
    os.makedirs(output_folder, exist_ok=True)

    run_folders = sorted(
        [os.path.join(experiments_root, d) for d in os.listdir(experiments_root)
         if d.startswith("run_")],
        key=lambda x: int(os.path.basename(x).split("_")[-1])
    )

    for run_folder in run_folders:
        process_single_run(run_folder, output_folder)


# ==========================================
# 主入口
# ==========================================

if __name__ == "__main__":
    EXPERIMENTS_ROOT = r"experiments_f9"
    OUTPUT_FOLDER_NAME = "bo_metrics_results_f9"

    batch_process_all_runs(EXPERIMENTS_ROOT, OUTPUT_FOLDER_NAME)
