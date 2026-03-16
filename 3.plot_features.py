# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 22:37:37 2025

@author: 91278
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, digamma
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')


def calculate_differential_entropy_knn(X, k=5):
    """基于 k-NN 的微分熵估计"""
    N, d = X.shape
    if N <= k: return 0 # 样本太少无法计算
    
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(X)
    distances, _ = nbrs.kneighbors(X)
    rho = distances[:, k-1]
    rho = np.where(rho == 0, 1e-10, rho)
    
    cd = (np.pi ** (d/2)) / gamma(d/2 + 1)
    const_term = -digamma(k) + digamma(N) + np.log(cd)
    sum_log_dist = np.mean(np.log(rho)) * d
    return const_term + sum_log_dist


def visualize_diversity_metrics(X, title=None):
    """可视化展示数据分散性与熵值的关系"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 数据点分布
    if X.shape[1] >= 2:
        axes[0, 0].scatter(X[:, 0], X[:, 1], alpha=0.6, s=20)
        axes[0, 0].set_title(f"Data Distribution (N={X.shape[0]})")
        axes[0, 0].set_xlabel("Dim1")
        axes[0, 0].set_ylabel("Dim2")
    
    # 2. 不同k值的熵估计
    k_values = range(3, min(20, X.shape[0]-1))
    entropy_values = []
    for k in k_values:
        entropy_values.append(calculate_differential_entropy_knn(X, k=k))
    
    axes[0, 1].plot(k_values, entropy_values, 'bo-')
    axes[0, 1].set_title("Differential Entropy vs K Values")
    axes[0, 1].set_xlabel("k value")
    axes[0, 1].set_ylabel("KNN Differential Entropy")
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 最近邻距离分布
    if X.shape[0] > 1:
        nbrs = NearestNeighbors(n_neighbors=2).fit(X)
        distances, _ = nbrs.kneighbors(X)
        nn_distances = distances[:, 1]
        
        axes[1, 0].hist(nn_distances, bins=30, alpha=0.7, density=True)
        axes[1, 0].set_title("NN Distribution")
        axes[1, 0].set_xlabel("Distance")
        axes[1, 0].set_ylabel("rho")
    
    # 4. 维度相关性热图（如果维度不高）
    if X.shape[1] <= 10:
        corr_matrix = np.corrcoef(X.T)
        im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 1].set_title("Correlation between Dims")
        plt.colorbar(im, ax=axes[1, 1])
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


from mpl_toolkits.axes_grid1 import make_axes_locatable
def visualize_tSNE(data):
    """t-SNE降维可视化"""
    
    X_columns = data.columns[:10].tolist()
    y_column = data.columns[-1]
    X = data[X_columns].values
    y = data[y_column].values
    print(f"数据形状: X{X.shape}, y{y.shape}")
    print(f"Y值范围: [{y.min():.3f}, {y.max():.3f}]")
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # t-SNE降维
    print("执行t-SNE降维...")
    tsne = TSNE(
        n_components=2        ,
        random_state=42,
        perplexity=min(30, len(X) // 3),
        n_iter=1000,
        init='pca'
    )
    X_tsne = tsne.fit_transform(X_scaled)
    print(f"t-SNE完成，形状: {X_tsne.shape}")
    
    # 绘制散点图（使用viridis配色）
    fig, ax = plt.subplots(figsize=(12, 10))

    scatter = ax.scatter(
        X_tsne[:, 0], 
        X_tsne[:, 1], 
        c=y, 
        cmap='viridis',
        alpha=0.7,
        s=60,
        edgecolors='white',
        linewidth=0.5
    )

    ax.set_xlabel('t-SNE Dimension 1', fontsize=14)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=14)
    ax.set_title(f't-SNE Visualization of 10D Data (n={len(X)})', fontsize=16, pad=20)
    ax.grid(True, alpha=0.3)
    
    # 创建带直方图的颜色条
    # 使用make_axes_locatable来调整布局
    divider = make_axes_locatable(ax)
    
    # 颜色条轴
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(scatter, cax=cax)
    cbar.set_label(f'{y_column} Value', fontsize=12, rotation=270, labelpad=20)
    
    # 直方图轴
    hist_ax = divider.append_axes("right", size="30%", pad=0.4)
    hist_ax.hist(
        y, 
        bins=30, 
        orientation='horizontal', 
        alpha=0.7, 
        color='steelblue',
        edgecolor='black',
        linewidth=0.5
    )
    
    # 设置直方图轴属性
    hist_ax.set_xlabel('Frequency', fontsize=10)
    hist_ax.set_ylabel(f'{y_column} Value', fontsize=10)
    hist_ax.set_title('Y Value Distribution', fontsize=11, pad=10)
    hist_ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    print("可视化完成!")
    

if __name__ == "__main__":
    
    data = pd.read_csv("D:/sampling/code_10dim/10dim_code/Bayesian sampling/f5/experiments_f5_2.0/run_010/optimized_al_samples/n1000/optimized_al_n1000.csv", header=0)
    # data = pd.read_csv("external_test_fun9.csv", header=0)
    # 分离 X 和 Y
    X_raw = data.iloc[:, :10].values
    Y_raw = data.iloc[:, -1].values

    # 归一化 (用于差异度和熵计算)
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X_raw)
    visualize_diversity_metrics(X_norm)
    
    # t-SNE特征图
    visualize_tSNE(data)
            