import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import base64
import io
from PIL import Image
import time
from datetime import datetime
import matplotlib.font_manager as fm
import platform

# For clustering algorithms
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("--- clustering.py 模块已加载 ---")



# Font setup for visualization
def setup_better_chinese_font():
    """设置更好的中文字体支持"""
    system = platform.system()

    # 字体候选列表 - 按优先级排序
    if system == 'Windows':
        font_candidates = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']
    elif system == 'Darwin':  # macOS
        font_candidates = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Apple LiGothic Medium']
    else:  # Linux 和其他系统
        font_candidates = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'Droid Sans Fallback']

    # 添加通用备选字体
    font_candidates.extend(['DejaVu Sans', 'Arial', 'Helvetica'])

    # 查找第一个可用的字体
    font_found = False
    font_prop = None

    for font_name in font_candidates:
        try:
            font_path = fm.findfont(fm.FontProperties(family=font_name))
            if os.path.exists(font_path) and not font_path.endswith('DejaVuSans.ttf'):
                print(f"使用字体: {font_name}, 路径: {font_path}")
                plt.rcParams['font.family'] = 'sans-serif'
                plt.rcParams['font.sans-serif'] = [font_name] + list(plt.rcParams['font.sans-serif'])
                font_prop = fm.FontProperties(family=font_name)
                font_found = True
                break
        except Exception as e:
            print(f"尝试字体 {font_name} 失败: {e}")

    if not font_found:
        print("警告: 未找到支持中文的字体，将使用系统默认字体")

    # 修复负号显示
    plt.rcParams['axes.unicode_minus'] = False

    # 忽略中文字体缺失的警告
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, message="Glyph.*? missing from current font")

    return font_prop


# 使用改进的字体设置
FONT_PROP = setup_better_chinese_font()

# 设置英文字体
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


# Helper functions for clustering visualization and analysis
def get_current_time():
    """返回当前时间字符串"""
    return datetime.now().isoformat()


def apply_plot_style(ax):
    """应用统一的绘图样式"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.7)
    return ax


def plot_clusters_2d(X, labels, method_name="Clustering", centers=None):
    """Plot clusters in 2D space using PCA if needed"""
    fig, ax = create_figure_with_safe_dimensions(10, 6)
    apply_plot_style(ax)

    try:
        # If dimensions > 2, use PCA to reduce to 2D for visualization
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)
            explained_var = pca.explained_variance_ratio_
        else:
            X_2d = X.copy()
            explained_var = None

        # Get unique clusters and colors
        unique_clusters = np.unique(labels)
        is_noise = -1 in unique_clusters  # Check for DBSCAN noise points

        # Create colormap (excluding gray for noise)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))

        # Plot each cluster
        for i, cluster in enumerate(unique_clusters):
            if cluster == -1:  # Noise points (only for DBSCAN)
                color = (0.7, 0.7, 0.7)  # Gray for noise
                marker = 'x'
                alpha = 0.5
            else:
                color = colors[i]
                marker = 'o'
                alpha = 0.7

            cluster_points = X_2d[labels == cluster]
            ax.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                c=[color],
                marker=marker,
                s=50,
                alpha=alpha,
                label=f'聚类 {cluster}' if cluster != -1 else '异常点'
            )

        # Plot cluster centers for K-means
        if centers is not None and X.shape[1] > 2:
            centers_2d = pca.transform(centers)
            ax.scatter(
                centers_2d[:, 0],
                centers_2d[:, 1],
                c='black',
                marker='*',
                s=200,
                alpha=0.8,
                label='聚类中心'
            )
        elif centers is not None:
            ax.scatter(
                centers[:, 0],
                centers[:, 1],
                c='black',
                marker='*',
                s=200,
                alpha=0.8,
                label='聚类中心'
            )

        # Set title and labels
        title = f"{method_name} 聚类结果"
        if explained_var is not None:
            title += f" (PCA: {explained_var[0]:.2f}, {explained_var[1]:.2f})"

        if FONT_PROP:
            ax.set_title(title, fontproperties=FONT_PROP, fontsize=14, fontweight='bold')
            ax.set_xlabel("维度 1", fontproperties=FONT_PROP, fontsize=12)
            ax.set_ylabel("维度 2", fontproperties=FONT_PROP, fontsize=12)
        else:
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel("维度 1", fontsize=12)
            ax.set_ylabel("维度 2", fontsize=12)

        # Add legend
        legend = ax.legend(loc='best', frameon=True, framealpha=0.85)

        # Apply font to legend if available
        if FONT_PROP:
            for text in legend.get_texts():
                text.set_fontproperties(FONT_PROP)

        plt.tight_layout()
        return fig
    except Exception as e:
        import traceback
        print(f"Error plotting clusters: {e}\n{traceback.format_exc()}")
        ax.text(0.5, 0.5, f'绘制聚类时出错: {e}',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12, color='red',
                fontproperties=FONT_PROP if FONT_PROP else None)
        return fig


def plot_silhouette(X, labels, method_name="Clustering"):
    """Plot silhouette values for all samples"""
    from sklearn.metrics import silhouette_samples

    fig, ax = create_figure_with_safe_dimensions(10, 6)
    apply_plot_style(ax)

    try:
        # Calculate silhouette scores for each sample
        silhouette_vals = silhouette_samples(X, labels)
        clusters = np.unique(labels)
        clusters = clusters[clusters >= 0]  # Remove noise points (cluster = -1)

        y_ticks = []
        y_lower, y_upper = 0, 0

        # Color map
        colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))

        for i, cluster in enumerate(clusters):
            # Get silhouette values for this cluster
            cluster_silhouette_vals = silhouette_vals[labels == cluster]
            cluster_silhouette_vals.sort()

            # Size of this cluster
            cluster_size = cluster_silhouette_vals.shape[0]
            y_upper += cluster_size

            # Plot silhouette values
            color = colors[i % len(colors)]
            ax.barh(
                range(y_lower, y_upper),
                cluster_silhouette_vals,
                height=1.0,
                edgecolor='none',
                color=color,
                alpha=0.8
            )

            # Add cluster label at the middle
            y_ticks.append((y_lower + y_upper) / 2)
            y_lower += cluster_size

        # Calculate average silhouette score
        avg_silhouette = np.mean(silhouette_vals)

        # Add vertical line for average silhouette score
        ax.axvline(x=avg_silhouette, color="red", linestyle="--")
        ax.text(
            avg_silhouette + 0.02,
            y_upper * 0.99,
            f'平均: {avg_silhouette:.3f}',
            color='red',
            fontproperties=FONT_PROP if FONT_PROP else None
        )

        # Set labels and ticks
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'聚类 {cluster}' for cluster in clusters])

        if FONT_PROP:
            ax.set_title(f"{method_name} 轮廓系数分析", fontproperties=FONT_PROP, fontsize=14, fontweight='bold')
            ax.set_xlabel("轮廓系数", fontproperties=FONT_PROP, fontsize=12)
            ax.set_ylabel("聚类", fontproperties=FONT_PROP, fontsize=12)
        else:
            ax.set_title(f"{method_name} 轮廓系数分析", fontsize=14, fontweight='bold')
            ax.set_xlabel("轮廓系数", fontsize=12)
            ax.set_ylabel("聚类", fontsize=12)

        ax.set_xlim([-0.1, 1.0])

        plt.tight_layout()
        return fig
    except Exception as e:
        import traceback
        print(f"Error plotting silhouette: {e}\n{traceback.format_exc()}")
        ax.text(0.5, 0.5, f'绘制轮廓系数时出错: {e}',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12, color='red',
                fontproperties=FONT_PROP if FONT_PROP else None)
        return fig


def plot_elbow(X, k_values, inertias, method_name="K-Means"):
    """Plot elbow method for K-Means clustering"""
    fig, ax = create_figure_with_safe_dimensions(10, 6)
    apply_plot_style(ax)

    try:
        # Plot inertia vs. k values
        ax.plot(k_values, inertias, 'o-', linewidth=2, markersize=8, color='#3498db')

        # Add markers
        for i, (k, inertia) in enumerate(zip(k_values, inertias)):
            ax.text(
                k, inertia + max(inertias) * 0.02,
                f'{inertia:.0f}',
                ha='center',
                fontsize=9,
                fontproperties=FONT_PROP if FONT_PROP else None
            )

        # Set labels
        if FONT_PROP:
            ax.set_title(f"{method_name} 肘部法则", fontproperties=FONT_PROP, fontsize=14, fontweight='bold')
            ax.set_xlabel("聚类数量 (k)", fontproperties=FONT_PROP, fontsize=12)
            ax.set_ylabel("惯性 (Inertia)", fontproperties=FONT_PROP, fontsize=12)
        else:
            ax.set_title(f"{method_name} 肘部法则", fontsize=14, fontweight='bold')
            ax.set_xlabel("聚类数量 (k)", fontsize=12)
            ax.set_ylabel("惯性 (Inertia)", fontsize=12)

        # Set ticks
        ax.set_xticks(k_values)

        plt.tight_layout()
        return fig
    except Exception as e:
        import traceback
        print(f"Error plotting elbow method: {e}\n{traceback.format_exc()}")
        ax.text(0.5, 0.5, f'绘制肘部法则时出错: {e}',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12, color='red',
                fontproperties=FONT_PROP if FONT_PROP else None)
        return fig


def plot_multi_model_comparison(models_data):
    """Plot metrics comparison for multiple clustering models"""
    fig, ax = create_figure_with_safe_dimensions(10, 6)
    apply_plot_style(ax)

    try:
        # Extract data
        model_names = [data['name'] for data in models_data]
        metrics = ['轮廓系数', 'Calinski-Harabasz指数', 'Davies-Bouldin指数']

        metric_values = [
            [data['metrics']['silhouette'] for data in models_data],
            [data['metrics']['calinski_harabasz'] for data in models_data],
            [data['metrics']['davies_bouldin'] for data in models_data]
        ]

        # Set up positions
        bar_width = 0.25
        r1 = np.arange(len(model_names))
        r2 = [x + bar_width for x in r1]
        r3 = [x + bar_width for x in r2]

        # Plot bars
        ax.bar(r1, metric_values[0], width=bar_width, label=metrics[0], color='#3498db')
        ax.bar(r2, metric_values[1], width=bar_width, label=metrics[1], color='#2ecc71')
        ax.bar(r3, metric_values[2], width=bar_width, label=metrics[2], color='#e74c3c')

        # Add metric values as text
        for i, r in enumerate([r1, r2, r3]):
            for j, v in enumerate(metric_values[i]):
                ax.text(
                    r[j], v + 0.05,
                    f'{v:.2f}',
                    ha='center',
                    fontsize=8,
                    fontproperties=FONT_PROP if FONT_PROP else None
                )

        # Set labels
        if FONT_PROP:
            ax.set_title("聚类模型性能对比", fontproperties=FONT_PROP, fontsize=14, fontweight='bold')
            ax.set_xlabel("聚类模型", fontproperties=FONT_PROP, fontsize=12)
            ax.set_ylabel("评估指标值", fontproperties=FONT_PROP, fontsize=12)
        else:
            ax.set_title("聚类模型性能对比", fontsize=14, fontweight='bold')
            ax.set_xlabel("聚类模型", fontsize=12)
            ax.set_ylabel("评估指标值", fontsize=12)

        # Set ticks
        ax.set_xticks([r + bar_width for r in range(len(model_names))])
        ax.set_xticklabels(model_names)

        # Add legend
        legend = ax.legend(loc='upper right')
        if FONT_PROP:
            for text in legend.get_texts():
                text.set_fontproperties(FONT_PROP)

        plt.tight_layout()
        return fig
    except Exception as e:
        import traceback
        print(f"Error plotting model comparison: {e}\n{traceback.format_exc()}")
        ax.text(0.5, 0.5, f'绘制模型对比时出错: {e}',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12, color='red',
                fontproperties=FONT_PROP if FONT_PROP else None)
        return fig


def plot_group_clusters(X, labels, groups, method_name="Clustering"):
    """Plot clusters with group information"""
    try:
        # 如果维度 > 2，使用 PCA 降维至 2D 进行可视化
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)
            explained_var = pca.explained_variance_ratio_
        else:
            X_2d = X.copy()
            explained_var = None

        # 获取唯一的聚类和组
        unique_clusters = np.unique(labels)
        unique_groups = np.unique(groups)

        # 创建带有多个子图的图表
        n_groups = len(unique_groups)

        # 限制最大宽度，防止图像过大
        safe_width = min(6 * n_groups, 16)
        # 降低DPI以确保不超过限制
        safe_dpi = 80

        # 关键修改：添加 squeeze=False 参数，确保始终返回数组
        fig, axes = plt.subplots(1, n_groups, figsize=(safe_width, 5), dpi=safe_dpi, squeeze=False)
        axes = axes.flatten()  # 将 axes 转换为一维数组

        # 不再需要单独处理单组情况
        # if n_groups == 1:
        #     axes = [axes]

        # 聚类的颜色映射
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))

        # 为每个组绘图
        for i, group in enumerate(unique_groups):
            ax = axes[i]
            apply_plot_style(ax)

            # 获取此组的数据
            group_mask = (groups == group)
            group_X = X_2d[group_mask]
            group_labels = labels[group_mask]

            # 计算此组中的聚类计数
            cluster_counts = {cluster: np.sum(group_labels == cluster) for cluster in unique_clusters}

            # 绘制每个聚类
            for j, cluster in enumerate(unique_clusters):
                if cluster == -1:  # 噪声点（仅DBSCAN）
                    color = (0.7, 0.7, 0.7)  # 灰色表示噪声
                    marker = 'x'
                    alpha = 0.5
                else:
                    color = colors[j]
                    marker = 'o'
                    alpha = 0.7

                cluster_points = group_X[group_labels == cluster]
                if len(cluster_points) > 0:
                    ax.scatter(
                        cluster_points[:, 0],
                        cluster_points[:, 1],
                        c=[color],
                        marker=marker,
                        s=50,
                        alpha=alpha,
                        label=f'聚类 {cluster} ({cluster_counts[cluster]})'
                    )

            # 设置标题和标签
            group_title = f"组: {group} ({np.sum(group_mask)}个样本)"

            if FONT_PROP:
                ax.set_title(group_title, fontproperties=FONT_PROP, fontsize=12, fontweight='bold')
                ax.set_xlabel("维度 1", fontproperties=FONT_PROP, fontsize=10)
                ax.set_ylabel("维度 2", fontproperties=FONT_PROP, fontsize=10)
            else:
                ax.set_title(group_title, fontsize=12, fontweight='bold')
                ax.set_xlabel("维度 1", fontsize=10)
                ax.set_ylabel("维度 2", fontsize=10)

            # 添加图例
            legend = ax.legend(loc='best', frameon=True, framealpha=0.85, fontsize=8)
            if FONT_PROP:
                for text in legend.get_texts():
                    text.set_fontproperties(FONT_PROP)

        # 主标题
        main_title = f"{method_name} 聚类结果 (按组)"
        if explained_var is not None:
            main_title += f" (PCA: {explained_var[0]:.2f}, {explained_var[1]:.2f})"

        if FONT_PROP:
            fig.suptitle(main_title, fontproperties=FONT_PROP, fontsize=14, fontweight='bold')
        else:
            fig.suptitle(main_title, fontsize=14, fontweight='bold')

        # 增加 pad 参数以避免布局问题
        plt.tight_layout(rect=[0, 0, 1, 0.95], pad=1.5)
        return fig
    except Exception as e:
        import traceback
        print(f"Error plotting group clusters: {e}\n{traceback.format_exc()}")
        fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
        ax.text(0.5, 0.5, f'绘制组聚类时出错: {e}',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12, color='red',
                fontproperties=FONT_PROP if FONT_PROP else None)
        return fig


# Core clustering functions
def perform_kmeans_clustering(X, n_clusters=3, n_init=10, max_iter=300, random_state=42, use_scaler=True):
    """Perform K-Means clustering"""
    if use_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.copy()

    # Initialize and fit K-Means model
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state
    )
    labels = kmeans.fit_predict(X_scaled)
    centers = kmeans.cluster_centers_

    # Calculate metrics
    metrics = calculate_metrics(X_scaled, labels)

    # Add model-specific metrics
    model_metrics = {
        'inertia': kmeans.inertia_,
        'iterations': kmeans.n_iter_
    }

    return {
        'X': X_scaled,
        'labels': labels,
        'centers': centers,
        'metrics': metrics,
        'model_metrics': model_metrics,
        'method': 'K-Means',
        'params': {
            'n_clusters': n_clusters,
            'n_init': n_init,
            'max_iter': max_iter,
            'random_state': random_state
        }
    }


def perform_dbscan_clustering(X, eps=0.5, min_samples=5, metric='euclidean', algorithm='auto', use_scaler=True):
    """Perform DBSCAN clustering"""
    if use_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.copy()

    # Initialize and fit DBSCAN model
    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric=metric,
        algorithm=algorithm
    )
    labels = dbscan.fit_predict(X_scaled)

    # DBSCAN doesn't have centers
    centers = None

    # Calculate metrics
    metrics = calculate_metrics(X_scaled, labels)

    # No specific model metrics for DBSCAN
    model_metrics = {}

    return {
        'X': X_scaled,
        'labels': labels,
        'centers': centers,
        'metrics': metrics,
        'model_metrics': model_metrics,
        'method': 'DBSCAN',
        'params': {
            'eps': eps,
            'min_samples': min_samples,
            'metric': metric,
            'algorithm': algorithm
        }
    }


def perform_hierarchical_clustering(X, n_clusters=3, linkage='ward', affinity='euclidean',
                                    compute_full_tree='auto', use_scaler=True):
    """Perform Hierarchical clustering"""
    if use_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.copy()

    # For ward linkage, must use euclidean
    params = {
        'n_clusters': n_clusters,
        'linkage': linkage,
        'compute_full_tree': compute_full_tree
    }

    # Only add affinity if not using ward linkage
    if linkage != 'ward':
        params['affinity'] = affinity

    # Initialize and fit Hierarchical model
    hierarchical = AgglomerativeClustering(**params)
    labels = hierarchical.fit_predict(X_scaled)

    # Hierarchical doesn't provide centers
    centers = None

    # Calculate metrics
    metrics = calculate_metrics(X_scaled, labels)

    # No specific model metrics for Hierarchical
    model_metrics = {}

    return {
        'X': X_scaled,
        'labels': labels,
        'centers': centers,
        'metrics': metrics,
        'model_metrics': model_metrics,
        'method': '层次聚类',
        'params': params
    }


def perform_gmm_clustering(X, n_components=3, covariance_type='full', max_iter=100,
                           random_state=42, use_scaler=True):
    """Perform Gaussian Mixture Model clustering"""
    if use_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.copy()

    # Initialize and fit GMM model
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        max_iter=max_iter,
        random_state=random_state
    )
    labels = gmm.fit_predict(X_scaled)
    centers = gmm.means_

    # Calculate metrics
    metrics = calculate_metrics(X_scaled, labels)

    # Add model-specific metrics
    model_metrics = {
        'converged': gmm.converged_,
        'n_iter': gmm.n_iter_
    }

    return {
        'X': X_scaled,
        'labels': labels,
        'centers': centers,
        'metrics': metrics,
        'model_metrics': model_metrics,
        'method': '高斯混合模型',
        'params': {
            'n_components': n_components,
            'covariance_type': covariance_type,
            'max_iter': max_iter,
            'random_state': random_state
        }
    }


def perform_elbow_analysis(X, k_range=(1, 10), n_init=10, max_iter=300, random_state=42, use_scaler=True):
    """Perform elbow method analysis for K-Means"""
    if use_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.copy()

    k_values = list(range(k_range[0], k_range[1] + 1))
    inertias = []
    silhouette_scores = []

    for k in k_values:
        # Fit K-Means with current k
        kmeans = KMeans(
            n_clusters=k,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state
        )
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)

        # Calculate silhouette score if k > 1
        if k > 1:
            try:
                silhouette = silhouette_score(X_scaled, labels)
                silhouette_scores.append(silhouette)
            except Exception as e:
                print(f"Error calculating silhouette score for k={k}: {e}")
                silhouette_scores.append(float('nan'))
        else:
            silhouette_scores.append(float('nan'))

    return {
        'k_values': k_values,
        'inertias': inertias,
        'silhouette_scores': silhouette_scores
    }


def perform_multi_clustering(X, methods_params, use_scaler=True, groups=None):
    """Perform multiple clustering methods comparison"""
    if use_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.copy()

    # Initialize results
    all_results = []

    # Process each method
    for method, params, name in methods_params:
        # Initialize clustering model based on method
        if method == "K-Means":
            result = perform_kmeans_clustering(X_scaled, **params, use_scaler=False)
        elif method == "DBSCAN":
            result = perform_dbscan_clustering(X_scaled, **params, use_scaler=False)
        elif method == "层次聚类":
            result = perform_hierarchical_clustering(X_scaled, **params, use_scaler=False)
        elif method == "高斯混合模型":
            result = perform_gmm_clustering(X_scaled, **params, use_scaler=False)
        else:
            continue

        # Add name to result
        result['name'] = name

        # Group analysis if groups are provided
        if groups is not None:
            result['group_analysis'] = analyze_groups(X_scaled, result['labels'], groups)

        # Add to results
        all_results.append(result)

    return {
        'results': all_results,
        'groups': groups if groups is not None else None
    }


def analyze_groups(X, labels, groups):
    """Analyze clustering results by group"""
    unique_groups = np.unique(groups)
    unique_clusters = np.unique(labels)

    group_analysis = {}

    for group in unique_groups:
        group_mask = (groups == group)
        group_X = X[group_mask]
        group_labels = labels[group_mask]

        # Cluster distribution in this group
        cluster_counts = {int(cluster): int(np.sum(group_labels == cluster))
                          for cluster in unique_clusters}

        # Calculate percentage
        total_samples = len(group_labels)
        cluster_percentages = {cluster: (count / total_samples * 100)
                               for cluster, count in cluster_counts.items()}

        # Calculate metrics for this group if possible
        group_metrics = {}
        try:
            unique_group_labels = np.unique(group_labels)
            if len(unique_group_labels) > 1 and len(unique_group_labels) < len(group_labels):
                group_metrics['silhouette'] = silhouette_score(group_X, group_labels)
            else:
                group_metrics['silhouette'] = float('nan')
        except Exception:
            group_metrics['silhouette'] = float('nan')

        group_analysis[str(group)] = {
            'sample_count': int(np.sum(group_mask)),
            'cluster_counts': cluster_counts,
            'cluster_percentages': cluster_percentages,
            'metrics': group_metrics
        }

    return group_analysis


def calculate_metrics(X, labels):
    """Calculate evaluation metrics for clustering"""
    metrics = {}
    try:
        # Only compute silhouette score if more than one cluster and not all samples in same cluster
        unique_labels = np.unique(labels)
        if len(unique_labels) > 1 and len(unique_labels) < len(labels):
            metrics['silhouette'] = silhouette_score(X, labels)
        else:
            metrics['silhouette'] = float('nan')

        # Only compute CH score if more than one cluster
        if len(unique_labels) > 1:
            metrics['calinski_harabasz'] = calinski_harabasz_score(X, labels)
        else:
            metrics['calinski_harabasz'] = float('nan')

        # Only compute DB score if more than one cluster
        if len(unique_labels) > 1:
            metrics['davies_bouldin'] = davies_bouldin_score(X, labels)
        else:
            metrics['davies_bouldin'] = float('nan')
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        metrics = {
            'silhouette': float('nan'),
            'calinski_harabasz': float('nan'),
            'davies_bouldin': float('nan')
        }

    return metrics


def process_single_file(file_path):
    """Process a single CSV or Excel file"""
    try:
        if file_path.lower().endswith('.csv'):
            data = pd.read_csv(file_path)
        else:
            data = pd.read_excel(file_path)

        # Handle potential mixed-type columns or empty values
        for col in data.select_dtypes(include=['object']).columns:
            try:
                data[col] = pd.to_numeric(data[col], errors='ignore')
            except Exception:
                pass

        # Drop columns that are all NaN
        data.dropna(axis=1, how='all', inplace=True)

        return data, None
    except Exception as e:
        return None, str(e)


def process_folder(folder_path):
    """Process a folder containing subfolders as classes"""
    try:
        # Check if the folder has subfolders
        subfolders = [f for f in os.listdir(folder_path)
                      if os.path.isdir(os.path.join(folder_path, f))]

        if not subfolders:
            return None, "所选文件夹没有包含子文件夹。聚类验证需要每个类别有一个单独的子文件夹。"

        # Initialize data collection
        all_data = []
        labels = []
        file_names = []

        # Process each subfolder
        for subfolder in subfolders:
            subfolder_path = os.path.join(folder_path, subfolder)

            # Get all CSV/Excel files in the subfolder
            files = []
            for ext in ['.csv', '.xlsx', '.xls']:
                files.extend([f for f in os.listdir(subfolder_path)
                              if f.lower().endswith(ext)])

            # Process each file
            for file in files:
                file_path = os.path.join(subfolder_path, file)

                try:
                    # Load data
                    if file.lower().endswith('.csv'):
                        df = pd.read_csv(file_path)
                    else:
                        df = pd.read_excel(file_path)

                    # Skip empty files
                    if df.empty:
                        continue

                    # Handle potential mixed-type columns or empty values
                    for col in df.select_dtypes(include=['object']).columns:
                        try:
                            df[col] = pd.to_numeric(df[col], errors='ignore')
                        except Exception:
                            pass

                    # Drop columns that are all NaN
                    df.dropna(axis=1, how='all', inplace=True)

                    # Drop rows with any NaN
                    df.dropna(inplace=True)

                    # Skip if empty after cleaning
                    if df.empty:
                        continue

                    # Ensure all data is numeric
                    numeric_df = df.select_dtypes(include=['number'])

                    # Skip if no numeric columns
                    if numeric_df.empty:
                        continue

                    # Add to data collection
                    all_data.append(numeric_df)
                    labels.extend([subfolder] * len(numeric_df))
                    file_names.extend([file] * len(numeric_df))

                except Exception as e:
                    print(f"Error processing file {file}: {e}")

        # Combine all data
        if not all_data:
            return None, "未找到有效的数据文件或所有文件处理失败"

        # Ensure all dataframes have the same columns
        common_columns = set.intersection(*[set(df.columns) for df in all_data])
        if not common_columns:
            return None, "文件之间没有共同的数值列，无法合并数据"

        # Filter to common columns and concatenate
        all_data = [df[list(common_columns)] for df in all_data]
        X = pd.concat(all_data, ignore_index=True)
        y = pd.Series(labels, name='label')
        file_paths = pd.Series(file_names, name='file')

        # Combine into final dataframe
        result_df = pd.concat([X, y], axis=1)

        return result_df, file_paths
    except Exception as e:
        return None, str(e)


def export_results_to_csv(results, file_path, data_source_type="file", current_data=None, file_names=None):
    """Export clustering results to CSV file"""
    if not results:
        return False, "没有可导出的结果"

    try:
        # Determine which type of results
        if 'method' in results:  # Single clustering
            method = results['method']
            params = results['params']
            labels = results['labels']
            metrics = results['metrics']

            # Prepare header info
            header_lines = [
                "# 聚类分析结果",
                f"# 分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"# 聚类方法: {method}",
                "# 参数:",
            ]

            for key, value in params.items():
                header_lines.append(f"#   {key}: {value}")

            header_lines.extend([
                "#",
                "# 评估指标:",
                f"#   轮廓系数 (Silhouette): {metrics.get('silhouette', 'N/A')}",
                f"#   Calinski-Harabasz指数: {metrics.get('calinski_harabasz', 'N/A')}",
                f"#   Davies-Bouldin指数: {metrics.get('davies_bouldin', 'N/A')}",
                "#",
                "# 聚类分布:"
            ])

            # Add cluster distribution
            cluster_counts = pd.Series(labels).value_counts().sort_index()
            for cluster, count in cluster_counts.items():
                percent = count / len(labels) * 100
                header_lines.append(f"#   聚类 {cluster}: {count} ({percent:.1f}%)")

            header_lines.append("#")

            # Create dataframe for export
            if data_source_type == "file":
                # For file data, create a dataframe with original data and cluster labels
                export_df = current_data.copy()
                export_df['cluster'] = labels
            else:
                # For folder data, create a dataframe with file names and cluster labels
                export_df = pd.DataFrame({
                    'file': file_names,
                    'label': current_data['label'],
                    'cluster': labels
                })

            # Write to file
            with open(file_path, 'w', encoding='utf-8-sig') as f:
                # Write header information
                for line in header_lines:
                    f.write(line + '\n')

                # Write data
                export_df.to_csv(f, index=False)

            return True, "结果已成功导出"

        elif 'results' in results:  # Multi-clustering
            all_results = results['results']

            # Prepare header info
            header_lines = [
                "# 多模型聚类分析结果",
                f"# 分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"# 模型数量: {len(all_results)}",
                "#"
            ]

            # Add model information
            for i, result in enumerate(all_results):
                method = result['method']
                name = result['name']
                params = result['params']
                metrics = result['metrics']

                header_lines.extend([
                    f"# 模型 {i + 1}: {name} ({method})",
                    f"#   参数: {params}",
                    f"#   轮廓系数: {metrics.get('silhouette', 'N/A')}",
                    f"#   Calinski-Harabasz指数: {metrics.get('calinski_harabasz', 'N/A')}",
                    f"#   Davies-Bouldin指数: {metrics.get('davies_bouldin', 'N/A')}",
                    "#"
                ])

            # Create export dataframe
            if data_source_type == "file":
                # For file data, create a dataframe with original data and all model cluster labels
                export_df = current_data.copy()
                for i, result in enumerate(all_results):
                    export_df[f'cluster_{result["name"]}'] = result['labels']
            else:
                # For folder data, create a dataframe with file names, original labels, and all model cluster labels
                export_df = pd.DataFrame({
                    'file': file_names,
                    'original_label': current_data['label']
                })
                for i, result in enumerate(all_results):
                    export_df[f'cluster_{result["name"]}'] = result['labels']

            # Write to file
            with open(file_path, 'w', encoding='utf-8-sig') as f:
                # Write header information
                for line in header_lines:
                    f.write(line + '\n')

                # Write data
                export_df.to_csv(f, index=False)

            return True, "结果已成功导出"

        elif 'k_values' in results:  # Elbow analysis
            k_values = results['k_values']
            inertias = results['inertias']
            silhouette_scores = results.get('silhouette_scores', [])

            # Prepare header info
            header_lines = [
                "# K值选择分析结果 (肘部法则)",
                f"# 分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "#"
            ]

            # Add elbow point suggestion if possible
            if len(k_values) > 2:
                inertia_diffs = np.diff(inertias)
                elbow_idx = np.argmin(np.diff(inertia_diffs)) + 1
                suggested_k = k_values[elbow_idx]
                header_lines.append(f"# 基于肘部法则推荐的K值: {suggested_k}")

            # Add best silhouette score suggestion if possible
            valid_indices = ~np.isnan(silhouette_scores)
            if any(valid_indices):
                valid_k = np.array(k_values)[valid_indices]
                valid_scores = np.array(silhouette_scores)[valid_indices]
                best_idx = np.argmax(valid_scores)
                best_k = valid_k[best_idx]
                best_score = valid_scores[best_idx]
                header_lines.append(f"# 基于轮廓系数推荐的K值: {best_k} (轮廓系数: {best_score:.3f})")

            header_lines.append("#")

            # Create results dataframe
            export_df = pd.DataFrame({
                'K': k_values,
                'Inertia': inertias
            })

            # Add silhouette scores if available
            if len(silhouette_scores) == len(k_values):
                export_df['Silhouette_Score'] = silhouette_scores

            # Write to file
            with open(file_path, 'w', encoding='utf-8-sig') as f:
                # Write header information
                for line in header_lines:
                    f.write(line + '\n')

                # Write data
                export_df.to_csv(f, index=False)

            return True, "结果已成功导出"

        else:
            return False, "未知的结果格式"

    except Exception as e:
        return False, f"导出失败: {str(e)}"


# Streamlit UI implementation
def create_clustering_ui():
    """Main function to create the Streamlit clustering UI"""
    st.markdown("# 聚类分析")

    # Initialize session state variables if not already set
    if 'clustering_mode' not in st.session_state:
        st.session_state.clustering_mode = 'single'

    if 'current_data' not in st.session_state:
        st.session_state.current_data = None

    if 'column_names' not in st.session_state:
        st.session_state.column_names = []

    if 'selected_input_columns' not in st.session_state:
        st.session_state.selected_input_columns = []

    if 'data_source_type' not in st.session_state:
        st.session_state.data_source_type = 'file'

    if 'file_names' not in st.session_state:
        st.session_state.file_names = None

    if 'has_group_column' not in st.session_state:
        st.session_state.has_group_column = False

    if 'selected_group_column' not in st.session_state:
        st.session_state.selected_group_column = None

    if 'clustering_results' not in st.session_state:
        st.session_state.clustering_results = None

    if 'multi_clustering_results' not in st.session_state:
        st.session_state.multi_clustering_results = None

    if 'elbow_analysis_results' not in st.session_state:
        st.session_state.elbow_analysis_results = None

    # Apply CSS styling
    st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 16px;
            border-radius: 4px 4px 0px 0px;
        }
        .stTabs [aria-selected="true"] {
            background-color: rgba(138, 43, 226, 0.1);
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["1. 聚类模式", "2. 数据导入", "3. 参数设置", "4. 聚类结果"])

    with tab1:
        create_clustering_mode_section()

    with tab2:
        create_data_import_section()

    with tab3:
        create_parameters_section()

    with tab4:
        create_results_section()


def create_clustering_mode_section():
    """Create clustering mode selection section"""
    st.subheader("聚类模式选择")

    # Create radio buttons for mode selection
    mode = st.radio(
        "选择聚类模式",
        ["单一聚类分析", "多模型聚类比较 (最多3个模型)", "K值选择 (肘部法则)"],
        horizontal=True,
        # key="clustering_mode_select_radio" # <--- 修改这里的 key
    )

    # Update session state based on selection
    if mode == "单一聚类分析":
        st.session_state.clustering_mode = "single"
    elif mode == "多模型聚类比较 (最多3个模型)":
        st.session_state.clustering_mode = "multi"
    else:
        st.session_state.clustering_mode = "elbow"

    # Add description based on selected mode
    if st.session_state.clustering_mode == "single":
        st.info("选择单一聚类分析对数据进行聚类，可以查看聚类结果和评估指标。")
    elif st.session_state.clustering_mode == "multi":
        st.info("选择多模型聚类比较可以同时运行多个聚类算法并比较它们的性能。")
    else:
        st.info("选择K值选择分析最佳聚类数量，适用于K-Means等需要预先指定聚类数的算法。")


def create_data_import_section():
    """Create data import section with options for file and folder"""
    st.subheader("数据导入")

    # File upload columns
    col1, col2 = st.columns(2)

    with col1:
        # File uploader
        uploaded_file = st.file_uploader("上传CSV或Excel文件", type=["csv", "xlsx", "xls"])

        if uploaded_file is not None:
            try:
                # Process uploaded file
                if uploaded_file.name.lower().endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                else:
                    data = pd.read_excel(uploaded_file)

                # Clean data
                for col in data.select_dtypes(include=['object']).columns:
                    try:
                        data[col] = pd.to_numeric(data[col], errors='ignore')
                    except:
                        pass

                data.dropna(axis=1, how='all', inplace=True)

                if data.empty:
                    st.error("上传的文件为空或不包含有效数据。")
                else:
                    # Store data in session state
                    st.session_state.current_data = data
                    st.session_state.column_names = list(data.columns)
                    st.session_state.data_source_type = "file"
                    st.session_state.file_names = None

                    # Success message
                    st.success(f"已成功加载: {uploaded_file.name} (包含 {len(data)} 行, {len(data.columns)} 列)")

                    # Clear previous results
                    st.session_state.clustering_results = None
                    st.session_state.multi_clustering_results = None
                    st.session_state.elbow_analysis_results = None
            except Exception as e:
                st.error(f"加载数据时出错: {str(e)}")

    with col2:
        # Folder path input
        folder_path = st.text_input("或输入文件夹路径 (子文件夹作为分类)")

        if folder_path and os.path.isdir(folder_path):
            process_button = st.button("处理文件夹", key="process_folder_button")

            if process_button:
                with st.spinner("正在处理文件夹..."):
                    data, file_paths = process_folder(folder_path)

                    if data is not None:
                        # Store data in session state
                        st.session_state.current_data = data
                        st.session_state.column_names = list(data.columns)
                        st.session_state.data_source_type = "folder"
                        st.session_state.file_names = file_paths

                        # Success message
                        st.success(f"已成功加载文件夹数据: {len(data)} 行, {len(np.unique(data['label']))} 个类别")

                        # Clear previous results
                        st.session_state.clustering_results = None
                        st.session_state.multi_clustering_results = None
                        st.session_state.elbow_analysis_results = None
                    else:
                        st.error(f"处理文件夹时出错: {file_paths}")

    # Group selection (only for file data)
    if st.session_state.current_data is not None and st.session_state.data_source_type == "file":
        st.subheader("分组设置")

        group_option = st.radio(
            "数据分组方式",
            ["作为单一组处理", "使用分组列"],
            horizontal=True,
            key="group_option"
        )

        st.session_state.has_group_column = (group_option == "使用分组列")

        if st.session_state.has_group_column:
            group_column = st.selectbox(
                "选择分组列",
                st.session_state.column_names,
                key="group_column"
            )
            st.session_state.selected_group_column = group_column

    # Column selection
    if st.session_state.current_data is not None:
        st.subheader("特征列选择")

        # Display available columns
        all_columns = st.session_state.column_names.copy()

        # Remove label column for folder data
        if st.session_state.data_source_type == "folder" and "label" in all_columns:
            all_columns.remove("label")

        # Also remove group column if selected
        if st.session_state.has_group_column and st.session_state.selected_group_column in all_columns:
            all_columns.remove(st.session_state.selected_group_column)

        # Multi-select for input columns
        selected_columns = st.multiselect(
            "选择聚类特征列",
            all_columns,
            default=st.session_state.selected_input_columns if st.session_state.selected_input_columns else [],
            key="input_columns"
        )

        # Update session state
        st.session_state.selected_input_columns = selected_columns

        # Display selected columns count
        if selected_columns:
            st.info(f"已选择 {len(selected_columns)} 列作为聚类特征")
        else:
            st.warning("请至少选择一个特征列用于聚类")


def create_parameters_section():
    """Create parameters section for different clustering methods"""
    st.subheader("聚类参数设置")

    # Display different parameters based on clustering mode
    if st.session_state.clustering_mode == "single":
        create_single_method_parameters()
    elif st.session_state.clustering_mode == "multi":
        create_multi_method_parameters()
    else:  # Elbow method
        create_elbow_method_parameters()

    # Run clustering button
    if st.session_state.current_data is not None and st.session_state.selected_input_columns:
        run_button = st.button(
            "运行聚类分析",
            type="primary",
            key="run_clustering"
        )

        if run_button:
            # Call appropriate function based on mode
            if st.session_state.clustering_mode == "single":
                run_single_clustering()
            elif st.session_state.clustering_mode == "multi":
                run_multi_clustering()
            else:
                run_elbow_analysis()


def create_single_method_parameters():
    """Create parameters UI for single clustering method"""
    # Method selection
    method = st.selectbox(
        "聚类方法",
        ["K-Means", "DBSCAN", "层次聚类", "高斯混合模型"],
        key="single_method"
    )

    # Use scaler checkbox
    use_scaler = st.checkbox("使用标准化预处理", value=True, key="single_use_scaler")

    # Parameters based on selected method
    if method == "K-Means":
        col1, col2 = st.columns(2)
        with col1:
            n_clusters = st.slider("聚类数量 (n_clusters)", 2, 20, 3, key="kmeans_n_clusters")
            n_init = st.slider("初始化次数 (n_init)", 1, 30, 10, key="kmeans_n_init")
        with col2:
            max_iter = st.slider("最大迭代次数 (max_iter)", 50, 1000, 300, 50, key="kmeans_max_iter")
            random_state = st.slider("随机种子 (random_state)", 0, 100, 42, key="kmeans_random_state")

    elif method == "DBSCAN":
        col1, col2 = st.columns(2)
        with col1:
            eps = st.slider("邻域距离 (eps)", 0.01, 10.0, 0.5, 0.1, key="dbscan_eps")
            min_samples = st.slider("最小样本数 (min_samples)", 2, 100, 5, key="dbscan_min_samples")
        with col2:
            metric = st.selectbox("距离度量 (metric)", ["euclidean", "manhattan", "chebyshev"], key="dbscan_metric")
            algorithm = st.selectbox("算法 (algorithm)", ["auto", "ball_tree", "kd_tree", "brute"],
                                     key="dbscan_algorithm")

    elif method == "层次聚类":
        col1, col2 = st.columns(2)
        with col1:
            h_n_clusters = st.slider("聚类数量 (n_clusters)", 2, 20, 3, key="hierarchical_n_clusters")
            linkage = st.selectbox("连接方式 (linkage)", ["ward", "complete", "average", "single"],
                                   key="hierarchical_linkage")
        with col2:
            affinity = st.selectbox("相似度度量 (affinity)", ["euclidean", "l1", "l2", "manhattan", "cosine"],
                                    key="hierarchical_affinity")
            compute_full_tree = st.selectbox("计算完整树", ["auto", "True", "False"],
                                             key="hierarchical_compute_full_tree")

    elif method == "高斯混合模型":
        col1, col2 = st.columns(2)
        with col1:
            n_components = st.slider("组件数量 (n_components)", 1, 20, 3, key="gmm_n_components")
            covariance_type = st.selectbox("协方差类型 (covariance_type)", ["full", "tied", "diag", "spherical"],
                                           key="gmm_covariance_type")
        with col2:
            gmm_max_iter = st.slider("最大迭代次数 (max_iter)", 50, 1000, 100, 10, key="gmm_max_iter")
            gmm_random_state = st.slider("随机种子 (random_state)", 0, 100, 42, key="gmm_random_state")


def create_multi_method_parameters():
    """Create parameters UI for multiple clustering methods"""
    # Use scaler checkbox
    use_scaler = st.checkbox("使用标准化预处理", value=True, key="multi_use_scaler")

    # Create 3 columns for methods
    st.subheader("配置多个聚类模型")
    st.caption("可以配置最多3个不同的聚类模型进行比较")

    # Method 1
    st.markdown("#### 模型 1")
    col1, col2, col3 = st.columns(3)
    with col1:
        method1 = st.selectbox("聚类方法", ["K-Means", "DBSCAN", "层次聚类", "高斯混合模型"], key="method1")
    with col2:
        if method1 == "K-Means":
            params1 = {"n_clusters": st.slider("聚类数量", 2, 20, 3, key="m1_kmeans_n_clusters")}
            name1 = st.text_input("模型名称", "K-Means", key="name1")
        elif method1 == "DBSCAN":
            params1 = {
                "eps": st.slider("邻域距离", 0.01, 10.0, 0.5, 0.1, key="m1_dbscan_eps"),
                "min_samples": st.slider("最小样本数", 2, 100, 5, key="m1_dbscan_min_samples")
            }
            name1 = st.text_input("模型名称", "DBSCAN", key="name1")
        elif method1 == "层次聚类":
            params1 = {
                "n_clusters": st.slider("聚类数量", 2, 20, 3, key="m1_hierarchical_n_clusters"),
                "linkage": st.selectbox("连接方式", ["ward", "complete", "average", "single"],
                                        key="m1_hierarchical_linkage")
            }
            name1 = st.text_input("模型名称", "层次聚类", key="name1")
        elif method1 == "高斯混合模型":
            params1 = {"n_components": st.slider("组件数量", 1, 20, 3, key="m1_gmm_n_components")}
            name1 = st.text_input("模型名称", "高斯混合模型", key="name1")

    # Method 2
    st.markdown("#### 模型 2")
    col1, col2, col3 = st.columns(3)
    with col1:
        method2 = st.selectbox("聚类方法", ["K-Means", "DBSCAN", "层次聚类", "高斯混合模型"], index=1, key="method2")
    with col2:
        if method2 == "K-Means":
            params2 = {"n_clusters": st.slider("聚类数量", 2, 20, 3, key="m2_kmeans_n_clusters")}
            name2 = st.text_input("模型名称", "K-Means", key="name2")
        elif method2 == "DBSCAN":
            params2 = {
                "eps": st.slider("邻域距离", 0.01, 10.0, 0.5, 0.1, key="m2_dbscan_eps"),
                "min_samples": st.slider("最小样本数", 2, 100, 5, key="m2_dbscan_min_samples")
            }
            name2 = st.text_input("模型名称", "DBSCAN", key="name2")
        elif method2 == "层次聚类":
            params2 = {
                "n_clusters": st.slider("聚类数量", 2, 20, 3, key="m2_hierarchical_n_clusters"),
                "linkage": st.selectbox("连接方式", ["ward", "complete", "average", "single"],
                                        key="m2_hierarchical_linkage")
            }
            name2 = st.text_input("模型名称", "层次聚类", key="name2")
        elif method2 == "高斯混合模型":
            params2 = {"n_components": st.slider("组件数量", 1, 20, 3, key="m2_gmm_n_components")}
            name2 = st.text_input("模型名称", "高斯混合模型", key="name2")

    # Method 3
    st.markdown("#### 模型 3")
    col1, col2, col3 = st.columns(3)
    with col1:
        method3 = st.selectbox("聚类方法", ["K-Means", "DBSCAN", "层次聚类", "高斯混合模型"], index=2, key="method3")
    with col2:
        if method3 == "K-Means":
            params3 = {"n_clusters": st.slider("聚类数量", 2, 20, 3, key="m3_kmeans_n_clusters")}
            name3 = st.text_input("模型名称", "K-Means", key="name3")
        elif method3 == "DBSCAN":
            params3 = {
                "eps": st.slider("邻域距离", 0.01, 10.0, 0.5, 0.1, key="m3_dbscan_eps"),
                "min_samples": st.slider("最小样本数", 2, 100, 5, key="m3_dbscan_min_samples")
            }
            name3 = st.text_input("模型名称", "DBSCAN", key="name3")
        elif method3 == "层次聚类":
            params3 = {
                "n_clusters": st.slider("聚类数量", 2, 20, 3, key="m3_hierarchical_n_clusters"),
                "linkage": st.selectbox("连接方式", ["ward", "complete", "average", "single"],
                                        key="m3_hierarchical_linkage")
            }
            name3 = st.text_input("模型名称", "层次聚类", key="name3")
        elif method3 == "高斯混合模型":
            params3 = {"n_components": st.slider("组件数量", 1, 20, 3, key="m3_gmm_n_components")}
            name3 = st.text_input("模型名称", "高斯混合模型", key="name3")


def create_elbow_method_parameters():
    """Create parameters UI for elbow method analysis"""
    # Use scaler checkbox
    use_scaler = st.checkbox("使用标准化预处理", value=True, key="elbow_use_scaler")

    # Elbow method parameters
    st.subheader("肘部法则参数")

    col1, col2 = st.columns(2)
    with col1:
        k_min = st.slider("最小K值", 1, 15, 1, key="elbow_k_min")
        n_init = st.slider("初始化次数", 1, 30, 10, key="elbow_n_init")
    with col2:
        k_max = st.slider("最大K值", 2, 20, 10, key="elbow_k_max")
        max_iter = st.slider("最大迭代次数", 50, 1000, 300, 50, key="elbow_max_iter")

    # Validate k_range
    if k_max <= k_min:
        st.error("最大K值必须大于最小K值")


def create_results_section():
    """Create results section to display clustering results"""
    st.subheader("聚类结果")

    # Check if we have results to display
    has_results = False

    if st.session_state.clustering_mode == "single" and st.session_state.clustering_results is not None:
        has_results = True
        display_single_clustering_results()
    elif st.session_state.clustering_mode == "multi" and st.session_state.multi_clustering_results is not None:
        has_results = True
        display_multi_clustering_results()
    elif st.session_state.clustering_mode == "elbow" and st.session_state.elbow_analysis_results is not None:
        has_results = True
        display_elbow_analysis_results()
    else:
        st.info("运行聚类分析后将在此处显示结果")

    # Export button
    if has_results:
        st.download_button(
            label="导出结果到CSV",
            data=get_export_data(),
            file_name="clustering_results.csv",
            mime="text/csv",
            key="export_button"
        )


def display_single_clustering_results():
    """根据不同聚类方法显示适当的可视化结果"""
    results = st.session_state.clustering_results
    method = results['method']

    # 创建两列用于基本信息显示
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 聚类信息")
        params = results['params']
        labels = results['labels']

        # 显示基本信息
        st.markdown(f"**聚类方法:** {method}")
        st.markdown(f"**聚类数量:** {len(np.unique(labels))}")
        st.markdown(f"**样本数量:** {len(labels)}")

        # 参数
        st.markdown("**参数:**")
        for key, value in params.items():
            st.markdown(f"- {key}: {value}")

        # 聚类分布
        st.markdown("**聚类分布:**")
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        distribution_df = pd.DataFrame({
            '聚类': cluster_counts.index,
            '样本数': cluster_counts.values,
            '比例 (%)': (cluster_counts.values / len(labels) * 100).round(2)
        })
        st.dataframe(distribution_df)

    with col2:
        st.markdown("#### 评估指标")
        metrics = results['metrics']
        model_metrics = results.get('model_metrics', {})

        # 创建度量指标数据框
        metrics_data = []

        # 通用指标
        metrics_data.append({
            '指标': '轮廓系数 (Silhouette)',
            '值': f"{metrics.get('silhouette', float('nan')):.4f}",
            '说明': '越高越好 (-1到1)'
        })
        metrics_data.append({
            '指标': 'Calinski-Harabasz指数',
            '值': f"{metrics.get('calinski_harabasz', float('nan')):.1f}",
            '说明': '越高越好'
        })
        metrics_data.append({
            '指标': 'Davies-Bouldin指数',
            '值': f"{metrics.get('davies_bouldin', float('nan')):.4f}",
            '说明': '越低越好'
        })

        # 特定模型指标
        if model_metrics:
            if 'inertia' in model_metrics and method == "K-Means":
                metrics_data.append({
                    '指标': '惯性 (Inertia)',
                    '值': f"{model_metrics['inertia']:.1f}",
                    '说明': '越低越好'
                })
            if 'iterations' in model_metrics:
                metrics_data.append({
                    '指标': '收敛迭代次数',
                    '值': f"{model_metrics['iterations']}",
                    '说明': ''
                })
            if 'converged' in model_metrics and method == "高斯混合模型":
                metrics_data.append({
                    '指标': '是否收敛',
                    '值': f"{model_metrics['converged']}",
                    '说明': ''
                })

        st.dataframe(pd.DataFrame(metrics_data))

    # 基于聚类方法选择可视化类型
    st.markdown("#### 聚类可视化结果")

    # 获取基本数据
    X = results['X']
    labels = results['labels']
    centers = results['centers']
    groups = results.get('groups')

    # 对不同聚类方法显示不同的可视化组合
    if method == "K-Means":
        # K-Means 显示四个标准图表
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)

        with col1:
            st.markdown("##### 聚类散点图")
            fig = plot_clusters_2d(X, labels, method, centers)
            st.pyplot(fig)

        with col2:
            st.markdown("##### 轮廓系数分析")
            unique_labels = np.unique(labels)
            if len(unique_labels) <= 1 or len(unique_labels) >= len(labels):
                st.warning('无法绘制轮廓系数分析：聚类数太少或太多')
            else:
                try:
                    fig = plot_silhouette(X, labels, method)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"轮廓分析绘制失败: {str(e)}")

        with col3:
            st.markdown("##### 聚类分布")
            try:
                fig, ax = create_figure_with_safe_dimensions(10, 6)
                apply_plot_style(ax)

                unique_clusters = np.unique(labels)
                cluster_counts = pd.Series(labels).value_counts().sort_index()

                bars = ax.bar(
                    [str(c) for c in cluster_counts.index],
                    cluster_counts.values,
                    color=plt.cm.tab10(np.linspace(0, 1, len(unique_clusters))),
                    alpha=0.7
                )

                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2., height + 0.1,
                        f'{int(height)}',
                        ha='center', fontsize=9
                    )

                if FONT_PROP:
                    ax.set_title(f"{method} 聚类分布", fontproperties=FONT_PROP, fontsize=12)
                    ax.set_xlabel("聚类", fontproperties=FONT_PROP, fontsize=10)
                    ax.set_ylabel("样本数量", fontproperties=FONT_PROP, fontsize=10)
                else:
                    ax.set_title(f"{method} 聚类分布", fontsize=12)
                    ax.set_xlabel("聚类", fontsize=10)
                    ax.set_ylabel("样本数量", fontsize=10)

                plt.tight_layout(pad=1.2)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"聚类分布绘制失败: {str(e)}")

        with col4:
            st.markdown("##### 评估指标雷达图")
            try:
                metrics = results['metrics']

                fig, ax = create_figure_with_safe_dimensions(8, 8, max_dpi=80)
                ax = plt.subplot(111, polar=True)

                metrics_names = ['轮廓系数', 'Calinski-Harabasz指数', 'Davies-Bouldin指数']
                metrics_values = [
                    metrics.get('silhouette', 0),
                    metrics.get('calinski_harabasz', 0) / 100,  # 缩小CH指数
                    metrics.get('davies_bouldin', 0)
                ]

                max_values = [1, 10, 5]
                metrics_values_normalized = [min(v / m, 1) for v, m in zip(metrics_values, max_values)]

                angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
                angles += angles[:1]  # 闭合图形

                metrics_values_normalized = metrics_values_normalized + metrics_values_normalized[:1]

                ax.plot(angles, metrics_values_normalized, 'o-', linewidth=2)
                ax.fill(angles, metrics_values_normalized, alpha=0.25)

                ax.set_xticks(angles[:-1])

                if FONT_PROP:
                    ax.set_xticklabels(metrics_names, fontproperties=FONT_PROP, fontsize=9)
                    ax.set_title(f"{method} 评估指标", fontproperties=FONT_PROP, fontsize=12)
                else:
                    ax.set_xticklabels(metrics_names, fontsize=9)
                    ax.set_title(f"{method} 评估指标", fontsize=12)

                plt.tight_layout(pad=1.5)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"指标雷达图绘制失败: {str(e)}")

    elif method == "DBSCAN":
        # DBSCAN 专用可视化
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)

        with col1:
            st.markdown("##### 聚类散点图 (含噪声点)")
            fig = plot_clusters_2d(X, labels, method, centers)
            st.pyplot(fig)

        with col2:
            st.markdown("##### 噪声点分析")
            try:
                fig, ax = create_figure_with_safe_dimensions(10, 6)
                apply_plot_style(ax)

                # 计算噪声点和聚类点比例
                noise_mask = (labels == -1)
                noise_count = np.sum(noise_mask)
                cluster_count = len(labels) - noise_count

                # 创建饼图
                sizes = [noise_count, cluster_count]
                labels_pie = ['噪声点', '聚类点']
                colors = ['#d3d3d3', '#3498db']

                # 突出显示噪声点部分
                explode = (0.1, 0)

                ax.pie(
                    sizes, explode=explode, labels=labels_pie, colors=colors,
                    autopct='%1.1f%%', shadow=True, startangle=90
                )
                ax.axis('equal')  # 确保饼图是圆形

                if FONT_PROP:
                    ax.set_title("噪声点与聚类点比例", fontproperties=FONT_PROP, fontsize=12)
                else:
                    ax.set_title("噪声点与聚类点比例", fontsize=12)

                plt.tight_layout(pad=1.2)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"噪声点分析绘制失败: {str(e)}")

        with col3:
            st.markdown("##### 聚类分布")
            try:
                fig, ax = create_figure_with_safe_dimensions(10, 6)
                apply_plot_style(ax)

                # 只显示有效聚类的分布，排除噪声点
                valid_labels = labels[labels != -1]
                if len(valid_labels) > 0:
                    unique_clusters = np.unique(valid_labels)
                    cluster_counts = pd.Series(valid_labels).value_counts().sort_index()

                    bars = ax.bar(
                        [str(c) for c in cluster_counts.index],
                        cluster_counts.values,
                        color=plt.cm.tab10(np.linspace(0, 1, len(unique_clusters))),
                        alpha=0.7
                    )

                    for bar in bars:
                        height = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width() / 2., height + 0.1,
                            f'{int(height)}',
                            ha='center', fontsize=9
                        )

                    if FONT_PROP:
                        ax.set_title(f"{method} 有效聚类分布", fontproperties=FONT_PROP, fontsize=12)
                        ax.set_xlabel("聚类", fontproperties=FONT_PROP, fontsize=10)
                        ax.set_ylabel("样本数量", fontproperties=FONT_PROP, fontsize=10)
                    else:
                        ax.set_title(f"{method} 有效聚类分布", fontsize=12)
                        ax.set_xlabel("聚类", fontsize=10)
                        ax.set_ylabel("样本数量", fontsize=10)

                    plt.tight_layout(pad=1.2)
                    st.pyplot(fig)
                else:
                    st.warning("没有找到有效的聚类，所有点均被标记为噪声")
            except Exception as e:
                st.error(f"聚类分布绘制失败: {str(e)}")

        with col4:
            st.markdown("##### 密度分布")
            try:
                # 如果维度>2，使用PCA降维
                if X.shape[1] > 2:
                    pca = PCA(n_components=2)
                    X_2d = pca.fit_transform(X)
                else:
                    X_2d = X.copy()

                fig, ax = create_figure_with_safe_dimensions(10, 6)
                apply_plot_style(ax)

                # 创建密度散点图，颜色代表密度
                from sklearn.neighbors import KernelDensity
                kde = KernelDensity(bandwidth=0.5, metric='euclidean')
                kde.fit(X_2d)

                # 计算每个点的密度
                density = np.exp(kde.score_samples(X_2d))

                # 使用颜色映射显示密度
                scatter = ax.scatter(
                    X_2d[:, 0], X_2d[:, 1],
                    c=density, cmap='viridis',
                    s=30, alpha=0.8
                )

                # 添加颜色条
                cbar = plt.colorbar(scatter, ax=ax)
                if FONT_PROP:
                    cbar.set_label("密度", fontproperties=FONT_PROP)
                else:
                    cbar.set_label("密度")

                # 设置标题和标签
                if FONT_PROP:
                    ax.set_title(f"{method} 密度分布", fontproperties=FONT_PROP, fontsize=12)
                    ax.set_xlabel("维度 1", fontproperties=FONT_PROP, fontsize=10)
                    ax.set_ylabel("维度 2", fontproperties=FONT_PROP, fontsize=10)
                else:
                    ax.set_title(f"{method} 密度分布", fontsize=12)
                    ax.set_xlabel("维度 1", fontsize=10)
                    ax.set_ylabel("维度 2", fontsize=10)

                plt.tight_layout(pad=1.2)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"密度分布绘制失败: {str(e)}")

    elif method == "层次聚类":
        # 层次聚类专用可视化
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)

        with col1:
            st.markdown("##### 聚类散点图")
            fig = plot_clusters_2d(X, labels, method, centers)
            st.pyplot(fig)

        with col2:
            st.markdown("##### 聚类分布")
            try:
                fig, ax = create_figure_with_safe_dimensions(10, 6)
                apply_plot_style(ax)

                unique_clusters = np.unique(labels)
                cluster_counts = pd.Series(labels).value_counts().sort_index()

                bars = ax.bar(
                    [str(c) for c in cluster_counts.index],
                    cluster_counts.values,
                    color=plt.cm.tab10(np.linspace(0, 1, len(unique_clusters))),
                    alpha=0.7
                )

                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2., height + 0.1,
                        f'{int(height)}',
                        ha='center', fontsize=9
                    )

                if FONT_PROP:
                    ax.set_title(f"{method} 聚类分布", fontproperties=FONT_PROP, fontsize=12)
                    ax.set_xlabel("聚类", fontproperties=FONT_PROP, fontsize=10)
                    ax.set_ylabel("样本数量", fontproperties=FONT_PROP, fontsize=10)
                else:
                    ax.set_title(f"{method} 聚类分布", fontsize=12)
                    ax.set_xlabel("聚类", fontsize=10)
                    ax.set_ylabel("样本数量", fontsize=10)

                plt.tight_layout(pad=1.2)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"聚类分布绘制失败: {str(e)}")

        with col3:
            st.markdown("##### 类间距离")
            try:
                from scipy.spatial.distance import pdist, squareform

                fig, ax = create_figure_with_safe_dimensions(10, 6)
                apply_plot_style(ax)

                # 计算每个类的中心点
                unique_clusters = np.unique(labels)
                cluster_centers = []

                for cluster in unique_clusters:
                    cluster_data = X[labels == cluster]
                    cluster_center = np.mean(cluster_data, axis=0)
                    cluster_centers.append(cluster_center)

                # 计算类中心之间的距离矩阵
                centers_array = np.array(cluster_centers)
                distances = squareform(pdist(centers_array))

                # 绘制热图
                im = ax.imshow(distances, cmap='YlGnBu')

                # 添加刻度和标签
                ax.set_xticks(np.arange(len(unique_clusters)))
                ax.set_yticks(np.arange(len(unique_clusters)))
                ax.set_xticklabels([f'类 {c}' for c in unique_clusters])
                ax.set_yticklabels([f'类 {c}' for c in unique_clusters])

                # 旋转x轴标签以避免重叠
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

                # 添加颜色条
                cbar = plt.colorbar(im, ax=ax)
                if FONT_PROP:
                    cbar.set_label("距离", fontproperties=FONT_PROP)
                else:
                    cbar.set_label("距离")

                # 设置标题
                if FONT_PROP:
                    ax.set_title("聚类中心间距离矩阵", fontproperties=FONT_PROP, fontsize=12)
                else:
                    ax.set_title("聚类中心间距离矩阵", fontsize=12)

                # 在每个单元格中添加距离值
                for i in range(len(unique_clusters)):
                    for j in range(len(unique_clusters)):
                        if i != j:  # 跳过对角线
                            text = ax.text(j, i, f"{distances[i, j]:.2f}",
                                           ha="center", va="center",
                                           color="black" if distances[i, j] > np.max(distances) / 2 else "white")

                plt.tight_layout(pad=1.2)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"类间距离绘制失败: {str(e)}")
                st.error(str(e))

        with col4:
            st.markdown("##### 评估指标")
            try:
                metrics = results['metrics']

                fig, ax = create_figure_with_safe_dimensions(8, 8, max_dpi=80)
                ax = plt.subplot(111, polar=True)

                metrics_names = ['轮廓系数', 'Calinski-Harabasz指数', 'Davies-Bouldin指数']
                metrics_values = [
                    metrics.get('silhouette', 0),
                    metrics.get('calinski_harabasz', 0) / 100,
                    metrics.get('davies_bouldin', 0)
                ]

                max_values = [1, 10, 5]
                metrics_values_normalized = [min(v / m, 1) for v, m in zip(metrics_values, max_values)]

                angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
                angles += angles[:1]

                metrics_values_normalized = metrics_values_normalized + metrics_values_normalized[:1]

                ax.plot(angles, metrics_values_normalized, 'o-', linewidth=2)
                ax.fill(angles, metrics_values_normalized, alpha=0.25)

                ax.set_xticks(angles[:-1])

                if FONT_PROP:
                    ax.set_xticklabels(metrics_names, fontproperties=FONT_PROP, fontsize=9)
                    ax.set_title(f"{method} 评估指标", fontproperties=FONT_PROP, fontsize=12)
                else:
                    ax.set_xticklabels(metrics_names, fontsize=9)
                    ax.set_title(f"{method} 评估指标", fontsize=12)

                plt.tight_layout(pad=1.5)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"指标雷达图绘制失败: {str(e)}")

    elif method == "高斯混合模型":
        # GMM专用可视化
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)

        with col1:
            st.markdown("##### 聚类散点图")
            fig = plot_clusters_2d(X, labels, method, centers)
            st.pyplot(fig)

        with col2:
            st.markdown("##### 聚类分布")
            try:
                fig, ax = create_figure_with_safe_dimensions(10, 6)
                apply_plot_style(ax)

                unique_clusters = np.unique(labels)
                cluster_counts = pd.Series(labels).value_counts().sort_index()

                bars = ax.bar(
                    [str(c) for c in cluster_counts.index],
                    cluster_counts.values,
                    color=plt.cm.tab10(np.linspace(0, 1, len(unique_clusters))),
                    alpha=0.7
                )

                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2., height + 0.1,
                        f'{int(height)}',
                        ha='center', fontsize=9
                    )

                if FONT_PROP:
                    ax.set_title(f"{method} 聚类分布", fontproperties=FONT_PROP, fontsize=12)
                    ax.set_xlabel("聚类", fontproperties=FONT_PROP, fontsize=10)
                    ax.set_ylabel("样本数量", fontproperties=FONT_PROP, fontsize=10)
                else:
                    ax.set_title(f"{method} 聚类分布", fontsize=12)
                    ax.set_xlabel("聚类", fontsize=10)
                    ax.set_ylabel("样本数量", fontsize=10)

                plt.tight_layout(pad=1.2)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"聚类分布绘制失败: {str(e)}")

        with col3:
            st.markdown("##### 高斯组件可视化")
            try:
                # 如果维度>2，使用PCA降维
                if X.shape[1] > 2:
                    pca = PCA(n_components=2)
                    X_2d = pca.fit_transform(X)
                    if centers is not None:
                        centers_2d = pca.transform(centers)
                    else:
                        centers_2d = None
                else:
                    X_2d = X.copy()
                    centers_2d = centers

                fig, ax = create_figure_with_safe_dimensions(10, 6)
                apply_plot_style(ax)

                # 绘制数据点，颜色表示类别
                unique_clusters = np.unique(labels)
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))

                for i, cluster in enumerate(unique_clusters):
                    cluster_points = X_2d[labels == cluster]
                    ax.scatter(
                        cluster_points[:, 0],
                        cluster_points[:, 1],
                        c=[colors[i]],
                        marker='o',
                        s=30,
                        alpha=0.5,
                        label=f'聚类 {cluster}'
                    )

                # 如果有中心点，绘制中心点
                if centers_2d is not None:
                    # 计算网格点进行等高线可视化
                    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
                    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
                    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                                         np.linspace(y_min, y_max, 100))

                    # 绘制每个组件的等高线
                    for i, (center, color) in enumerate(zip(centers_2d, colors)):
                        # 绘制中心点
                        ax.scatter(
                            center[0], center[1],
                            c='black',
                            marker='*',
                            s=200,
                            alpha=0.8
                        )

                        # 绘制等高线（简化版，只显示平均距离的圆）
                        # 计算当前类别的点到中心的平均距离
                        cluster_points = X_2d[labels == i]
                        if len(cluster_points) > 0:
                            distances = np.sqrt(((cluster_points - center) ** 2).sum(axis=1))
                            avg_distance = np.mean(distances)

                            # 绘制圆形表示分布
                            circle = plt.Circle(
                                (center[0], center[1]),
                                avg_distance,
                                color=color,
                                fill=False,
                                linestyle='-',
                                alpha=0.6
                            )
                            ax.add_artist(circle)

                # 设置标题和标签
                if FONT_PROP:
                    ax.set_title("高斯组件可视化", fontproperties=FONT_PROP, fontsize=12)
                    ax.set_xlabel("维度 1", fontproperties=FONT_PROP, fontsize=10)
                    ax.set_ylabel("维度 2", fontproperties=FONT_PROP, fontsize=10)
                else:
                    ax.set_title("高斯组件可视化", fontsize=12)
                    ax.set_xlabel("维度 1", fontsize=10)
                    ax.set_ylabel("维度 2", fontsize=10)

                # 添加图例
                legend = ax.legend(loc='best', frameon=True, framealpha=0.85, fontsize=8)
                if FONT_PROP:
                    for text in legend.get_texts():
                        text.set_fontproperties(FONT_PROP)

                plt.tight_layout(pad=1.2)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"高斯组件可视化失败: {str(e)}")
                st.error(str(e))

        with col4:
            st.markdown("##### 评估指标")
            try:
                metrics = results['metrics']

                fig, ax = create_figure_with_safe_dimensions(8, 8, max_dpi=80)
                ax = plt.subplot(111, polar=True)

                metrics_names = ['轮廓系数', 'Calinski-Harabasz指数', 'Davies-Bouldin指数']
                metrics_values = [
                    metrics.get('silhouette', 0),
                    metrics.get('calinski_harabasz', 0) / 100,
                    metrics.get('davies_bouldin', 0)
                ]

                max_values = [1, 10, 5]
                metrics_values_normalized = [min(v / m, 1) for v, m in zip(metrics_values, max_values)]

                angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
                angles += angles[:1]

                metrics_values_normalized = metrics_values_normalized + metrics_values_normalized[:1]

                ax.plot(angles, metrics_values_normalized, 'o-', linewidth=2)
                ax.fill(angles, metrics_values_normalized, alpha=0.25)

                ax.set_xticks(angles[:-1])

                if FONT_PROP:
                    ax.set_xticklabels(metrics_names, fontproperties=FONT_PROP, fontsize=9)
                    ax.set_title(f"{method} 评估指标", fontproperties=FONT_PROP, fontsize=12)
                else:
                    ax.set_xticklabels(metrics_names, fontsize=9)
                    ax.set_title(f"{method} 评估指标", fontsize=12)

                plt.tight_layout(pad=1.5)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"指标雷达图绘制失败: {str(e)}")


def display_multi_clustering_results():
    """Display results for multiple clustering methods"""
    results = st.session_state.multi_clustering_results
    all_results = results['results']

    # Create columns for summary
    st.markdown("#### 多模型聚类比较")

    # Create summary table
    summary_data = []
    for result in all_results:
        method = result['method']
        name = result['name']
        labels = result['labels']
        metrics = result['metrics']

        summary_data.append({
            '模型名称': name,
            '聚类方法': method,
            '聚类数量': len(np.unique(labels)),
            '轮廓系数': f"{metrics.get('silhouette', float('nan')):.4f}",
            'CH指数': f"{metrics.get('calinski_harabasz', float('nan')):.1f}",
            'DB指数': f"{metrics.get('davies_bouldin', float('nan')):.4f}"
        })

    st.dataframe(pd.DataFrame(summary_data))

    # Visualization options
    st.markdown("#### 可视化")

    vis_type = st.selectbox(
        "可视化类型",
        ["聚类散点图", "轮廓系数分析", "聚类分布比较", "评估指标比较"],
        index=3,  # Default to metrics comparison
        key="multi_vis_type"
    )

    # Display visualization based on selection
    if vis_type == "聚类散点图":
        # Select model to visualize
        model_names = [result['name'] for result in all_results]
        selected_model = st.selectbox("选择模型", model_names, key="scatter_model")

        # Find selected model
        selected_idx = model_names.index(selected_model)
        result = all_results[selected_idx]

        X = result['X']
        labels = result['labels']
        method = result['method']
        name = result['name']
        centers = result['centers']
        groups = results.get('groups')

        if groups is not None and st.session_state.has_group_column:
            fig = plot_group_clusters(X, labels, groups, f"{name} ({method})")
        else:
            fig = plot_clusters_2d(X, labels, f"{name} ({method})", centers)

        st.pyplot(fig)

    elif vis_type == "轮廓系数分析":
        # Select model to visualize
        valid_models = []
        for result in all_results:
            X = result['X']
            labels = result['labels']
            unique_labels = np.unique(labels)
            if len(unique_labels) > 1 and len(unique_labels) < len(labels):
                valid_models.append(result['name'])

        if not valid_models:
            st.error('无法绘制轮廓系数分析：所有模型的聚类数均不适合分析')
        else:
            selected_model = st.selectbox("选择模型", valid_models, key="silhouette_model")

            # Find selected model
            for result in all_results:
                if result['name'] == selected_model:
                    X = result['X']
                    labels = result['labels']
                    method = result['method']
                    name = result['name']

                    fig = plot_silhouette(X, labels, f"{name} ({method})")
                    st.pyplot(fig)
                    break

    elif vis_type == "聚类分布比较":
        # Collect cluster counts for each model
        all_counts = []
        model_names = []
        all_clusters = set()

        for result in all_results:
            labels = result['labels']
            name = result['name']
            counts = pd.Series(labels).value_counts().to_dict()
            all_counts.append(counts)
            model_names.append(name)
            all_clusters.update(counts.keys())

        # Create figure
        fig, ax = create_figure_with_safe_dimensions(10, 6)
        apply_plot_style(ax)

        # Sort clusters
        sorted_clusters = sorted(all_clusters)
        x = np.arange(len(model_names))

        # Plot stacked bars
        bottoms = np.zeros(len(model_names))

        for i, cluster in enumerate(sorted_clusters):
            heights = []
            for counts in all_counts:
                heights.append(counts.get(cluster, 0))

            color = plt.cm.tab10(i / 10)
            ax.bar(x, heights, bottom=bottoms,
                   label=f'聚类 {cluster}', color=color, alpha=0.7)

            # Update bottoms for next stack
            bottoms += heights

        # Set labels and legend
        if FONT_PROP:
            ax.set_title("聚类分布比较", fontproperties=FONT_PROP, fontsize=14, fontweight='bold')
            ax.set_xlabel("模型", fontproperties=FONT_PROP, fontsize=12)
            ax.set_ylabel("样本数量", fontproperties=FONT_PROP, fontsize=12)
        else:
            ax.set_title("聚类分布比较", fontsize=14, fontweight='bold')
            ax.set_xlabel("模型", fontsize=12)
            ax.set_ylabel("样本数量", fontsize=12)

        ax.set_xticks(x)
        ax.set_xticklabels(model_names)

        # Add legend
        legend = ax.legend(loc='upper right')
        if FONT_PROP:
            for text in legend.get_texts():
                text.set_fontproperties(FONT_PROP)

        plt.tight_layout()
        st.pyplot(fig)

    elif vis_type == "评估指标比较":
        # Extract data for visualization
        model_data = []
        for result in all_results:
            model_data.append({
                'name': result['name'],
                'metrics': result['metrics']
            })

        # Plot metrics comparison
        fig = plot_multi_model_comparison(model_data)
        st.pyplot(fig)


def display_elbow_analysis_results():
    """Display results for elbow method analysis"""
    results = st.session_state.elbow_analysis_results

    if not results or 'k_values' not in results or 'inertias' not in results:
        st.error("肘部法则分析未返回有效结果")
        return

    k_values = results['k_values']
    inertias = results['inertias']
    silhouette_scores = results.get('silhouette_scores', [])

    # Create columns for summary
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 肘部法则分析")

        # Recommendations
        if len(k_values) > 2:
            inertia_diffs = np.diff(inertias)
            elbow_idx = np.argmin(np.diff(inertia_diffs)) + 1
            suggested_k = k_values[elbow_idx]
            st.markdown(f"**基于肘部法则推荐的K值:** {suggested_k}")

        # Best silhouette score
        valid_indices = ~np.isnan(silhouette_scores)
        if any(valid_indices):
            valid_k = np.array(k_values)[valid_indices]
            valid_scores = np.array(silhouette_scores)[valid_indices]
            best_idx = np.argmax(valid_scores)
            best_k = valid_k[best_idx]
            best_score = valid_scores[best_idx]
            st.markdown(f"**基于轮廓系数推荐的K值:** {best_k} (轮廓系数: {best_score:.3f})")

    with col2:
        st.markdown("#### K值与指标关系")

        # Create dataframe with k values and metrics
        data = {'K值': k_values, '惯性 (Inertia)': inertias}

        # Add silhouette scores if available
        if len(silhouette_scores) == len(k_values):
            data['轮廓系数'] = silhouette_scores

        st.dataframe(pd.DataFrame(data))

    # Visualization options
    st.markdown("#### 可视化")

    vis_type = st.selectbox(
        "可视化类型",
        ["肘部法则图", "轮廓系数图"],
        key="elbow_vis_type"
    )

    # Display visualization based on selection
    if vis_type == "肘部法则图":
        fig = plot_elbow(None, k_values, inertias, "K-Means聚类")
        st.pyplot(fig)

    elif vis_type == "轮廓系数图":
        # Check if we have silhouette scores
        if len(silhouette_scores) != len(k_values):
            st.error("无可用的轮廓系数数据")
            return

        # Create figure
        fig, ax = create_figure_with_safe_dimensions(10, 6)
        apply_plot_style(ax)

        # Filter out invalid scores (K=1 will be NaN)
        valid_indices = ~np.isnan(silhouette_scores)
        valid_k = np.array(k_values)[valid_indices]
        valid_scores = np.array(silhouette_scores)[valid_indices]

        if len(valid_k) > 0:
            # Plot silhouette scores
            ax.plot(valid_k, valid_scores, 'o-', linewidth=2, color='#9b59b6')

            # Find best K
            best_idx = np.argmax(valid_scores)
            best_k = valid_k[best_idx]
            best_score = valid_scores[best_idx]

            # Highlight best K
            ax.plot(best_k, best_score, 'o', markersize=10, color='red')
            ax.text(
                best_k, best_score + 0.02,
                f'最佳K = {best_k} (轮廓系数 = {best_score:.3f})',
                ha='center', fontsize=10
            )

            # Set labels
            if FONT_PROP:
                ax.set_title("K-Means 轮廓系数分析", fontproperties=FONT_PROP, fontsize=14, fontweight='bold')
                ax.set_xlabel("聚类数量 (K)", fontproperties=FONT_PROP, fontsize=12)
                ax.set_ylabel("轮廓系数", fontproperties=FONT_PROP, fontsize=12)
            else:
                ax.set_title("K-Means 轮廓系数分析", fontsize=14, fontweight='bold')
                ax.set_xlabel("聚类数量 (K)", fontsize=12)
                ax.set_ylabel("轮廓系数", fontsize=12)

            ax.set_xticks(valid_k)

            # Set y-axis limits for better visualization
            y_min = max(0, min(valid_scores) - 0.05)
            y_max = min(1, max(valid_scores) + 0.1)
            ax.set_ylim(y_min, y_max)

            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.error("无可用的轮廓系数数据")


def run_single_clustering():
    """Run single clustering method"""
    if (st.session_state.current_data is None or
            len(st.session_state.selected_input_columns) == 0):
        st.error("请先导入数据并选择输入特征")
        return

    with st.spinner("正在进行聚类分析..."):
        try:
            # Extract input data
            X = st.session_state.current_data[st.session_state.selected_input_columns].copy()

            # Handle NaN values
            if X.isnull().values.any():
                X = X.dropna()
                if X.empty:
                    st.error("移除缺失值后数据为空，无法进行聚类")
                    return

            # Extract group data if available
            groups = None
            if st.session_state.has_group_column and st.session_state.selected_group_column:
                if st.session_state.selected_group_column in st.session_state.current_data.columns:
                    groups = st.session_state.current_data[st.session_state.selected_group_column].values

            # Get method and parameters
            method = st.session_state.single_method
            use_scaler = st.session_state.single_use_scaler

            # Call appropriate function based on method
            if method == "K-Means":
                n_clusters = st.session_state.kmeans_n_clusters
                n_init = st.session_state.kmeans_n_init
                max_iter = st.session_state.kmeans_max_iter
                random_state = st.session_state.kmeans_random_state

                results = perform_kmeans_clustering(X, n_clusters, n_init, max_iter, random_state, use_scaler)

            elif method == "DBSCAN":
                eps = st.session_state.dbscan_eps
                min_samples = st.session_state.dbscan_min_samples
                metric = st.session_state.dbscan_metric
                algorithm = st.session_state.dbscan_algorithm

                results = perform_dbscan_clustering(X, eps, min_samples, metric, algorithm, use_scaler)

            elif method == "层次聚类":
                n_clusters = st.session_state.hierarchical_n_clusters
                linkage = st.session_state.hierarchical_linkage
                affinity = st.session_state.hierarchical_affinity
                compute_full_tree = st.session_state.hierarchical_compute_full_tree

                # Convert compute_full_tree from string to proper value
                if compute_full_tree == 'True':
                    compute_full_tree = True
                elif compute_full_tree == 'False':
                    compute_full_tree = False

                results = perform_hierarchical_clustering(X, n_clusters, linkage, affinity, compute_full_tree,
                                                          use_scaler)

            elif method == "高斯混合模型":
                n_components = st.session_state.gmm_n_components
                covariance_type = st.session_state.gmm_covariance_type
                max_iter = st.session_state.gmm_max_iter
                random_state = st.session_state.gmm_random_state

                results = perform_gmm_clustering(X, n_components, covariance_type, max_iter, random_state, use_scaler)

                # Add group information if available
            if groups is not None:
                results['groups'] = groups

                # Store results in session state
            st.session_state.clustering_results = results

            # Show success message
            st.success("聚类分析成功完成！")

        except Exception as e:
            import traceback
            st.error(f"聚类分析失败: {str(e)}")
            print(f"Error in clustering: {e}\n{traceback.format_exc()}")

        # 存储当前状态到URL参数
        params = {}
        params['clustering_mode'] = 'single'
        params['method'] = method
        st.query_params.update(**params)


def run_multi_clustering():
    """Run multiple clustering methods comparison"""
    if (st.session_state.current_data is None or
            len(st.session_state.selected_input_columns) == 0):
        st.error("请先导入数据并选择输入特征")
        return

    with st.spinner("正在进行多模型聚类比较..."):
        try:
            # Extract input data
            X = st.session_state.current_data[st.session_state.selected_input_columns].copy()

            # Handle NaN values
            if X.isnull().values.any():
                X = X.dropna()
                if X.empty:
                    st.error("移除缺失值后数据为空，无法进行聚类")
                    return

            # Extract group data if available
            groups = None
            if st.session_state.has_group_column and st.session_state.selected_group_column:
                if st.session_state.selected_group_column in st.session_state.current_data.columns:
                    groups = st.session_state.current_data[st.session_state.selected_group_column].values

            # Get method parameters
            use_scaler = st.session_state.multi_use_scaler

            # Prepare methods and parameters
            methods_params = []

            # Method 1
            method1 = st.session_state.method1
            name1 = st.session_state.name1
            if method1 == "K-Means":
                params1 = {"n_clusters": st.session_state.m1_kmeans_n_clusters}
            elif method1 == "DBSCAN":
                params1 = {
                    "eps": st.session_state.m1_dbscan_eps,
                    "min_samples": st.session_state.m1_dbscan_min_samples
                }
            elif method1 == "层次聚类":
                params1 = {
                    "n_clusters": st.session_state.m1_hierarchical_n_clusters,
                    "linkage": st.session_state.m1_hierarchical_linkage
                }
            elif method1 == "高斯混合模型":
                params1 = {"n_components": st.session_state.m1_gmm_n_components}

            methods_params.append((method1, params1, name1))

            # Method 2
            method2 = st.session_state.method2
            name2 = st.session_state.name2
            if method2 == "K-Means":
                params2 = {"n_clusters": st.session_state.m2_kmeans_n_clusters}
            elif method2 == "DBSCAN":
                params2 = {
                    "eps": st.session_state.m2_dbscan_eps,
                    "min_samples": st.session_state.m2_dbscan_min_samples
                }
            elif method2 == "层次聚类":
                params2 = {
                    "n_clusters": st.session_state.m2_hierarchical_n_clusters,
                    "linkage": st.session_state.m2_hierarchical_linkage
                }
            elif method2 == "高斯混合模型":
                params2 = {"n_components": st.session_state.m2_gmm_n_components}

            methods_params.append((method2, params2, name2))

            # Method 3
            method3 = st.session_state.method3
            name3 = st.session_state.name3
            if method3 == "K-Means":
                params3 = {"n_clusters": st.session_state.m3_kmeans_n_clusters}
            elif method3 == "DBSCAN":
                params3 = {
                    "eps": st.session_state.m3_dbscan_eps,
                    "min_samples": st.session_state.m3_dbscan_min_samples
                }
            elif method3 == "层次聚类":
                params3 = {
                    "n_clusters": st.session_state.m3_hierarchical_n_clusters,
                    "linkage": st.session_state.m3_hierarchical_linkage
                }
            elif method3 == "高斯混合模型":
                params3 = {"n_components": st.session_state.m3_gmm_n_components}

            methods_params.append((method3, params3, name3))

            # Run multi-clustering
            results = perform_multi_clustering(X, methods_params, use_scaler, groups)

            # Store results in session state
            st.session_state.multi_clustering_results = results

            # Show success message
            st.success("多模型聚类比较成功完成！")

        except Exception as e:
            import traceback
            st.error(f"多模型聚类比较失败: {str(e)}")
            print(f"Error in multi-clustering: {e}\n{traceback.format_exc()}")



def run_elbow_analysis():
    """Run elbow method analysis"""
    if (st.session_state.current_data is None or
            len(st.session_state.selected_input_columns) == 0):
        st.error("请先导入数据并选择输入特征")
        return

    with st.spinner("正在进行肘部法则分析..."):
        try:
            # Extract input data
            X = st.session_state.current_data[st.session_state.selected_input_columns].copy()

            # Handle NaN values
            if X.isnull().values.any():
                X = X.dropna()
                if X.empty:
                    st.error("移除缺失值后数据为空，无法进行肘部法则分析")
                    return

            # Get parameters
            k_min = st.session_state.elbow_k_min
            k_max = st.session_state.elbow_k_max
            n_init = st.session_state.elbow_n_init
            max_iter = st.session_state.elbow_max_iter
            use_scaler = st.session_state.elbow_use_scaler

            # Validate k_range
            if k_max <= k_min:
                st.error("最大K值必须大于最小K值")
                return

            # Run elbow analysis
            results = perform_elbow_analysis(X, (k_min, k_max), n_init, max_iter, 42, use_scaler)

            # Store results in session state
            st.session_state.elbow_analysis_results = results

            # Show success message
            st.success("肘部法则分析成功完成！")

        except Exception as e:
            import traceback
            st.error(f"肘部法则分析失败: {str(e)}")
            print(f"Error in elbow analysis: {e}\n{traceback.format_exc()}")

def get_export_data():
    """Prepare data for CSV export"""
    try:
        # Choose results based on mode
        if st.session_state.clustering_mode == "single" and st.session_state.clustering_results is not None:
            results = st.session_state.clustering_results
            mode = "single"
        elif st.session_state.clustering_mode == "multi" and st.session_state.multi_clustering_results is not None:
            results = st.session_state.multi_clustering_results
            mode = "multi"
        elif st.session_state.clustering_mode == "elbow" and st.session_state.elbow_analysis_results is not None:
            results = st.session_state.elbow_analysis_results
            mode = "elbow"
        else:
            return "No results available"

        # Create CSV string
        import io
        import csv

        output = io.StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow(["# 聚类分析结果"])
        writer.writerow([f"# 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
        writer.writerow(["#"])

        if mode == "single":
            method = results['method']
            params = results['params']
            labels = results['labels']
            metrics = results['metrics']

            writer.writerow([f"# 聚类方法: {method}"])
            writer.writerow(["# 参数:"])
            for key, value in params.items():
                writer.writerow([f"#   {key}: {value}"])

            writer.writerow(["#"])
            writer.writerow(["# 评估指标:"])
            writer.writerow([f"#   轮廓系数 (Silhouette): {metrics.get('silhouette', 'N/A')}"])
            writer.writerow([f"#   Calinski-Harabasz指数: {metrics.get('calinski_harabasz', 'N/A')}"])
            writer.writerow([f"#   Davies-Bouldin指数: {metrics.get('davies_bouldin', 'N/A')}"])

            writer.writerow(["#"])
            writer.writerow(["# 聚类分布:"])
            cluster_counts = pd.Series(labels).value_counts().sort_index()
            for cluster, count in cluster_counts.items():
                percent = count / len(labels) * 100
                writer.writerow([f"#   聚类 {cluster}: {count} ({percent:.1f}%)"])

            writer.writerow(["#"])

            # Write data with cluster labels
            data = st.session_state.current_data.copy()
            data['cluster'] = labels

            # Convert to CSV
            data_csv = data.to_csv(index=False)
            output.write(data_csv)

        elif mode == "multi":
            all_results = results['results']

            writer.writerow([f"# 多模型聚类比较"])
            writer.writerow([f"# 模型数量: {len(all_results)}"])
            writer.writerow(["#"])

            for i, result in enumerate(all_results):
                method = result['method']
                name = result['name']
                params = result['params']
                metrics = result['metrics']

                writer.writerow([f"# 模型 {i + 1}: {name} ({method})"])
                writer.writerow([f"#   参数: {params}"])
                writer.writerow([f"#   轮廓系数: {metrics.get('silhouette', 'N/A')}"])
                writer.writerow([f"#   Calinski-Harabasz指数: {metrics.get('calinski_harabasz', 'N/A')}"])
                writer.writerow([f"#   Davies-Bouldin指数: {metrics.get('davies_bouldin', 'N/A')}"])
                writer.writerow(["#"])

            # Create data with all model labels
            data = st.session_state.current_data.copy()
            for result in all_results:
                data[f'cluster_{result["name"]}'] = result['labels']

            # Convert to CSV
            data_csv = data.to_csv(index=False)
            output.write(data_csv)

        elif mode == "elbow":
            k_values = results['k_values']
            inertias = results['inertias']
            silhouette_scores = results.get('silhouette_scores', [])

            writer.writerow([f"# K值选择分析结果 (肘部法则)"])

            # Add elbow point suggestion if possible
            if len(k_values) > 2:
                inertia_diffs = np.diff(inertias)
                elbow_idx = np.argmin(np.diff(inertia_diffs)) + 1
                suggested_k = k_values[elbow_idx]
                writer.writerow([f"# 基于肘部法则推荐的K值: {suggested_k}"])

            # Add best silhouette score suggestion if possible
            valid_indices = ~np.isnan(silhouette_scores)
            if any(valid_indices):
                valid_k = np.array(k_values)[valid_indices]
                valid_scores = np.array(silhouette_scores)[valid_indices]
                best_idx = np.argmax(valid_scores)
                best_k = valid_k[best_idx]
                best_score = valid_scores[best_idx]
                writer.writerow([f"# 基于轮廓系数推荐的K值: {best_k} (轮廓系数: {best_score:.3f})"])

            writer.writerow(["#"])

            # Create results dataframe and convert to CSV
            data = pd.DataFrame({
                'K': k_values,
                'Inertia': inertias
            })

            # Add silhouette scores if available
            if len(silhouette_scores) == len(k_values):
                data['Silhouette_Score'] = silhouette_scores

            data_csv = data.to_csv(index=False)
            output.write(data_csv)

        return output.getvalue()

    except Exception as e:
        import traceback
        print(f"Error preparing export data: {e}\n{traceback.format_exc()}")
        return f"Error preparing export data: {str(e)}"


def add_example_datasets():
    """Add example datasets for users to try clustering"""
    st.sidebar.markdown("## 示例数据集")
    st.sidebar.info("选择一个示例数据集来尝试聚类分析")

    # Example datasets
    datasets = {
        "鸢尾花数据集": "iris",
        "葡萄酒数据集": "wine",
        "乳腺癌数据集": "breast_cancer"
    }

    selected_dataset = st.sidebar.selectbox("选择数据集", list(datasets.keys()))

    load_example = st.sidebar.button("加载示例数据")

    if load_example:
        with st.spinner(f"加载{selected_dataset}..."):
            try:
                from sklearn import datasets

                dataset_name = datasets[selected_dataset]

                if dataset_name == "iris":
                    data = datasets.load_iris()
                    df = pd.DataFrame(data.data, columns=data.feature_names)
                    df['species'] = pd.Series(data.target).map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
                    group_col = 'species'

                elif dataset_name == "wine":
                    data = datasets.load_wine()
                    df = pd.DataFrame(data.data, columns=data.feature_names)
                    df['wine_type'] = pd.Series(data.target).astype(str)
                    group_col = 'wine_type'

                elif dataset_name == "breast_cancer":
                    data = datasets.load_breast_cancer()
                    df = pd.DataFrame(data.data, columns=data.feature_names)
                    df['diagnosis'] = pd.Series(data.target).map({0: 'malignant', 1: 'benign'})
                    group_col = 'diagnosis'

                # Store data in session state
                st.session_state.current_data = df
                st.session_state.column_names = list(df.columns)
                st.session_state.data_source_type = "file"
                st.session_state.file_names = None

                # Set default selected columns (numeric only)
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                st.session_state.selected_input_columns = numeric_cols

                # Set group column
                st.session_state.has_group_column = True
                st.session_state.selected_group_column = group_col

                # Success message
                st.sidebar.success(f"已加载 {selected_dataset}，包含 {len(df)} 行")

                # Rerun to update UI
                st.rerun()

            except Exception as e:
                st.sidebar.error(f"加载示例数据集失败: {str(e)}")

def add_clustering_recommendations():
    """Add recommendations sidebar for clustering parameters"""
    if not st.session_state.current_data is None and len(st.session_state.selected_input_columns) > 0:
        X = st.session_state.current_data[st.session_state.selected_input_columns].copy()

        if X.shape[0] > 0:
            st.sidebar.markdown("## 聚类建议")

            # Sample size recommendations
            sample_size = X.shape[0]
            st.sidebar.markdown(f"**样本数量:** {sample_size}")

            if sample_size < 50:
                st.sidebar.warning("样本数量较少，K-Means 或层次聚类可能更适合")
            elif sample_size > 10000:
                st.sidebar.warning("样本数量较大，DBSCAN 或 K-Means 可能更高效")

            # Feature recommendations
            feature_count = X.shape[1]
            st.sidebar.markdown(f"**特征数量:** {feature_count}")

            if feature_count > 10:
                st.sidebar.info("特征数量较多，可考虑先进行降维处理")

            # K-Means specific recommendations (heuristic: sqrt(n/2))
            suggested_k = max(2, int(np.sqrt(sample_size / 2)))
            suggested_k = min(suggested_k, 20)  # Cap at 20
            st.sidebar.markdown(f"**K-Means建议聚类数:** {suggested_k}")

            # DBSCAN recommendations (heuristic based on feature count)
            suggested_eps = 0.5
            if feature_count > 5:
                suggested_eps = 1.0
            if feature_count > 10:
                suggested_eps = 1.5

            suggested_min_samples = max(2, int(np.log10(sample_size) * 2))

            st.sidebar.markdown(f"**DBSCAN建议参数:**")
            st.sidebar.markdown(f"- eps: {suggested_eps:.1f}")
            st.sidebar.markdown(f"- min_samples: {suggested_min_samples}")

def add_interpretation_guide():
    """Add interpretation guide for clustering results"""
    if ((st.session_state.clustering_mode == "single" and st.session_state.clustering_results is not None) or
            (
                    st.session_state.clustering_mode == "multi" and st.session_state.multi_clustering_results is not None) or
            (
                    st.session_state.clustering_mode == "elbow" and st.session_state.elbow_analysis_results is not None)):

        st.sidebar.markdown("## 结果解读指南")

        st.sidebar.markdown("**评估指标说明:**")
        st.sidebar.markdown("- **轮廓系数 (Silhouette)**: 测量聚类的分离程度，取值-1到1，越高越好")
        st.sidebar.markdown("- **CH指数**: 测量聚类的紧密度和分离度，越高越好")
        st.sidebar.markdown("- **DB指数**: 测量簇内相似度与簇间差异性的比例，越低越好")

        if st.session_state.clustering_mode == "single" and st.session_state.clustering_results is not None:
            method = st.session_state.clustering_results['method']

            if method == "K-Means":
                st.sidebar.markdown("**K-Means解读:**")
                st.sidebar.markdown("- 簇中心代表该簇的平均特征值")
                st.sidebar.markdown("- 惯性值(Inertia)表示所有点到其簇中心的距离平方和")

            elif method == "DBSCAN":
                st.sidebar.markdown("**DBSCAN解读:**")
                st.sidebar.markdown("- 标签为-1的点为噪声点/异常点")
                st.sidebar.markdown("- 簇的数量由算法自动确定，不需要预先指定")

            elif method == "层次聚类":
                st.sidebar.markdown("**层次聚类解读:**")
                st.sidebar.markdown("- 基于点或簇之间的距离递归地合并")
                st.sidebar.markdown("- 连接方式决定了簇之间距离的计算方法")

            elif method == "高斯混合模型":
                st.sidebar.markdown("**高斯混合模型解读:**")
                st.sidebar.markdown("- 将数据建模为多个高斯分布的混合")
                st.sidebar.markdown("- 每个点属于各分布的概率可以计算")


def create_figure_with_safe_dimensions(width_inches, height_inches, max_dpi=80):
    """Creates a figure with dimensions that won't exceed Matplotlib's limits"""
    # Ensure dimensions won't exceed the 2^16 limit
    max_pixels = 65000  # Just under 2^16

    # Calculate DPI that would keep dimensions under the limit
    width_dpi = max_pixels / width_inches
    height_dpi = max_pixels / height_inches

    # Use the smaller of the calculated DPI values to ensure both dimensions are safe
    safe_dpi = min(width_dpi, height_dpi, max_dpi)

    # Create figure with safe DPI
    fig, ax = plt.subplots(figsize=(width_inches, height_inches), dpi=safe_dpi)
    return fig, ax


def advanced_visualization_options():
    """Add advanced visualization options"""
    if ((st.session_state.clustering_mode == "single" and st.session_state.clustering_results is not None) or
            (
                    st.session_state.clustering_mode == "multi" and st.session_state.multi_clustering_results is not None)):

        st.sidebar.markdown("## 高级可视化选项")

        # 3D plotting for 3+ dimensions
        if (st.session_state.current_data is not None and
                len(st.session_state.selected_input_columns) >= 3):

            enable_3d = st.sidebar.checkbox("启用3D可视化", value=False)

            if enable_3d:
                st.sidebar.info("在3D可视化中你可以旋转和缩放视图来查看聚类结果")

                # Selecting dimensions for 3D plot
                dims = st.sidebar.multiselect(
                    "选择三个维度用于3D可视化",
                    st.session_state.selected_input_columns,
                    default=st.session_state.selected_input_columns[:3] if len(
                        st.session_state.selected_input_columns) >= 3 else []
                )

                # TODO: Implement 3D visualization
                # This would involve using Plotly or another library with 3D capabilities
                # For now, just show a message
                if len(dims) == 3:
                    st.sidebar.success("已选择3D维度。在完整实现中，这将显示3D交互式可视化。")
                else:
                    st.sidebar.warning("请选择恰好3个维度用于3D可视化")

def main():
    """Main function to run the Streamlit clustering app"""
    st.set_page_config(
        page_title="AI Predictor Pro - 聚类分析",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Add sidebar logo and title
    st.sidebar.image("https://raw.githubusercontent.com/yourusername/ai-predictor-pro/main/logo.png", width=80,
                     use_column_width=False)
    st.sidebar.title("AI Predictor Pro")
    st.sidebar.markdown("### 聚类分析模块")

    # Add example datasets section
    add_example_datasets()

    # Add clustering recommendations
    add_clustering_recommendations()

    # Add interpretation guide (when results are available)
    add_interpretation_guide()

    # Add advanced visualization options
    advanced_visualization_options()

    # Create main UI
    create_clustering_ui()

    # Add footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='text-align: center; color: #888; font-size: 12px;'>"
        "© 2025 NEU_gong AI Predictor Pro | 版本 1.0.0"
        "</div>",
        unsafe_allow_html=True
    )

def show_clustering_page():
    """Function to be called from main app to show clustering UI"""
    create_clustering_ui()



if __name__ == "__main__":
    main()