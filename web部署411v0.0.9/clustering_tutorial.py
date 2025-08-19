# -*- coding: utf-8 -*-
"""
Clustering Tutorial Module for Streamlit App
Provides an interactive interface to demonstrate clustering algorithms.
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import traceback # 用于打印详细错误

# --- Import shared functions from the main clustering module ---
# 确保 clustering.py 在同一目录或 Python 路径中
try:
    # 尝试从 clustering 模块导入必要的函数和字体设置
    # 注意：你需要确保你的 clustering.py 中有这些函数或类似的实现
    from clustering import (
        plot_clusters_2d, # 假设这个函数用于绘制2D聚类结果
        apply_plot_style, # 假设这个函数用于应用绘图样式
        create_figure_with_safe_dimensions, # 假设这个函数用于创建安全尺寸的图像
        FONT_PROP # 假设这个对象包含了中文字体属性
    )
    CLUSTERING_MODULE_AVAILABLE = True
    print("成功从 clustering.py 导入绘图函数。")
except ImportError:
    CLUSTERING_MODULE_AVAILABLE = False
    # 如果导入失败，定义占位函数，并在UI中显示错误
    def plot_clusters_2d(*args, **kwargs):
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "错误: 绘图函数\nplot_clusters_2d\n无法从 clustering.py 导入", ha='center', va='center', color='red', fontsize=9)
        return fig
    def apply_plot_style(ax): return ax
    def create_figure_with_safe_dimensions(w, h, dpi=80): return plt.subplots(figsize=(w,h), dpi=dpi)
    FONT_PROP = None
    st.error("错误：无法导入主聚类模块 (clustering.py) 中的绘图函数。教学演示的可视化功能将受限。")

# --- 教学状态初始化 ---
def initialize_tutorial_state():
    """初始化教学模块专用的会话状态变量"""
    defaults = {
        'tut_dataset_name': 'Blobs', 'tut_n_samples': 150, 'tut_n_features': 2,
        'tut_centers': 3, 'tut_cluster_std': 1.0, 'tut_noise': 0.05,
        'tut_factor': 0.5, # 用于 Circles 数据集
        'tut_method': 'K-Means', 'tut_kmeans_k': 3, 'tut_dbscan_eps': 0.5,
        'tut_dbscan_min_samples': 5, 'tut_data_X': None, # 存储特征数据 (numpy array)
        'tut_data_y': None, # 存储真实标签 (用于某些数据集)
        'tut_labels': None, # 存储聚类结果标签
        'tut_centers_result': None, # 存储 K-Means 的中心点
        'tut_scaler': StandardScaler(), # 存储用于数据的标准化器
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- 教学 UI 函数 ---
def show_tutorial_page():
    """创建交互式聚类教学演示的用户界面"""
    initialize_tutorial_state() # 确保教学状态已设置

    st.header("🎓 聚类教学演示")
    st.markdown("""
    欢迎来到聚类教学模块！在这里，你可以：
    1.  选择不同的 **示例数据集** 类型。
    2.  调整生成数据集的 **参数**，观察数据分布的变化。
    3.  选择 **聚类算法**（如 K-Means, DBSCAN）并调整其关键参数。
    4.  运行聚类算法，并在下方 **可视化** 聚类结果。
    5.  查看 **评估指标**（如轮廓系数）和结果解读。

    通过互动操作，直观理解不同算法的特性以及参数变化对聚类效果的影响。
    """)
    st.markdown("---")

    # --- 1. 选择示例数据集 ---
    st.subheader("1. 选择示例数据集")
    dataset_options = {
        "Blobs": "生成清晰的团状数据 (适用于 K-Means)",
        "Moons": "生成两个交织的半圆形数据 (适用于 DBSCAN)",
        "Circles": "生成两个同心圆数据 (适用于 DBSCAN)"
    }
    st.session_state.tut_dataset_name = st.selectbox(
        "选择数据集类型:",
        options=list(dataset_options.keys()),
        format_func=lambda x: f"{x} - {dataset_options[x]}",
        key="tut_dataset_select",
        help="选择一个内置的数据集生成器来创建演示数据。"
    )

    # --- 2. 数据集参数 ---
    st.subheader("2. 调整数据集参数")
    col_data1, col_data2 = st.columns(2)
    with col_data1:
        st.session_state.tut_n_samples = st.slider(
            "样本数量:", min_value=50, max_value=500,
            value=st.session_state.tut_n_samples, step=50, key="tut_samples",
            help="生成数据点的总数。"
        )
        # 根据数据集类型显示特定参数
        if st.session_state.tut_dataset_name == "Blobs":
            st.session_state.tut_centers = st.slider(
                "团簇数量 (真实):", min_value=2, max_value=6,
                value=st.session_state.tut_centers, step=1, key="tut_blob_centers",
                help="生成数据的真实中心点数量。"
            )

    with col_data2:
        # 保持2D以便可视化
        st.text_input("特征数量 (固定为2D):", value="2", key="tut_features_display", disabled=True)

        if st.session_state.tut_dataset_name == "Blobs":
            st.session_state.tut_cluster_std = st.slider(
                "团簇标准差:", min_value=0.1, max_value=2.5,
                value=st.session_state.tut_cluster_std, step=0.1, format="%.1f", key="tut_blob_std",
                help="控制每个团簇内点的分散程度。"
            )
        if st.session_state.tut_dataset_name == "Moons" or st.session_state.tut_dataset_name == "Circles":
            st.session_state.tut_noise = st.slider(
                "噪声水平:", min_value=0.00, max_value=0.30,
                value=st.session_state.tut_noise, step=0.01, format="%.2f", key="tut_noise_slider",
                help="添加到数据中的高斯噪声的标准差。"
            )
        if st.session_state.tut_dataset_name == "Circles":
            st.session_state.tut_factor = st.slider(
                "圆环因子:", min_value=0.1, max_value=0.9,
                value=st.session_state.tut_factor, step=0.1, format="%.1f", key="tut_circle_factor",
                help="内圆与外圆半径的比例 (仅适用于 Circles)。"
            )

    # --- 生成数据集按钮 ---
    if st.button("🔄 生成/更新数据集", key="tut_generate_data", help="点击根据当前参数生成新的示例数据集"):
        random_state_data = 42 # 固定随机种子以便重现
        X_raw = None
        y_true = None
        try:
            if st.session_state.tut_dataset_name == "Blobs":
                X_raw, y_true = make_blobs(n_samples=st.session_state.tut_n_samples,
                                      n_features=st.session_state.tut_n_features,
                                      centers=st.session_state.tut_centers,
                                      cluster_std=st.session_state.tut_cluster_std,
                                      random_state=random_state_data)
            elif st.session_state.tut_dataset_name == "Moons":
                X_raw, y_true = make_moons(n_samples=st.session_state.tut_n_samples,
                                      noise=st.session_state.tut_noise,
                                      random_state=random_state_data)
            elif st.session_state.tut_dataset_name == "Circles":
                X_raw, y_true = make_circles(n_samples=st.session_state.tut_n_samples,
                                        noise=st.session_state.tut_noise,
                                        factor=st.session_state.tut_factor,
                                        random_state=random_state_data)

            if X_raw is not None:
                # --- 数据标准化 ---
                # 使用会话状态中的 scaler 对象
                st.session_state.tut_data_X = st.session_state.tut_scaler.fit_transform(X_raw)
                st.session_state.tut_data_y = y_true # 存储真实标签（如果有）
                st.session_state.tut_labels = None # 重置聚类标签
                st.session_state.tut_centers_result = None # 重置中心点
                st.success("示例数据集已生成/更新。")
            else:
                 st.error("无法生成所选数据集。")

        except Exception as data_gen_e:
            st.error(f"生成数据集时出错: {data_gen_e}")
            print(traceback.format_exc()) # 打印详细错误到控制台
            st.session_state.tut_data_X = None # 出错时清空

    # --- 显示生成的数据集 ---
    if st.session_state.tut_data_X is not None:
        st.write("---")
        st.markdown("#### 生成的数据集（已标准化）")
        # 检查绘图函数是否可用
        if CLUSTERING_MODULE_AVAILABLE:
            try:
                fig_data, ax_data = create_figure_with_safe_dimensions(8, 5)
                # 使用真实标签着色（如果可用），否则用单一颜色
                point_colors = None
                if st.session_state.tut_data_y is not None:
                    n_classes = len(np.unique(st.session_state.tut_data_y))
                    cmap = plt.cm.get_cmap('viridis', n_classes)
                    point_colors = [cmap(label) for label in st.session_state.tut_data_y]
                else:
                    point_colors = '#3498db' # Default color if no true labels

                ax_data.scatter(st.session_state.tut_data_X[:, 0], st.session_state.tut_data_X[:, 1],
                                c=point_colors, s=30, alpha=0.7)
                apply_plot_style(ax_data) # 应用样式
                title_str = f"示例数据集: {st.session_state.tut_dataset_name}"
                if st.session_state.tut_data_y is not None:
                    title_str += " (按真实类别着色)"

                ax_data.set_title(title_str, fontproperties=FONT_PROP if FONT_PROP else None)
                ax_data.set_xlabel("特征 1 (标准化后)", fontproperties=FONT_PROP if FONT_PROP else None)
                ax_data.set_ylabel("特征 2 (标准化后)", fontproperties=FONT_PROP if FONT_PROP else None)
                st.pyplot(fig_data)
            except Exception as plot_err:
                st.warning(f"绘制数据集图表时出错: {plot_err}")
                print(traceback.format_exc())
        else:
            st.warning("无法显示数据集图表，因为主聚类模块 (clustering.py) 加载失败。")
    else:
        st.info("请点击 **“🔄 生成/更新数据集”** 按钮来创建数据。")
        return # 如果没有数据，则停止后续操作

    st.markdown("---")

    # --- 3. 选择聚类方法与参数 ---
    st.subheader("3. 选择聚类方法与参数")
    tut_method_options = ["K-Means", "DBSCAN"] # 目前支持这两种
    st.session_state.tut_method = st.selectbox(
        "选择聚类算法:",
        options=tut_method_options,
        key="tut_method_select",
        help="选择要应用于上方生成的数据的聚类算法。"
    )

    # 根据选择的方法显示参数
    if st.session_state.tut_method == "K-Means":
        st.session_state.tut_kmeans_k = st.slider(
            "K值 (期望的聚类数量):", min_value=1, max_value=10,
            value=st.session_state.tut_kmeans_k, step=1, key="tut_kmeans_slider",
            help="K-Means 算法需要预先指定要将数据划分成的簇的数量 (K)。"
        )
        st.markdown(f"**算法说明:** K-Means 尝试将数据划分为 **{st.session_state.tut_kmeans_k}** 个簇，使得每个数据点都属于与其最近的簇中心（质心）对应的簇。算法通过迭代更新簇中心和点的归属来最小化簇内平方和。")
        st.markdown(f"**参数影响:** K值的选择对结果影响很大。过小的K值可能将不同的簇合并，过大的K值可能将同一个簇分割。可以尝试不同的K值，观察轮廓系数的变化。")
    elif st.session_state.tut_method == "DBSCAN":
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.session_state.tut_dbscan_eps = st.slider(
                "Epsilon (邻域半径 ε):", min_value=0.1, max_value=2.0,
                value=st.session_state.tut_dbscan_eps, step=0.05, format="%.2f", key="tut_dbscan_eps_slider",
                help="定义一个点的“邻域”范围。这是DBSCAN中最重要的参数之一。"
            )
            st.markdown("**算法说明:** DBSCAN (Density-Based Spatial Clustering of Applications with Noise) 是一种基于密度的聚类算法。它将密集区域中的点连接起来形成簇，并将稀疏区域中的点标记为噪声（异常点）。它不需要预先指定簇的数量。")
        with col_p2:
            st.session_state.tut_dbscan_min_samples = st.slider(
                "Min Samples (核心点最小邻居数):", min_value=2, max_value=15,
                value=st.session_state.tut_dbscan_min_samples, step=1, key="tut_dbscan_min_samples_slider",
                help="一个点要成为“核心点”，其 ε 邻域内至少需要包含的样本点数（包括自身）。"
            )
        st.markdown("**参数影响:**")
        st.markdown(f"- **Epsilon (ε):** 控制邻域的大小。值越小，要求的密度越高，可能产生更多小簇或噪声点；值越大，可能将不同的簇合并。对于标准化数据，通常取值在 0.1 到 1.0 之间。")
        st.markdown(f"- **Min Samples:** 控制成为核心点的密度阈值。值越大，要求的密度越高，可能产生更多噪声点。通常建议取值为 `维度数 + 1` 或更高。对于2D数据，常用值为 3 到 5。")
        st.markdown("尝试调整这两个参数，观察簇的形状、数量以及噪声点的变化。DBSCAN 擅长发现非凸形状的簇，并能识别噪声点。")

    # --- 运行聚类演示按钮 ---
    if st.button("🚀 运行聚类演示", key="tut_run_clustering", help="使用当前选择的算法和参数对上方数据进行聚类"):
        if st.session_state.tut_data_X is None:
             st.error("请先生成数据集！")
        else:
            X_tut = st.session_state.tut_data_X # 使用标准化后的数据
            labels_tut = None
            centers_tut = None
            method_tut = st.session_state.tut_method
            success_flag = False
            try:
                with st.spinner(f"正在运行 {method_tut}..."):
                    if method_tut == "K-Means":
                        # 检查 K 值是否有效
                        if st.session_state.tut_kmeans_k > len(X_tut):
                             st.error(f"K 值 ({st.session_state.tut_kmeans_k}) 不能大于样本数量 ({len(X_tut)})。请减小 K 值。")
                        else:
                            kmeans = KMeans(n_clusters=st.session_state.tut_kmeans_k,
                                            n_init=10, # 推荐设置 n_init 以提高稳定性
                                            random_state=42)
                            labels_tut = kmeans.fit_predict(X_tut)
                            centers_tut = kmeans.cluster_centers_ # 获取聚类中心
                            success_flag = True
                    elif method_tut == "DBSCAN":
                        dbscan = DBSCAN(eps=st.session_state.tut_dbscan_eps,
                                        min_samples=st.session_state.tut_dbscan_min_samples)
                        labels_tut = dbscan.fit_predict(X_tut)
                        centers_tut = None # DBSCAN 没有中心点
                        success_flag = True

                if success_flag:
                    st.session_state.tut_labels = labels_tut
                    st.session_state.tut_centers_result = centers_tut # 存储聚类结果中心点
                    st.success(f"{method_tut} 聚类完成！请查看下方结果。")
                # else: # No need for else if errors are handled inside
                     # pass

            except Exception as e:
                st.error(f"运行 {method_tut} 出错: {e}")
                print(traceback.format_exc())
                st.session_state.tut_labels = None
                st.session_state.tut_centers_result = None

    st.markdown("---")

    # --- 4. 显示聚类结果 ---
    if st.session_state.tut_labels is not None:
        st.subheader("4. 聚类结果可视化与评估")
        st.markdown(f"下图显示了使用 **{st.session_state.tut_method}** 和当前参数得到的聚类结果。不同颜色代表不同的簇。")

        # 检查绘图函数是否可用
        if CLUSTERING_MODULE_AVAILABLE:
            try:
                # 调用从 clustering.py 导入的绘图函数
                fig_results = plot_clusters_2d(
                    st.session_state.tut_data_X, # 绘制标准化后的数据
                    st.session_state.tut_labels,
                    method_name=st.session_state.tut_method,
                    centers=st.session_state.tut_centers_result # 传入聚类中心点
                )
                # --- 修正标题 ---
                title_suffix = "(教学演示)"
                current_title = fig_results.axes[0].get_title()
                fig_results.axes[0].set_title(f"{current_title} {title_suffix}", fontproperties=FONT_PROP if FONT_PROP else None)
                # --- 结束修正 ---
                st.pyplot(fig_results)
            except Exception as plot_err:
                 st.warning(f"绘制聚类结果图表时出错: {plot_err}")
                 print(traceback.format_exc())

            # --- 显示评估指标 ---
            try:
                # 轮廓系数
                unique_labels_tut = np.unique(st.session_state.tut_labels)
                # 只有当簇数大于1且小于样本数时才能计算轮廓系数
                if len(unique_labels_tut) > 1 and len(unique_labels_tut) < len(st.session_state.tut_labels):
                    score = silhouette_score(st.session_state.tut_data_X, st.session_state.tut_labels)
                    help_text = "轮廓系数衡量簇内点的紧密度和簇间点的分离度。取值范围 [-1, 1]，值越接近 1 表示聚类效果越好，簇内紧密且簇间分离；接近 0 表示簇有重叠；负值通常表示样本可能被分配到了错误的簇。"
                    st.metric("轮廓系数 (Silhouette Score)", f"{score:.3f}", help=help_text)
                elif len(unique_labels_tut) <= 1:
                     st.warning("无法计算轮廓系数，因为只找到了一个簇或没有找到簇。")
                else: # len(unique_labels_tut) == len(st.session_state.tut_labels)
                     st.warning("无法计算轮廓系数，因为每个点都被分配到了自己的簇。")

            except Exception as score_e:
                st.error(f"计算轮廓系数时出错: {score_e}")
                print(traceback.format_exc())

            # --- 结果解读 ---
            st.markdown("#### 结果解读")
            if st.session_state.tut_method == "K-Means":
                st.markdown(f"K-Means 将数据分为了 **{st.session_state.tut_kmeans_k}** 个簇。")
                st.markdown("- 每个点的颜色代表它所属的簇。")
                st.markdown("- 黑色星号 (*) 代表每个簇的计算中心（质心）。")
                st.markdown("- **观察:** 尝试改变 K 值：")
                st.markdown("  - 如果 K 值小于真实的团簇数（例如在 Blobs 数据集上），K-Means 可能会将不同的团簇合并。")
                st.markdown("  - 如果 K 值大于真实的团簇数，K-Means 可能会将一个团簇分割成多个。")
                st.markdown("  - 观察轮廓系数如何随着 K 值的变化而变化。通常，合适的 K 值会对应一个较高的轮廓系数。")
                st.markdown("- **局限性:** K-Means 假设簇是凸形的（类圆形），并且对初始中心点敏感。对于非凸形状（如 Moons, Circles）的数据效果不佳。")
            elif st.session_state.tut_method == "DBSCAN":
                n_clusters_found = len(set(st.session_state.tut_labels)) - (1 if -1 in st.session_state.tut_labels else 0)
                n_noise = np.sum(st.session_state.tut_labels == -1)
                st.markdown(f"DBSCAN 根据数据密度自动识别了 **{n_clusters_found}** 个簇。")
                st.markdown(f"- 不同颜色的点代表不同的簇。")
                st.markdown(f"- 灰色叉号 (x) 代表被识别为噪声或异常的点 ({n_noise} 个)。")
                st.markdown(f"- **观察:** 尝试调整 **Epsilon (ε)** 和 **Min Samples** 参数：")
                st.markdown(f"  - **减小 ε** 或 **增大 Min Samples** 会提高密度要求，可能导致更多点被视为噪声，或将大簇分割成小簇。")
                st.markdown(f"  - **增大 ε** 或 **减小 Min Samples** 会降低密度要求，可能将噪声点纳入簇中，或将不同的簇合并。")
                st.markdown("- **优势:** DBSCAN 不需要预先指定簇的数量，可以发现任意形状的簇（如 Moons 和 Circles 数据集），并且能有效识别噪声点。")
                st.markdown("- **局限性:** 对参数选择比较敏感；对于密度差异很大的簇效果可能不佳。")
        else:
             st.warning("无法显示聚类结果图表或评估指标，因为主聚类模块 (clustering.py) 加载失败。")



# --- 允许直接运行此脚本进行测试 ---
if __name__ == "__main__":
    # st.set_page_config(layout="wide", page_title="聚类教学演示（独立运行）")
    st.sidebar.info("这是聚类教学模块的独立测试运行。")
    # 在独立运行时，仍然尝试调用字体设置，即使 clustering.py 可能不在路径中
    # try:
    #     from clustering import setup_chinese_font
    #     FONT_PROP = setup_chinese_font()
    # except ImportError:
    #     # 如果直接运行且找不到 clustering，使用这里的字体设置
    #     from matplotlib.font_manager import FontProperties
    #     # 尝试加载一个常见的中文字体，如果失败则忽略
    #     try: FONT_PROP = FontProperties(fname='C:/Windows/Fonts/msyh.ttc') # Windows 示例
    #     except: FONT_PROP = None
    #     if FONT_PROP: plt.rcParams['font.family'] = FONT_PROP.get_name()
    #     plt.rcParams['axes.unicode_minus'] = False
    #     print("警告: 无法从 clustering.py 加载字体设置，尝试使用系统默认。")

    show_tutorial_page()