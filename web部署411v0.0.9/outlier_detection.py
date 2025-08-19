# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import base64
from io import BytesIO
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
import platform
import matplotlib.font_manager as fm
import datetime


# --- 字体设置 ---
def setup_chinese_font():
    """设置中文字体支持"""
    system = platform.system()
    font_candidates = []
    if system == 'Windows':
        font_candidates = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']
    elif system == 'Darwin':
        font_candidates = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Apple LiGothic Medium']
    else:
        font_candidates = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'Droid Sans Fallback']
    font_candidates.extend(['DejaVu Sans', 'Arial', 'Helvetica'])

    font_prop = None
    for font_name in font_candidates:
        try:
            font_path = fm.findfont(fm.FontProperties(family=font_name))
            if os.path.exists(font_path) and 'DejaVuSans' not in font_path:
                print(f"字体日志: 使用字体 '{font_name}' 在路径: {font_path}")
                plt.rcParams['font.family'] = 'sans-serif'
                plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
                font_prop = fm.FontProperties(family=font_name)
                break
        except Exception as e:
            print(f"字体日志: 尝试字体 {font_name} 失败: {e}")

    if not font_prop:
        print("字体日志: 未找到合适的中文字体，绘图中的中文可能无法正常显示。")
    plt.rcParams['axes.unicode_minus'] = False
    return font_prop


FONT_PROP = setup_chinese_font()


# --- 绘图辅助函数 ---
def apply_plot_style(ax):
    """应用统一的绘图样式"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.6, color='#bdc3c7')
    ax.tick_params(axis='both', which='major', labelsize=9, colors='#34495e')
    ax.xaxis.label.set_fontsize(10);
    ax.yaxis.label.set_fontsize(10)
    ax.title.set_fontsize(12);
    ax.title.set_fontweight('bold')
    ax.xaxis.label.set_fontweight('bold');
    ax.yaxis.label.set_fontweight('bold')
    return ax


def plot_outliers_2d(X_df, labels, method_name="异常点检测"):
    """使用PCA降维（如果需要）绘制2D散点图，突出显示异常点"""
    fig, ax = plt.subplots(figsize=(10, 7), dpi=80)
    apply_plot_style(ax)
    font_kwargs = {'fontproperties': FONT_PROP} if FONT_PROP else {}

    try:
        # PCA降维
        pca_applied = False
        if X_df.shape[1] > 2:
            pca = PCA(n_components=2, random_state=42)
            scaler_pca = StandardScaler()
            X_scaled_pca = scaler_pca.fit_transform(X_df.values)
            X_2d = pca.fit_transform(X_scaled_pca)
            explained_var = pca.explained_variance_ratio_
            pca_applied = True
            xlabel = "主成分 1"
            ylabel = "主成分 2"
        else:
            X_2d = X_df.values
            explained_var = None
            xlabel = X_df.columns[0] if X_df.shape[1] >= 1 else "维度 1"
            ylabel = X_df.columns[1] if X_df.shape[1] >= 2 else "维度 2"

        # 识别正常点和异常点
        normal_points = X_2d[labels != -1]
        outliers = X_2d[labels == -1]

        # 绘制正常点（蓝色圆点）
        ax.scatter(normal_points[:, 0], normal_points[:, 1], c='#3498db', marker='o',
                   s=25, alpha=0.6, label='正常点')

        # 绘制异常点（红色叉号）
        ax.scatter(outliers[:, 0], outliers[:, 1], c='#e74c3c', marker='x',
                   s=60, alpha=0.9, label='异常点')

        # 设置标题和标签
        title = f"{method_name} 检测结果"
        if pca_applied:
            title += f" (PCA降维: 方差解释率 {explained_var[0]:.2f}, {explained_var[1]:.2f})"

        ax.set_title(title, **font_kwargs)
        ax.set_xlabel(xlabel, **font_kwargs)
        ax.set_ylabel(ylabel, **font_kwargs)

        # 添加图例
        legend = ax.legend(prop=FONT_PROP, fontsize=9)
        plt.tight_layout()

    except Exception as e:
        print(f"绘制异常点图时出错: {e}")
        ax.text(0.5, 0.5, f'绘图错误: {e}', ha='center', va='center', color='red', **font_kwargs)

    return fig


def get_download_link(df, filename, text):
    """生成CSV下载链接"""
    csv = df.to_csv(index=False, encoding='utf-8-sig')
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href


# --- 异常点检测算法信息 ---
ALGORITHM_INFO = {
    "DBSCAN": {
        "name": "DBSCAN",
        "description": "基于密度的空间聚类算法",
        "suitable_for": [
            "• 能发现任意形状的异常点",
            "• 适合密度不均匀的数据",
            "• 不需要预先指定异常点数量",
            "• 能区分噪声和边界点"
        ],
        "pros": [
            "• 能识别不规则形状的异常区域",
            "• 对异常点大小不敏感",
            "• 不假设数据分布"
        ],
        "cons": [
            "• 需要手动调整参数（eps和min_samples）",
            "• 在高维数据中表现可能不佳",
            "• 对密度差异大的数据敏感"
        ],
        "params": ["eps", "min_samples"]
    },
    "IsolationForest": {
        "name": "孤立森林",
        "description": "基于随机森林的异常点检测算法",
        "suitable_for": [
            "• 高维数据",
            "• 大型数据集",
            "• 全局异常点检测",
            "• 不需要标签的数据"
        ],
        "pros": [
            "• 计算效率高，适合大数据集",
            "• 在高维数据中表现较好",
            "• 参数少，易于调优",
            "• 能够快速识别全局异常点"
        ],
        "cons": [
            "• 可能忽略局部异常点",
            "• 对树的数量敏感",
            "• 主要检测全局异常，不擅长局部异常"
        ],
        "params": ["n_estimators", "contamination"]
    },
    "LOF": {
        "name": "局部异常因子",
        "description": "基于局部密度的异常点检测算法",
        "suitable_for": [
            "• 密度变化的数据",
            "• 局部异常点检测",
            "• 中等规模数据集",
            "• 需要捕捉局部异常的场景"
        ],
        "pros": [
            "• 能够检测局部异常点",
            "• 适合密度不均匀的数据",
            "• 提供异常程度评分"
        ],
        "cons": [
            "• 计算复杂度较高，不适合大数据集",
            "• 需要调整邻居数量参数",
            "• 在高维数据中可能表现不佳"
        ],
        "params": ["n_neighbors", "contamination"]
    },
    "OneClassSVM": {
        "name": "单类支持向量机",
        "description": "基于支持向量机的异常点检测",
        "suitable_for": [
            "• 小到中等规模数据集",
            "• 希望找到决策边界的场景",
            "• 单类分类问题",
            "• 特征维度适中的数据"
        ],
        "pros": [
            "• 理论基础扎实",
            "• 能找到清晰的决策边界",
            "• 参数相对较少"
        ],
        "cons": [
            "• 计算复杂度高，不适合大数据集",
            "• 对核函数选择敏感",
            "• 内存需求大"
        ],
        "params": ["kernel", "nu", "gamma"]
    },
    "EllipticEnvelope": {
        "name": "椭圆包络",
        "description": "假设数据服从多元高斯分布的异常点检测",
        "suitable_for": [
            "• 数据接近正态分布",
            "• 中等规模数据集",
            "• 连续数值特征",
            "• 需要快速检测的场景"
        ],
        "pros": [
            "• 计算速度快",
            "• 参数很少",
            "• 适合高斯分布数据"
        ],
        "cons": [
            "• 假设数据服从正态分布",
            "• 不适合非高斯分布数据",
            "• 对异常点非常敏感"
        ],
        "params": ["contamination", "support_fraction"]
    }
}


# --- Streamlit UI 函数 ---
def initialize_outlier_session_state():
    """初始化异常点检测页面的会话状态"""
    defaults = {
        'outlier_data': None,
        'outlier_original_data_aligned': None,
        'outlier_column_names': [],
        'outlier_selected_features': [],
        'outlier_results': None,
        'outlier_algorithm': 'DBSCAN',
        'outlier_normalize': True,
        # DBSCAN参数
        'outlier_params_eps': 0.5,
        'outlier_params_min_samples': 5,
        # IsolationForest参数
        'outlier_params_n_estimators': 100,
        'outlier_params_contamination': 0.05,
        # LOF参数
        'outlier_params_n_neighbors': 20,
        'outlier_params_lof_contamination': 0.05,
        # OneClassSVM参数
        'outlier_params_kernel': 'rbf',
        'outlier_params_nu': 0.05,
        'outlier_params_gamma': 'scale',
        # EllipticEnvelope参数
        'outlier_params_ee_contamination': 0.05,
        'outlier_params_support_fraction': None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def show_outlier_detection_page():
    """显示异常点检测页面"""
    initialize_outlier_session_state()

    st.title("🔍 异常点检测")

    # 使用固定布局而不是expander，避免状态丢失
    st.markdown("## 📚 算法选择与介绍")

    # 创建算法选择和介绍的布局
    algo_col1, algo_col2 = st.columns([1, 2])

    with algo_col1:
        st.session_state.outlier_algorithm = st.selectbox(
            "选择异常点检测算法",
            options=list(ALGORITHM_INFO.keys()),
            index=list(ALGORITHM_INFO.keys()).index(st.session_state.outlier_algorithm),
            key="outlier_algorithm_select"
        )

    with algo_col2:
        # 显示算法信息
        algo_info = ALGORITHM_INFO[st.session_state.outlier_algorithm]
        st.markdown(f"### {algo_info['name']} 算法")
        st.markdown(f"**描述**: {algo_info['description']}")

    # 详细介绍使用可折叠的columns
    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.markdown("#### 🎯 适用场景")
        for item in algo_info['suitable_for']:
            st.markdown(item)

    with info_col2:
        st.markdown("#### ✅ 优点")
        for item in algo_info['pros']:
            st.markdown(item)

        st.markdown("#### ❌ 缺点")
        for item in algo_info['cons']:
            st.markdown(item)

    st.markdown("---")

    # 创建选项卡
    tab1, tab2, tab3 = st.tabs(["📁 数据导入", "⚙️ 参数设置", "📈 结果展示"])

    with tab1:
        create_outlier_data_import_section()

    with tab2:
        create_outlier_params_section()

    with tab3:
        create_outlier_results_section()


def create_outlier_data_import_section():
    """创建数据导入和特征选择部分"""
    st.header("1. 数据导入与特征选择")

    uploaded_file = st.file_uploader("上传包含数值特征的数据文件 (CSV/Excel)",
                                     type=["csv", "xlsx", "xls"], key="outlier_uploader")

    if uploaded_file:
        if st.button("加载数据", key="outlier_load_btn"):
            with st.spinner("正在加载和处理数据..."):
                try:
                    # 加载原始数据
                    data_original = pd.read_csv(uploaded_file) if uploaded_file.name.lower().endswith(
                        '.csv') else pd.read_excel(uploaded_file)

                    # 选择数值列
                    numeric_cols = data_original.select_dtypes(include=np.number).columns.tolist()
                    if not numeric_cols:
                        st.error("上传的文件中未找到数值列。")
                        st.session_state.outlier_data = None
                        st.session_state.outlier_original_data_aligned = None
                        return

                    # 从数值列中移除包含NaN的行
                    initial_rows = len(data_original)
                    data_numeric_raw = data_original[numeric_cols].copy()
                    data_numeric = data_numeric_raw.dropna()
                    cleaned_rows = len(data_numeric)

                    if cleaned_rows < initial_rows:
                        st.warning(f"移除了 {initial_rows - cleaned_rows} 行在数值列中包含缺失值的数据。")

                    if data_numeric.empty:
                        st.error("处理缺失值后数据为空。")
                        st.session_state.outlier_data = None
                        st.session_state.outlier_original_data_aligned = None
                        return

                    # 存储清理后的数值数据和对应的原始数据
                    st.session_state.outlier_data = data_numeric
                    st.session_state.outlier_original_data_aligned = data_original.loc[data_numeric.index]
                    st.session_state.outlier_column_names = list(data_numeric.columns)
                    st.session_state.outlier_selected_features = list(data_numeric.columns)
                    st.session_state.outlier_results = None
                    st.success(f"成功加载并处理数据: {cleaned_rows} 行有效数值数据。")

                except Exception as e:
                    st.error(f"加载或处理数据时出错: {e}")
                    st.session_state.outlier_data = None
                    st.session_state.outlier_original_data_aligned = None

    # 数据预览和特征选择
    if st.session_state.outlier_data is not None:
        st.subheader("数据预览 (数值列 - 清理后)")
        st.dataframe(st.session_state.outlier_data.head())

        st.subheader("特征选择")
        available_cols = st.session_state.outlier_column_names
        default_selection = [col for col in st.session_state.outlier_selected_features if col in available_cols]
        if not default_selection and available_cols:
            default_selection = available_cols

        st.session_state.outlier_selected_features = st.multiselect(
            "选择用于异常点检测的特征列",
            available_cols,
            default=default_selection,
            key="outlier_feature_select"
        )

        if not st.session_state.outlier_selected_features:
            st.warning("请至少选择一个特征列。")
        else:
            st.info(f"已选择 {len(st.session_state.outlier_selected_features)} 个特征。")


def create_outlier_params_section():
    """创建参数设置部分"""
    st.header("2. 参数设置")

    if st.session_state.outlier_data is None:
        st.info("请先在数据导入选项卡中加载数据。")
        return
    if not st.session_state.outlier_selected_features:
        st.warning("请先在数据导入选项卡中选择特征列。")
        return

    # 预处理选项
    st.subheader("预处理")
    st.session_state.outlier_normalize = st.checkbox(
        "标准化特征 (推荐)",
        value=st.session_state.outlier_normalize,
        key="outlier_norm_cb"
    )

    # 算法特定参数
    st.subheader(f"{st.session_state.outlier_algorithm} 算法参数")

    if st.session_state.outlier_algorithm == "DBSCAN":
        create_dbscan_params()
    elif st.session_state.outlier_algorithm == "IsolationForest":
        create_isolation_forest_params()
    elif st.session_state.outlier_algorithm == "LOF":
        create_lof_params()
    elif st.session_state.outlier_algorithm == "OneClassSVM":
        create_one_class_svm_params()
    elif st.session_state.outlier_algorithm == "EllipticEnvelope":
        create_elliptic_envelope_params()

    # 运行按钮
    can_run = st.session_state.outlier_data is not None and st.session_state.outlier_selected_features
    if st.button(f"运行 {st.session_state.outlier_algorithm} 检测异常点",
                 type="primary", key="run_outlier_btn", disabled=not can_run):
        run_outlier_detection()


def create_dbscan_params():
    """创建DBSCAN参数设置"""
    st.markdown("""
    - **Epsilon (eps)**: 定义一个点的邻域半径。值越小，要求的密度越高。
    - **Min Samples**: 定义一个核心点所需的邻域内最小样本数（包括自身）。
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.outlier_params_eps = st.number_input(
            "Epsilon (eps)", min_value=0.01, max_value=10.0,
            value=st.session_state.outlier_params_eps, step=0.05, format="%.2f",
            key="outlier_eps_input", help="邻域半径，影响密度要求"
        )
    with col2:
        st.session_state.outlier_params_min_samples = st.number_input(
            "Min Samples", min_value=2, max_value=100,
            value=st.session_state.outlier_params_min_samples, step=1,
            key="outlier_minsamples_input", help="核心点所需的最小邻域样本数"
        )


def create_isolation_forest_params():
    """创建Isolation Forest参数设置"""
    st.markdown("""
    - **n_estimators**: 构建的树的数量。通常值越大，性能越好，但计算成本也越高。
    - **contamination**: 数据集中异常点的预期比例。
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.outlier_params_n_estimators = st.number_input(
            "树的数量 (n_estimators)", min_value=10, max_value=1000,
            value=st.session_state.outlier_params_n_estimators, step=10,
            key="outlier_n_estimators_input", help="构建的决策树数量"
        )
    with col2:
        st.session_state.outlier_params_contamination = st.number_input(
            "异常点比例 (contamination)", min_value=0.001, max_value=0.5,
            value=st.session_state.outlier_params_contamination, step=0.001, format="%.3f",
            key="outlier_contamination_input", help="预期的异常点占比"
        )


def create_lof_params():
    """创建LOF参数设置"""
    st.markdown("""
    - **n_neighbors**: 计算局部密度使用的邻居数量。
    - **contamination**: 预期的异常点比例。
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.outlier_params_n_neighbors = st.number_input(
            "邻居数量 (n_neighbors)", min_value=2, max_value=100,
            value=st.session_state.outlier_params_n_neighbors, step=1,
            key="outlier_n_neighbors_input", help="计算密度时考虑的邻居数量"
        )
    with col2:
        st.session_state.outlier_params_lof_contamination = st.number_input(
            "异常点比例 (contamination)", min_value=0.001, max_value=0.5,
            value=st.session_state.outlier_params_lof_contamination, step=0.001, format="%.3f",
            key="outlier_lof_contamination_input", help="预期的异常点占比"
        )


def create_one_class_svm_params():
    """创建One-Class SVM参数设置"""
    st.markdown("""
    - **kernel**: 核函数类型。
    - **nu**: 决策边界错误率的上界。
    - **gamma**: 核函数的系数（仅用于rbf、poly、sigmoid核）。
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state.outlier_params_kernel = st.selectbox(
            "核函数 (kernel)", options=['rbf', 'linear', 'poly', 'sigmoid'],
            index=['rbf', 'linear', 'poly', 'sigmoid'].index(st.session_state.outlier_params_kernel),
            key="outlier_kernel_select", help="选择核函数类型"
        )
    with col2:
        st.session_state.outlier_params_nu = st.number_input(
            "Nu", min_value=0.001, max_value=1.0,
            value=st.session_state.outlier_params_nu, step=0.001, format="%.3f",
            key="outlier_nu_input", help="决策边界错误率上界"
        )
    with col3:
        st.session_state.outlier_params_gamma = st.selectbox(
            "Gamma", options=['scale', 'auto'],
            index=['scale', 'auto'].index(st.session_state.outlier_params_gamma),
            key="outlier_gamma_select", help="核函数系数"
        )


def create_elliptic_envelope_params():
    """创建Elliptic Envelope参数设置"""
    st.markdown("""
    - **contamination**: 预期的异常点比例。
    - **support_fraction**: 用于计算经验协方差的点的比例（None=auto）。
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.outlier_params_ee_contamination = st.number_input(
            "异常点比例 (contamination)", min_value=0.001, max_value=0.5,
            value=st.session_state.outlier_params_ee_contamination, step=0.001, format="%.3f",
            key="outlier_ee_contamination_input", help="预期的异常点占比"
        )
    with col2:
        support_fraction_val = st.session_state.outlier_params_support_fraction
        display_value = 1.0 if support_fraction_val is None else support_fraction_val

        st.session_state.outlier_params_support_fraction = st.number_input(
            "支持分数 (support_fraction)", min_value=0.001, max_value=1.0,
            value=display_value, step=0.001, format="%.3f",
            key="outlier_support_fraction_input", help="用于计算协方差的点的比例"
        )

        if st.session_state.outlier_params_support_fraction >= 0.999:
            st.session_state.outlier_params_support_fraction = None


def run_outlier_detection():
    """执行异常点检测"""
    if st.session_state.outlier_data is None or not st.session_state.outlier_selected_features:
        st.error("请先加载数据并选择特征。")
        return

    X = st.session_state.outlier_data[st.session_state.outlier_selected_features].copy()
    if X.empty:
        st.error("选择的特征数据为空，无法运行检测。")
        return

    processed_index = X.index.copy()

    with st.spinner(f"正在运行 {st.session_state.outlier_algorithm}..."):
        try:
            # 标准化
            X_processed_np = X.values
            if st.session_state.outlier_normalize:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X_processed_np = X_scaled
                print("数据已标准化。")
            else:
                print("数据未标准化。")

            # 根据算法执行检测
            if st.session_state.outlier_algorithm == "DBSCAN":
                labels = run_dbscan(X_processed_np)
            elif st.session_state.outlier_algorithm == "IsolationForest":
                labels = run_isolation_forest(X_processed_np)
            elif st.session_state.outlier_algorithm == "LOF":
                labels = run_lof(X_processed_np)
            elif st.session_state.outlier_algorithm == "OneClassSVM":
                labels = run_one_class_svm(X_processed_np)
            elif st.session_state.outlier_algorithm == "EllipticEnvelope":
                labels = run_elliptic_envelope(X_processed_np)
            else:
                st.error(f"未实现的算法: {st.session_state.outlier_algorithm}")
                return

            # 保存结果
            st.session_state.outlier_results = {
                'labels': labels,
                'processed_index': processed_index,
                'X_processed_for_plot': X_processed_np,
                'algorithm': st.session_state.outlier_algorithm
            }
            st.success(f"{st.session_state.outlier_algorithm} 运行完成！请查看结果展示。")

        except Exception as e:
            st.error(f"运行 {st.session_state.outlier_algorithm} 时出错: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.session_state.outlier_results = None


def run_dbscan(X):
    """运行DBSCAN算法"""
    dbscan = DBSCAN(
        eps=st.session_state.outlier_params_eps,
        min_samples=st.session_state.outlier_params_min_samples,
        n_jobs=-1
    )
    return dbscan.fit_predict(X)


def run_isolation_forest(X):
    """运行Isolation Forest算法"""
    iso_forest = IsolationForest(
        n_estimators=st.session_state.outlier_params_n_estimators,
        contamination=st.session_state.outlier_params_contamination,
        random_state=42,
        n_jobs=-1
    )
    # Isolation Forest返回1为正常点，-1为异常点
    return iso_forest.fit_predict(X)


def run_lof(X):
    """运行Local Outlier Factor算法"""
    lof = LocalOutlierFactor(
        n_neighbors=st.session_state.outlier_params_n_neighbors,
        contamination=st.session_state.outlier_params_lof_contamination,
        n_jobs=-1
    )
    # LOF返回1为正常点，-1为异常点
    return lof.fit_predict(X)


def run_one_class_svm(X):
    """运行One-Class SVM算法"""
    svm = OneClassSVM(
        kernel=st.session_state.outlier_params_kernel,
        nu=st.session_state.outlier_params_nu,
        gamma=st.session_state.outlier_params_gamma
    )
    # One-Class SVM返回1为正常点，-1为异常点
    return svm.fit_predict(X)


def run_elliptic_envelope(X):
    """运行Elliptic Envelope算法"""
    elliptic = EllipticEnvelope(
        contamination=st.session_state.outlier_params_ee_contamination,
        support_fraction=st.session_state.outlier_params_support_fraction,
        assume_centered=False
    )
    # Elliptic Envelope返回1为正常点，-1为异常点
    return elliptic.fit_predict(X)


def create_outlier_results_section():
    """创建结果展示部分"""
    st.header("3. 结果展示")

    results = st.session_state.get('outlier_results')
    original_data_aligned = st.session_state.get('outlier_original_data_aligned')

    if results is None or original_data_aligned is None:
        st.info("请先运行异常点检测。")
        return

    labels = results['labels']
    processed_index = results['processed_index']
    X_processed_for_plot = results['X_processed_for_plot']
    algorithm = results.get('algorithm', st.session_state.outlier_algorithm)

    # 计算异常点统计
    outlier_mask = (labels == -1)
    n_outliers = np.sum(outlier_mask)
    n_total = len(labels)
    outlier_percentage = (n_outliers / n_total) * 100 if n_total > 0 else 0

    # 对于DBSCAN，计算簇的数量
    if algorithm == "DBSCAN":
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    else:
        n_clusters = None

    # 摘要信息
    st.subheader("检测结果摘要")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("处理的总样本数", n_total)
    with col2:
        st.metric("检测到的异常点数", n_outliers,
                  delta=f"{outlier_percentage:.2f}%", delta_color="inverse")
    with col3:
        if n_clusters is not None:
            st.metric("检测到的有效聚类数", n_clusters)
        else:
            st.metric("检测算法", algorithm)

    # 异常点数据展示
    st.subheader("异常点数据 (原始行)")
    outlier_indices_in_original = processed_index[outlier_mask]
    outlier_data_original_order = original_data_aligned.loc[outlier_indices_in_original].copy()

    if not outlier_data_original_order.empty:
        st.dataframe(outlier_data_original_order.head(10))

        # 下载按钮
        csv_link = get_download_link(
            outlier_data_original_order,
            f"outlier_data_{algorithm}_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.csv",
            "下载异常点数据 (CSV)"
        )
        st.markdown(csv_link, unsafe_allow_html=True)
    else:
        st.info("未检测到异常点。")

    # 可视化
    st.subheader("可视化")
    plot_data_df = original_data_aligned.loc[processed_index, st.session_state.outlier_selected_features]

    if plot_data_df.empty:
        st.warning("没有可用于可视化的数据。")
    else:
        fig = plot_outliers_2d(plot_data_df, labels, method_name=algorithm)
        st.pyplot(fig)

        # 图表下载
        try:
            buffered = BytesIO()
            fig.savefig(buffered, format="png", dpi=100, bbox_inches='tight')
            img_str = base64.b64encode(buffered.getvalue()).decode()
            plt.close(fig)
            img_filename = f"outlier_visualization_{algorithm}_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.png"
            href = f'<a href="data:image/png;base64,{img_str}" download="{img_filename}">下载可视化图表</a>'
            st.markdown(href, unsafe_allow_html=True)
        except Exception as download_e:
            st.error(f"生成图表下载链接时出错: {download_e}")


# --- 主函数入口 (用于独立测试) ---
if __name__ == "__main__":
    st.set_page_config(page_title="异常点检测", layout="wide")
    show_outlier_detection_page()