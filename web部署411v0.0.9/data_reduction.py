# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import base64
from io import BytesIO
from sklearn.decomposition import PCA, TruncatedSVD, FastICA, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE, Isomap
from sklearn.preprocessing import StandardScaler, LabelEncoder
import platform
import matplotlib.font_manager as fm
import datetime
import warnings

warnings.filterwarnings('ignore')

try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    st.warning("UMAP库未安装，UMAP功能不可用。请运行：pip install umap-learn")


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
            font_path = fm.findfind(fm.FontProperties(family=font_name))
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


# --- 降维算法信息 ---
ALGORITHM_INFO = {
    "PCA": {
        "name": "主成分分析",
        "description": "通过正交变换将数据投影到主成分方向",
        "suitable_for": [
            "• 高维线性相关数据",
            "• 特征压缩和噪声降低",
            "• 数据预处理和可视化",
            "• 需要解释主成分含义的场景"
        ],
        "pros": [
            "• 线性变换，计算简单快速",
            "• 保留最大方差信息",
            "• 可以解释主成分含义",
            "• 适合线性结构数据"
        ],
        "cons": [
            "• 对非线性结构效果差",
            "• 受异常值影响大",
            "• 需要标准化预处理",
            "• 主成分解释需要专业知识"
        ],
        "params": ["n_components"]
    },
    "TSNE": {
        "name": "t-SNE",
        "description": "基于概率分布的非线性降维技术",
        "suitable_for": [
            "• 高维数据可视化",
            "• 复杂非线性结构数据",
            "• 聚类结构展示",
            "• 科学研究和数据探索"
        ],
        "pros": [
            "• 非常适合可视化",
            "• 能保持局部邻域结构",
            "• 处理非线性数据出色",
            "• 对聚类结构展示清晰"
        ],
        "cons": [
            "• 计算复杂度高",
            "• 超参数敏感",
            "• 不适合新数据投影",
            "• 结果可能不稳定"
        ],
        "params": ["n_components", "perplexity", "learning_rate", "n_iter"]
    },
    "UMAP": {
        "name": "UMAP",
        "description": "均匀流形逼近投影，现代非线性降维算法",
        "suitable_for": [
            "• 大规模数据降维",
            "• 保持全局结构",
            "• 聚类和分类预处理",
            "• 实时数据处理"
        ],
        "pros": [
            "• 比t-SNE速度快",
            "• 保持全局和局部结构",
            "• 支持新数据投影",
            "• 参数较少且稳定"
        ],
        "cons": [
            "• 需要额外安装库",
            "• 理论较新，文档较少",
            "• 对密度较敏感"
        ],
        "params": ["n_components", "n_neighbors", "min_dist"]
    },
    "LDA": {
        "name": "线性判别分析",
        "description": "有监督的线性降维算法",
        "suitable_for": [
            "• 分类预处理",
            "• 有标签的数据",
            "• 特征选择",
            "• 模式识别任务"
        ],
        "pros": [
            "• 有监督，效果更好",
            "• 最大化类间差异",
            "• 计算效率高",
            "• 直接用于分类"
        ],
        "cons": [
            "• 需要标签信息",
            "• 假设高斯分布",
            "• 线性边界限制",
            "• 维度受类别数限制"
        ],
        "params": ["n_components"]
    },
    "ICA": {
        "name": "独立成分分析",
        "description": "盲源分离技术，挖掘独立信号",
        "suitable_for": [
            "• 信号处理",
            "• 特征提取",
            "• 数据去噪",
            "• 图像处理"
        ],
        "pros": [
            "• 找到统计独立的成分",
            "• 适合信号分离",
            "• 处理非高斯数据",
            "• 可用于特征提取"
        ],
        "cons": [
            "• 需要假设统计独立",
            "• 对异常值敏感",
            "• 结果可能需要解释"
        ],
        "params": ["n_components", "max_iter"]
    },
    "Isomap": {
        "name": "等距映射",
        "description": "基于测地距离的非线性降维",
        "suitable_for": [
            "• 流形学习",
            "• 非线性数据结构",
            "• 保持测地距离",
            "• 有内在低维结构的数据"
        ],
        "pros": [
            "• 保持测地距离",
            "• 捕捉非线性结构",
            "• 理论基础扎实",
            "• 适合流形数据"
        ],
        "cons": [
            "• 需要选择邻居数",
            "• 对异常值敏感",
            "• 计算复杂度高",
            "• 需要数据有流形结构"
        ],
        "params": ["n_components", "n_neighbors"]
    }
}


# --- Streamlit UI 函数 ---
def initialize_reduction_session_state():
    """初始化降维页面的会话状态"""
    defaults = {
        'reduction_data': None,
        'reduction_original_data': None,
        'reduction_column_names': [],
        'reduction_selected_features': [],
        'reduction_target_column': None,
        'reduction_has_target': False,
        'reduction_results': None,
        'reduction_algorithm': 'PCA',
        'reduction_normalize': True,
        # PCA参数
        'reduction_pca_components': 2,
        # t-SNE参数
        'reduction_tsne_components': 2,
        'reduction_tsne_perplexity': 30.0,
        'reduction_tsne_learning_rate': 200.0,
        'reduction_tsne_n_iter': 1000,
        # UMAP参数
        'reduction_umap_components': 2,
        'reduction_umap_n_neighbors': 15,
        'reduction_umap_min_dist': 0.1,
        # LDA参数
        'reduction_lda_components': 2,
        # ICA参数
        'reduction_ica_components': 2,
        'reduction_ica_max_iter': 200,
        # Isomap参数
        'reduction_isomap_components': 2,
        'reduction_isomap_n_neighbors': 5,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def show_data_reduction_page():
    """显示数据降维页面"""
    initialize_reduction_session_state()

    st.title("📉 数据降维")

    # 算法选择区域
    st.markdown("## 📚 降维算法选择与介绍")

    algo_col1, algo_col2 = st.columns([1, 2])

    with algo_col1:
        available_algorithms = list(ALGORITHM_INFO.keys())
        if not UMAP_AVAILABLE:
            available_algorithms.remove("UMAP")

        st.session_state.reduction_algorithm = st.selectbox(
            "选择降维算法",
            options=available_algorithms,
            index=available_algorithms.index(
                st.session_state.reduction_algorithm) if st.session_state.reduction_algorithm in available_algorithms else 0,
            key="reduction_algorithm_select"
        )

    with algo_col2:
        algo_info = ALGORITHM_INFO[st.session_state.reduction_algorithm]
        st.markdown(f"### {algo_info['name']} 算法")
        st.markdown(f"**描述**: {algo_info['description']}")

    # 详细介绍
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
        create_reduction_data_import_section()

    with tab2:
        create_reduction_params_section()

    with tab3:
        create_reduction_results_section()


def create_reduction_data_import_section():
    """创建数据导入和特征选择部分"""
    st.header("1. 数据导入与特征选择")

    uploaded_file = st.file_uploader("上传数据文件 (CSV/Excel)",
                                     type=["csv", "xlsx", "xls"], key="reduction_uploader")

    if uploaded_file:
        if st.button("加载数据", key="reduction_load_btn"):
            with st.spinner("正在加载和处理数据..."):
                try:
                    # 加载数据
                    if uploaded_file.name.lower().endswith('.csv'):
                        data_original = pd.read_csv(uploaded_file)
                    else:
                        data_original = pd.read_excel(uploaded_file)

                    # 处理缺失值
                    initial_rows = len(data_original)
                    data_cleaned = data_original.dropna()
                    cleaned_rows = len(data_cleaned)

                    if cleaned_rows < initial_rows:
                        st.warning(f"移除了 {initial_rows - cleaned_rows} 行包含缺失值的数据。")

                    if data_cleaned.empty:
                        st.error("处理缺失值后数据为空。")
                        return

                    # 分离数值列和其他列
                    numeric_cols = data_cleaned.select_dtypes(include=np.number).columns.tolist()
                    categorical_cols = data_cleaned.select_dtypes(exclude=np.number).columns.tolist()

                    if not numeric_cols:
                        st.error("上传的文件中未找到数值列。")
                        return

                    # 存储数据
                    st.session_state.reduction_data = data_cleaned
                    st.session_state.reduction_original_data = data_original
                    st.session_state.reduction_column_names = data_cleaned.columns.tolist()
                    st.session_state.reduction_selected_features = numeric_cols
                    st.session_state.reduction_results = None

                    st.success(f"成功加载数据: {cleaned_rows} 行, {len(data_cleaned.columns)} 列")
                    st.info(f"找到 {len(numeric_cols)} 个数值列, {len(categorical_cols)} 个非数值列")

                except Exception as e:
                    st.error(f"加载数据时出错: {e}")

    # 数据预览和特征选择
    if st.session_state.reduction_data is not None:
        st.subheader("数据预览")
        st.dataframe(st.session_state.reduction_data.head())

        # 特征选择
        st.subheader("特征选择")

        # 数值特征选择
        numeric_cols = st.session_state.reduction_data.select_dtypes(include=np.number).columns.tolist()
        st.session_state.reduction_selected_features = st.multiselect(
            "选择用于降维的特征列",
            numeric_cols,
            default=[col for col in st.session_state.reduction_selected_features if col in numeric_cols],
            key="reduction_feature_select"
        )

        # 目标列选择（用于有监督算法）
        if st.session_state.reduction_algorithm == "LDA":
            st.session_state.reduction_has_target = True
            all_cols = st.session_state.reduction_data.columns.tolist()
            non_feature_cols = [col for col in all_cols if col not in st.session_state.reduction_selected_features]

            if non_feature_cols:
                st.session_state.reduction_target_column = st.selectbox(
                    "选择目标列（用于LDA）",
                    options=[None] + non_feature_cols,
                    index=([None] + non_feature_cols).index(
                        st.session_state.reduction_target_column) if st.session_state.reduction_target_column in [
                        None] + non_feature_cols else 0,
                    key="reduction_target_select"
                )
            else:
                st.warning("没有可用作目标列的列。")
        else:
            st.session_state.reduction_has_target = False
            st.session_state.reduction_target_column = None

        if not st.session_state.reduction_selected_features:
            st.warning("请至少选择一个特征列。")
        else:
            st.info(f"已选择 {len(st.session_state.reduction_selected_features)} 个特征")


def create_reduction_params_section():
    """创建参数设置部分"""
    st.header("2. 参数设置")

    if st.session_state.reduction_data is None:
        st.info("请先在数据导入选项卡中加载数据。")
        return

    if not st.session_state.reduction_selected_features:
        st.warning("请先在数据导入选项卡中选择特征列。")
        return

    # 预处理选项
    st.subheader("预处理")
    st.session_state.reduction_normalize = st.checkbox(
        "标准化特征 (推荐)",
        value=st.session_state.reduction_normalize,
        key="reduction_norm_cb"
    )

    # 算法特定参数
    st.subheader(f"{st.session_state.reduction_algorithm} 算法参数")

    # 检查LDA的特殊要求
    if st.session_state.reduction_algorithm == "LDA":
        if not st.session_state.reduction_target_column:
            st.error("LDA算法需要目标列。请在数据导入选项卡中选择目标列。")
            return

    # 根据算法显示参数设置
    if st.session_state.reduction_algorithm == "PCA":
        create_pca_params()
    elif st.session_state.reduction_algorithm == "TSNE":
        create_tsne_params()
    elif st.session_state.reduction_algorithm == "UMAP":
        create_umap_params()
    elif st.session_state.reduction_algorithm == "LDA":
        create_lda_params()
    elif st.session_state.reduction_algorithm == "ICA":
        create_ica_params()
    elif st.session_state.reduction_algorithm == "Isomap":
        create_isomap_params()

    # 运行按钮
    can_run = st.session_state.reduction_data is not None and st.session_state.reduction_selected_features
    if st.session_state.reduction_algorithm == "LDA":
        can_run = can_run and st.session_state.reduction_target_column is not None

    if st.button(f"运行 {st.session_state.reduction_algorithm} 降维",
                 type="primary", key="run_reduction_btn", disabled=not can_run):
        run_dimension_reduction()


def create_pca_params():
    """创建PCA参数设置"""
    st.markdown("选择主成分数量。建议先查看方差解释率，然后选择合适的成分数。")

    max_components = min(
        len(st.session_state.reduction_selected_features),
        len(st.session_state.reduction_data.dropna())
    )

    st.session_state.reduction_pca_components = st.slider(
        "主成分数量",
        min_value=1,
        max_value=max_components,
        value=min(st.session_state.reduction_pca_components, max_components),
        key="pca_components_slider"
    )


def create_tsne_params():
    """创建t-SNE参数设置"""
    col1, col2 = st.columns(2)

    with col1:
        st.session_state.reduction_tsne_components = st.selectbox(
            "降维目标维度",
            options=[2, 3],
            index=[2, 3].index(st.session_state.reduction_tsne_components),
            key="tsne_components_select"
        )

        st.session_state.reduction_tsne_perplexity = st.number_input(
            "Perplexity",
            min_value=5.0,
            max_value=100.0,
            value=st.session_state.reduction_tsne_perplexity,
            step=1.0,
            key="tsne_perplexity_input",
            help="控制局部和全局结构的平衡，通常在5-50之间"
        )

    with col2:
        st.session_state.reduction_tsne_learning_rate = st.number_input(
            "学习率",
            min_value=10.0,
            max_value=1000.0,
            value=st.session_state.reduction_tsne_learning_rate,
            step=10.0,
            key="tsne_lr_input",
            help="控制优化步长，通常在10-1000之间"
        )

        st.session_state.reduction_tsne_n_iter = st.number_input(
            "迭代次数",
            min_value=250,
            max_value=5000,
            value=st.session_state.reduction_tsne_n_iter,
            step=250,
            key="tsne_iter_input",
            help="优化迭代次数，至少250次"
        )


def create_umap_params():
    """创建UMAP参数设置"""
    if not UMAP_AVAILABLE:
        st.error("UMAP库未安装，请先安装：pip install umap-learn")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        st.session_state.reduction_umap_components = st.selectbox(
            "降维目标维度",
            options=[2, 3],
            index=[2, 3].index(st.session_state.reduction_umap_components),
            key="umap_components_select"
        )

    with col2:
        st.session_state.reduction_umap_n_neighbors = st.number_input(
            "邻居数量",
            min_value=2,
            max_value=100,
            value=st.session_state.reduction_umap_n_neighbors,
            step=1,
            key="umap_neighbors_input",
            help="控制局部结构保持的程度"
        )

    with col3:
        st.session_state.reduction_umap_min_dist = st.number_input(
            "最小距离",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.reduction_umap_min_dist,
            step=0.01,
            format="%.3f",
            key="umap_mindist_input",
            help="控制降维后点的间隔"
        )


def create_lda_params():
    """创建LDA参数设置"""
    if not st.session_state.reduction_target_column:
        st.error("请先在数据导入选项卡中选择目标列。")
        return

    # 计算最大可能的成分数
    target_values = st.session_state.reduction_data[st.session_state.reduction_target_column].unique()
    max_components = min(
        len(target_values) - 1,  # LDA最多可有类别数-1个成分
        len(st.session_state.reduction_selected_features)
    )

    if max_components <= 0:
        st.error("目标列的类别数太少，无法进行LDA降维。")
        return

    st.session_state.reduction_lda_components = st.slider(
        "降维目标维度",
        min_value=1,
        max_value=max_components,
        value=min(st.session_state.reduction_lda_components, max_components),
        key="lda_components_slider"
    )

    st.info(f"目标列 '{st.session_state.reduction_target_column}' 有 {len(target_values)} 个类别")


def create_ica_params():
    """创建ICA参数设置"""
    col1, col2 = st.columns(2)

    with col1:
        max_components = len(st.session_state.reduction_selected_features)
        st.session_state.reduction_ica_components = st.slider(
            "独立成分数量",
            min_value=1,
            max_value=max_components,
            value=min(st.session_state.reduction_ica_components, max_components),
            key="ica_components_slider"
        )

    with col2:
        st.session_state.reduction_ica_max_iter = st.number_input(
            "最大迭代次数",
            min_value=100,
            max_value=1000,
            value=st.session_state.reduction_ica_max_iter,
            step=50,
            key="ica_maxiter_input"
        )


def create_isomap_params():
    """创建Isomap参数设置"""
    col1, col2 = st.columns(2)

    with col1:
        max_components = len(st.session_state.reduction_selected_features)
        st.session_state.reduction_isomap_components = st.slider(
            "降维目标维度",
            min_value=1,
            max_value=max_components,
            value=min(st.session_state.reduction_isomap_components, max_components),
            key="isomap_components_slider"
        )

    with col2:
        max_neighbors = len(st.session_state.reduction_data) - 1
        st.session_state.reduction_isomap_n_neighbors = st.number_input(
            "邻居数量",
            min_value=2,
            max_value=min(100, max_neighbors),
            value=min(st.session_state.reduction_isomap_n_neighbors, max_neighbors),
            step=1,
            key="isomap_neighbors_input",
            help="构建邻域图所需的邻居数量"
        )


def run_dimension_reduction():
    """执行降维操作"""
    if st.session_state.reduction_data is None or not st.session_state.reduction_selected_features:
        st.error("请先加载数据并选择特征。")
        return

    with st.spinner(f"正在运行 {st.session_state.reduction_algorithm} 降维..."):
        try:
            # 准备数据
            X = st.session_state.reduction_data[st.session_state.reduction_selected_features].values

            # 标准化
            if st.session_state.reduction_normalize:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

            # 对于LDA，准备目标变量
            if st.session_state.reduction_algorithm == "LDA":
                y = st.session_state.reduction_data[st.session_state.reduction_target_column].values
                # 如果目标变量是文本，需要编码
                if y.dtype == object:
                    label_encoder = LabelEncoder()
                    y = label_encoder.fit_transform(y)
                else:
                    label_encoder = None
            else:
                y = None
                label_encoder = None

            # 执行降维
            if st.session_state.reduction_algorithm == "PCA":
                X_reduced, model = run_pca(X)
            elif st.session_state.reduction_algorithm == "TSNE":
                X_reduced, model = run_tsne(X)
            elif st.session_state.reduction_algorithm == "UMAP":
                X_reduced, model = run_umap_reduction(X)
            elif st.session_state.reduction_algorithm == "LDA":
                X_reduced, model = run_lda(X, y)
            elif st.session_state.reduction_algorithm == "ICA":
                X_reduced, model = run_ica(X)
            elif st.session_state.reduction_algorithm == "Isomap":
                X_reduced, model = run_isomap(X)

            # 保存结果
            st.session_state.reduction_results = {
                'X_reduced': X_reduced,
                'X_original': X,
                'model': model,
                'algorithm': st.session_state.reduction_algorithm,
                'feature_names': st.session_state.reduction_selected_features,
                'target': y,
                'target_column': st.session_state.reduction_target_column,
                'label_encoder': label_encoder,
                'n_components': X_reduced.shape[1]
            }

            st.success(f"{st.session_state.reduction_algorithm} 降维完成！")
            st.success(f"原始维度: {X.shape[1]} → 降维后维度: {X_reduced.shape[1]}")

        except Exception as e:
            st.error(f"运行 {st.session_state.reduction_algorithm} 时出错: {e}")
            import traceback
            st.code(traceback.format_exc())


def run_pca(X):
    """运行PCA降维"""
    pca = PCA(n_components=st.session_state.reduction_pca_components)
    X_reduced = pca.fit_transform(X)
    return X_reduced, pca


def run_tsne(X):
    """运行t-SNE降维"""
    tsne = TSNE(
        n_components=st.session_state.reduction_tsne_components,
        perplexity=st.session_state.reduction_tsne_perplexity,
        learning_rate=st.session_state.reduction_tsne_learning_rate,
        n_iter=st.session_state.reduction_tsne_n_iter,
        random_state=42
    )
    X_reduced = tsne.fit_transform(X)
    return X_reduced, tsne


def run_umap_reduction(X):
    """运行UMAP降维"""
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP未安装，请先安装：pip install umap-learn")

    reducer = umap.UMAP(
        n_components=st.session_state.reduction_umap_components,
        n_neighbors=st.session_state.reduction_umap_n_neighbors,
        min_dist=st.session_state.reduction_umap_min_dist,
        random_state=42
    )
    X_reduced = reducer.fit_transform(X)
    return X_reduced, reducer


def run_lda(X, y):
    """运行LDA降维"""
    lda = LinearDiscriminantAnalysis(n_components=st.session_state.reduction_lda_components)
    X_reduced = lda.fit_transform(X, y)
    return X_reduced, lda


def run_ica(X):
    """运行ICA降维"""
    ica = FastICA(
        n_components=st.session_state.reduction_ica_components,
        max_iter=st.session_state.reduction_ica_max_iter,
        random_state=42
    )
    X_reduced = ica.fit_transform(X)
    return X_reduced, ica


def run_isomap(X):
    """运行Isomap降维"""
    isomap = Isomap(
        n_components=st.session_state.reduction_isomap_components,
        n_neighbors=st.session_state.reduction_isomap_n_neighbors
    )
    X_reduced = isomap.fit_transform(X)
    return X_reduced, isomap


def create_reduction_results_section():
    """创建结果展示部分"""
    st.header("3. 结果展示")

    results = st.session_state.get('reduction_results')

    if results is None:
        st.info("请先运行降维算法。")
        return

    X_reduced = results['X_reduced']
    model = results['model']
    algorithm = results['algorithm']
    feature_names = results['feature_names']
    target = results.get('target')
    target_column = results.get('target_column')
    n_components = results['n_components']

    # 显示降维摘要
    st.subheader("降维结果摘要")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("原始维度", len(feature_names))
    with col2:
        st.metric("降维后维度", n_components)
    with col3:
        st.metric("降维算法", algorithm)

    # 算法特定的指标
    if algorithm == "PCA":
        st.subheader("主成分分析结果")
        explained_variance_ratio = model.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)

        # 方差解释率图表
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio,
               alpha=0.7, label='个体解释方差')
        ax.plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
                'ro-', label='累积解释方差')

        ax.set_xlabel('主成分')
        ax.set_ylabel('解释方差比率')
        ax.set_title('主成分方差解释率')
        ax.legend()
        apply_plot_style(ax)

        if FONT_PROP:
            ax.set_xlabel('主成分', fontproperties=FONT_PROP)
            ax.set_ylabel('解释方差比率', fontproperties=FONT_PROP)
            ax.set_title('主成分方差解释率', fontproperties=FONT_PROP)
            ax.legend(prop=FONT_PROP)

        st.pyplot(fig)
        plt.close()

        # 显示具体数值
        variance_df = pd.DataFrame({
            '主成分': [f'PC{i + 1}' for i in range(len(explained_variance_ratio))],
            '解释方差比率': explained_variance_ratio,
            '累积解释方差': cumulative_variance
        })
        st.dataframe(variance_df.round(4))

    # 降维结果可视化
    st.subheader("降维结果可视化")

    if n_components == 2:
        plot_2d_reduction(X_reduced, target, target_column, algorithm)
    elif n_components == 3:
        plot_3d_reduction(X_reduced, target, target_column, algorithm)
    else:
        st.info("当前只支持2D和3D可视化。")

    # 降维数据下载
    st.subheader("数据下载")

    # 创建降维后的DataFrame
    component_names = [f'{algorithm}_Component_{i + 1}' for i in range(n_components)]
    reduced_df = pd.DataFrame(X_reduced, columns=component_names)

    # 添加原始数据的非特征列
    original_data = st.session_state.reduction_data.reset_index(drop=True)
    for col in original_data.columns:
        if col not in feature_names:
            reduced_df[col] = original_data[col]

    # 下载按钮
    csv_link = get_download_link(
        reduced_df,
        f"dimensionality_reduction_{algorithm}_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.csv",
        "下载降维后数据 (CSV)"
    )
    st.markdown(csv_link, unsafe_allow_html=True)

    # 显示降维后数据预览
    st.subheader("降维后数据预览")
    st.dataframe(reduced_df.head())


def plot_2d_reduction(X_reduced, target, target_column, algorithm):
    """绘制2D降维结果"""
    fig, ax = plt.subplots(figsize=(10, 8))

    if target is not None:
        # 有目标变量，用颜色区分
        scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=target, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label=target_column or 'Target')
    else:
        # 无目标变量，统一颜色
        ax.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.7, color='#3498db')

    ax.set_xlabel(f'{algorithm} Component 1')
    ax.set_ylabel(f'{algorithm} Component 2')
    ax.set_title(f'{algorithm} 降维结果 (2D)')

    apply_plot_style(ax)

    if FONT_PROP:
        ax.set_xlabel(f'{algorithm} 成分 1', fontproperties=FONT_PROP)
        ax.set_ylabel(f'{algorithm} 成分 2', fontproperties=FONT_PROP)
        ax.set_title(f'{algorithm} 降维结果 (2D)', fontproperties=FONT_PROP)

    st.pyplot(fig)
    plt.close()


def plot_3d_reduction(X_reduced, target, target_column, algorithm):
    """绘制3D降维结果"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    if target is not None:
        # 有目标变量，用颜色区分
        scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2],
                             c=target, cmap='viridis', alpha=0.7)
        fig.colorbar(scatter, label=target_column or 'Target')
    else:
        # 无目标变量，统一颜色
        ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2],
                   alpha=0.7, color='#3498db')

    ax.set_xlabel(f'{algorithm} Component 1')
    ax.set_ylabel(f'{algorithm} Component 2')
    ax.set_zlabel(f'{algorithm} Component 3')
    ax.set_title(f'{algorithm} 降维结果 (3D)')

    if FONT_PROP:
        ax.set_xlabel(f'{algorithm} 成分 1', fontproperties=FONT_PROP)
        ax.set_ylabel(f'{algorithm} 成分 2', fontproperties=FONT_PROP)
        ax.set_zlabel(f'{algorithm} 成分 3', fontproperties=FONT_PROP)
        ax.set_title(f'{algorithm} 降维结果 (3D)', fontproperties=FONT_PROP)

    st.pyplot(fig)
    plt.close()


def get_download_link(df, filename, text):
    """生成CSV下载链接"""
    csv = df.to_csv(index=False, encoding='utf-8-sig')
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href


# --- 主函数入口 (用于独立测试) ---
if __name__ == "__main__":
    st.set_page_config(page_title="数据降维", layout="wide")
    show_data_reduction_page()