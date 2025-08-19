# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
# Import balancing methods from imbalanced-learn
# Make sure to install it: pip install imbalanced-learn
try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler, NearMiss
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    # Define dummy classes if imblearn is not installed to avoid errors
    class RandomOverSampler: pass
    class SMOTE: pass
    class ADASYN: pass
    class RandomUnderSampler: pass
    class NearMiss: pass

import io
import base64
import platform
import matplotlib.font_manager as fm
import os

balancing_options = {
    'none': "不进行处理",
    'random_over': "随机过采样 (RandomOverSampler)",
    'smote': "SMOTE (合成少数类过采样技术)",
    'adasyn': "ADASYN (自适应合成抽样)",
    'random_under': "随机欠采样 (RandomUnderSampler)",
    'nearmiss': "NearMiss (基于距离的欠采样)",
}

# --- Font Setup (Reusing from other modules) ---
def setup_chinese_font():
    """Set up Chinese font support for matplotlib."""
    system = platform.system()
    font_candidates = []
    if system == 'Windows':
        font_candidates = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']
    elif system == 'Darwin': # macOS
        font_candidates = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Apple LiGothic Medium']
    else: # Linux and other systems
        font_candidates = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'Droid Sans Fallback']

    font_candidates.extend(['DejaVu Sans', 'Arial', 'Helvetica']) # Fallback fonts

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

# --- Helper Functions ---
def apply_plot_style(ax):
    """Apply consistent styling to matplotlib axes."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.6, color='#bdc3c7')
    ax.tick_params(axis='both', which='major', labelsize=9, colors='#34495e')
    ax.xaxis.label.set_fontsize(10)
    ax.yaxis.label.set_fontsize(10)
    ax.title.set_fontsize(12)
    ax.title.set_fontweight('bold')
    ax.xaxis.label.set_fontweight('bold')
    ax.yaxis.label.set_fontweight('bold')
    # Apply Chinese font if available
    font_kwargs = {'fontproperties': FONT_PROP} if FONT_PROP else {}
    if font_kwargs:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(FONT_PROP)
        ax.xaxis.label.set_fontproperties(FONT_PROP)
        ax.yaxis.label.set_fontproperties(FONT_PROP)
        ax.title.set_fontproperties(FONT_PROP)
    return ax

def plot_class_distribution(y, title="类别分布"):
    """Plots the distribution of classes."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=80)
    apply_plot_style(ax)
    counts = y.value_counts().sort_index()
    bars = ax.bar(counts.index.astype(str), counts.values, color=plt.cm.viridis(np.linspace(0, 1, len(counts))))
    ax.set_title(title, fontproperties=FONT_PROP if FONT_PROP else None)
    ax.set_xlabel("类别", fontproperties=FONT_PROP if FONT_PROP else None)
    ax.set_ylabel("样本数量", fontproperties=FONT_PROP if FONT_PROP else None)
    # Add counts on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    return fig

def get_download_link_csv(df, filename, text):
    """Generates a link to download a DataFrame as a CSV file."""
    csv = df.to_csv(index=False, encoding='utf-8-sig')
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# --- Balancing Functions ---
def apply_balancing(X, y, method='random_over', random_state=42, **kwargs):
    """Applies the selected balancing method."""
    if not IMBLEARN_AVAILABLE:
        st.error("需要安装 'imbalanced-learn' 库来执行数据平衡。请运行: pip install imbalanced-learn")
        return None, None

    sampler = None
    try:
        if method == 'random_over':
            sampler = RandomOverSampler(random_state=random_state)
        elif method == 'smote':
            # SMOTE might require specific k_neighbors depending on minority class size
            n_neighbors = kwargs.get('smote_k_neighbors', 5)
            # Ensure k_neighbors is less than the number of samples in the smallest class
            min_class_count = y.value_counts().min()
            if n_neighbors >= min_class_count:
                 n_neighbors = max(1, min_class_count - 1) # Adjust k_neighbors
                 st.warning(f"SMOTE 的 k_neighbors 已调整为 {n_neighbors} 以适应最小类别样本数。")
            sampler = SMOTE(random_state=random_state, k_neighbors=n_neighbors)
        elif method == 'adasyn':
            n_neighbors = kwargs.get('adasyn_n_neighbors', 5)
            min_class_count = y.value_counts().min()
            if n_neighbors >= min_class_count:
                 n_neighbors = max(1, min_class_count - 1)
                 st.warning(f"ADASYN 的 n_neighbors 已调整为 {n_neighbors} 以适应最小类别样本数。")
            sampler = ADASYN(random_state=random_state, n_neighbors=n_neighbors)
        elif method == 'random_under':
            sampler = RandomUnderSampler(random_state=random_state)
        elif method == 'nearmiss':
            version = kwargs.get('nearmiss_version', 1)
            n_neighbors = kwargs.get('nearmiss_n_neighbors', 3)
            sampler = NearMiss(version=version, n_neighbors=n_neighbors)
        else:
            st.error(f"未知的平衡方法: {method}")
            return X, y # Return original data

        X_resampled, y_resampled = sampler.fit_resample(X, y)
        return X_resampled, y_resampled

    except ValueError as ve:
         st.error(f"应用平衡方法 '{method}' 时出错: {ve}")
         st.info("这通常发生在类别样本数过少，无法应用所选方法（例如SMOTE/ADASYN的邻居数大于最小类别样本数）。请尝试其他方法或检查数据。")
         return X, y # Return original data on error
    except Exception as e:
        st.error(f"应用平衡方法 '{method}' 时发生意外错误: {e}")
        import traceback
        st.code(traceback.format_exc())
        return X, y # Return original data on error


# --- Streamlit UI Functions ---
def initialize_balancing_state():
    """Initialize session state variables for the data balancing page."""
    defaults = {
        'db_data': None,            # Original uploaded data
        'db_X': None,               # Features DataFrame
        'db_y': None,               # Target Series
        'db_target_col': None,      # Selected target column name
        'db_balanced_X': None,      # Balanced features
        'db_balanced_y': None,      # Balanced target
        'db_class_counts_before': None, # Class counts before balancing
        'db_class_counts_after': None,  # Class counts after balancing
        'db_balancing_method': 'random_over', # Default balancing method
        # Parameters for specific methods
        'db_smote_k_neighbors': 5,
        'db_adasyn_n_neighbors': 5,
        'db_nearmiss_version': 1,
        'db_nearmiss_n_neighbors': 3,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def show_balancing_page():
    """Main function to display the data balancing handling page."""
    if not IMBLEARN_AVAILABLE:
        st.error("错误：缺少 `imbalanced-learn` 库。请在终端运行 `pip install imbalanced-learn` 来安装它，然后重新启动应用。")
        st.stop() # Stop execution if library is missing

    initialize_balancing_state()

    st.title("⚖️ 数据平衡处理")
    st.markdown("---")
    st.info("上传分类数据，分析类别分布，并应用采样方法来处理类别不平衡问题。")

    # --- Create Tabs ---
    tab1, tab2, tab3 = st.tabs(["📁 数据导入与分析", "⚙️ 平衡方法选择", "📊 平衡结果展示"])

    with tab1:
        create_balancing_data_import_analysis()

    with tab2:
        create_balancing_method_selection()

    with tab3:
        create_balancing_results_section()

def create_balancing_data_import_analysis():
    """UI section for data import and class distribution analysis."""
    st.header("1. 数据导入与类别分析")

    uploaded_file = st.file_uploader("上传包含特征和目标列的文件 (CSV/Excel)", type=["csv", "xlsx", "xls"], key="db_uploader")

    if uploaded_file:
        if st.button("加载并分析数据", key="db_load_analyze_btn"):
            with st.spinner("正在加载和分析数据..."):
                try:
                    # Load data
                    data = pd.read_csv(uploaded_file) if uploaded_file.name.lower().endswith('.csv') else pd.read_excel(uploaded_file)

                    # Store original data
                    st.session_state.db_data = data
                    st.session_state.db_X = None
                    st.session_state.db_y = None
                    st.session_state.db_target_col = None
                    st.session_state.db_balanced_X = None
                    st.session_state.db_balanced_y = None
                    st.session_state.db_class_counts_before = None
                    st.session_state.db_class_counts_after = None

                    st.success(f"成功加载文件: {uploaded_file.name} ({len(data)} 行, {len(data.columns)} 列)")
                    st.rerun() # Rerun to update target selection

                except Exception as e:
                    st.error(f"加载数据时出错: {e}")
                    st.session_state.db_data = None

    # Display analysis results if data is loaded
    if st.session_state.db_data is not None:
        st.subheader("数据预览 (前5行)")
        st.dataframe(st.session_state.db_data.head())

        # --- Target Column Selection ---
        st.subheader("选择目标列 (类别标签)")
        all_columns = st.session_state.db_data.columns.tolist()
        # Try to guess a categorical column as default
        potential_targets = [col for col in all_columns if st.session_state.db_data[col].nunique() < 20] # Heuristic
        default_target_index = 0
        if st.session_state.db_target_col and st.session_state.db_target_col in all_columns:
             try:
                  default_target_index = all_columns.index(st.session_state.db_target_col)
             except ValueError:
                  default_target_index = 0
        elif potential_targets:
             try:
                  # Try to find common target names
                  common_names = ['target', 'label', 'class', 'category']
                  found = False
                  for name in common_names:
                       if name in potential_targets:
                            default_target_index = all_columns.index(name)
                            found = True
                            break
                  if not found: default_target_index = all_columns.index(potential_targets[-1]) # Default to last potential target
             except ValueError:
                   default_target_index = 0


        selected_target = st.selectbox(
            "选择包含类别标签的目标列:",
            options=all_columns,
            index=default_target_index,
            key="db_target_select"
        )

        if selected_target:
            st.session_state.db_target_col = selected_target
            try:
                y = st.session_state.db_data[selected_target]
                # Check if target is suitable (not too many unique values)
                if y.nunique() > 50:
                     st.warning(f"警告：目标列 '{selected_target}' 有超过 50 个唯一值，可能不适合分类或平衡。")
                elif y.isnull().any():
                     st.warning(f"警告：目标列 '{selected_target}' 包含缺失值。请先处理缺失值。")
                else:
                    st.session_state.db_y = y
                    # Separate features (X) - exclude target column
                    st.session_state.db_X = st.session_state.db_data.drop(columns=[selected_target])
                    # Analyze class distribution
                    st.session_state.db_class_counts_before = y.value_counts()

                    st.subheader("原始类别分布")
                    st.dataframe(st.session_state.db_class_counts_before.reset_index().rename(columns={'index': '类别', selected_target: '数量'}))

                    # Plot distribution
                    try:
                        fig = plot_class_distribution(y, title="原始类别分布")
                        st.pyplot(fig)
                    except Exception as plot_e:
                        st.error(f"绘制类别分布图时出错: {plot_e}")
            except KeyError:
                 st.error(f"无法找到列 '{selected_target}'。")
                 st.session_state.db_y = None
                 st.session_state.db_X = None
                 st.session_state.db_class_counts_before = None
            except Exception as e:
                 st.error(f"处理目标列时出错: {e}")
                 st.session_state.db_y = None
                 st.session_state.db_X = None
                 st.session_state.db_class_counts_before = None
        else:
            st.warning("请选择一个目标列以进行分析。")
            st.session_state.db_y = None
            st.session_state.db_X = None
            st.session_state.db_class_counts_before = None


def create_balancing_method_selection():
    """UI section for selecting balancing method and parameters."""
    st.header("2. 平衡方法选择")

    if st.session_state.db_X is None or st.session_state.db_y is None:
        st.info("请先在“数据导入与分析”选项卡中加载数据并选择目标列。")
        return

    # --- Select Balancing Method ---
    st.subheader("选择平衡方法")
    balancing_options = {
        'none': "不进行处理",
        'random_over': "随机过采样 (RandomOverSampler)",
        'smote': "SMOTE (合成少数类过采样技术)",
        'adasyn': "ADASYN (自适应合成抽样)",
        'random_under': "随机欠采样 (RandomUnderSampler)",
        'nearmiss': "NearMiss (基于距离的欠采样)",
    }

    st.session_state.db_balancing_method = st.radio(
        "选择数据平衡策略:",
        options=list(balancing_options.keys()),
        format_func=lambda x: balancing_options[x],
        key="db_method_radio"
    )

    # --- Display Method Descriptions and Parameters ---
    method = st.session_state.db_balancing_method
    if method != 'none':
        with st.expander(f"关于 **{balancing_options[method]}** 的说明与参数"):
            if method == 'random_over':
                st.markdown("""
                - **类型**: 过采样
                - **简介**: 随机复制少数类样本，直到达到与多数类相同的数量。
                - **适用范围**: 简单易懂，适用于各种数据集。
                - **优点**: 实现简单，不丢失信息。
                - **缺点**: 可能导致过拟合，因为只是简单复制样本，没有产生新信息。
                """)
            elif method == 'smote':
                st.markdown("""
                - **类型**: 过采样
                - **简介**: 为少数类合成新的样本。对每个少数类样本，选择其k个近邻，然后在该样本与其近邻之间的连线上随机生成新样本。
                - **适用范围**: 适用于数值型特征，是处理不平衡问题的常用且有效方法。
                - **优点**: 产生新样本，提供更多信息，通常比随机过采样效果更好，能缓解过拟合。
                - **缺点**: 对高维数据效果可能下降；可能生成噪声样本或模糊类别边界；对参数k敏感。
                """)
                st.session_state.db_smote_k_neighbors = st.slider(
                    "k_neighbors (SMOTE近邻数)", min_value=1, max_value=20,
                    value=st.session_state.db_smote_k_neighbors, step=1, key="db_smote_k",
                    help="用于合成样本的近邻数量。注意：必须小于最小类别的样本数。"
                )
            elif method == 'adasyn':
                st.markdown("""
                - **类型**: 过采样
                - **简介**: 自适应合成抽样。与SMOTE类似，但更关注那些难以学习的少数类样本（即邻域中多数类样本比例高的样本），为这些样本生成更多合成数据。
                - **适用范围**: 适用于数值型特征，特别是当类别边界复杂时。
                - **优点**: 能自适应地在更需要的地方生成样本。
                - **缺点**: 对噪声数据更敏感；实现比SMOTE复杂；对参数n_neighbors敏感。
                """)
                st.session_state.db_adasyn_n_neighbors = st.slider(
                    "n_neighbors (ADASYN近邻数)", min_value=1, max_value=20,
                    value=st.session_state.db_adasyn_n_neighbors, step=1, key="db_adasyn_k",
                     help="用于确定样本密度的近邻数量。注意：必须小于最小类别的样本数。"
                )
            elif method == 'random_under':
                st.markdown("""
                - **类型**: 欠采样
                - **简介**: 随机删除多数类样本，直到其数量与少数类相同（或达到指定比例）。
                - **适用范围**: 当数据集非常大，且多数类样本包含大量冗余信息时。
                - **优点**: 实现简单；可以显著减少数据集大小，加快训练速度。
                - **缺点**: 可能丢失多数类的重要信息，导致模型性能下降。
                """)
            elif method == 'nearmiss':
                st.markdown("""
                - **类型**: 欠采样
                - **简介**: 基于距离选择要保留的多数类样本。有不同版本：
                    - **Version 1**: 选择与最近的k个少数类样本平均距离最小的多数类样本。
                    - **Version 2**: 选择与最远的k个少数类样本平均距离最小的多数类样本。
                    - **Version 3**: 对每个少数类样本，保留其最近的k个多数类样本。
                - **适用范围**: 当多数类和少数类边界清晰时可能有效。
                - **优点**: 尝试保留靠近边界的多数类信息。
                - **缺点**: 对噪声和异常值敏感；计算成本较高；可能扭曲数据分布。
                """)
                st.session_state.db_nearmiss_version = st.selectbox(
                    "NearMiss 版本", options=[1, 2, 3],
                    index=st.session_state.db_nearmiss_version - 1, key="db_nearmiss_v"
                )
                st.session_state.db_nearmiss_n_neighbors = st.slider(
                    "n_neighbors (NearMiss近邻数)", min_value=1, max_value=20,
                    value=st.session_state.db_nearmiss_n_neighbors, step=1, key="db_nearmiss_k"
                )

    # --- Apply Balancing Button ---
    if st.button("应用平衡方法", key="db_apply_btn", type="primary", disabled=(method == 'none')):
        run_balancing()

def run_balancing():
    """Runs the selected balancing method."""
    X = st.session_state.db_X
    y = st.session_state.db_y
    method = st.session_state.db_balancing_method

    if X is None or y is None:
        st.error("无法执行平衡：数据或目标列未准备好。")
        return
    if method == 'none':
        st.info("选择了“不进行处理”，数据未改变。")
        st.session_state.db_balanced_X = X.copy()
        st.session_state.db_balanced_y = y.copy()
        st.session_state.db_class_counts_after = y.value_counts()
        return

    # Check if features are numeric for methods that require it
    numeric_methods = ['smote', 'adasyn', 'nearmiss'] # KNN also implicitly requires numeric
    if method in numeric_methods:
        non_numeric_cols = X.select_dtypes(exclude=np.number).columns.tolist()
        if non_numeric_cols:
            st.error(f"选择的方法 '{method}' 仅适用于数值特征，但数据包含非数值列: {', '.join(non_numeric_cols)}。请先进行编码或选择其他方法。")
            return

    st.info(f"正在应用 '{balancing_options.get(method, method)}' 方法...")
    with st.spinner("正在平衡数据..."):
        kwargs = {}
        if method == 'smote':
            kwargs['smote_k_neighbors'] = st.session_state.db_smote_k_neighbors
        elif method == 'adasyn':
            kwargs['adasyn_n_neighbors'] = st.session_state.db_adasyn_n_neighbors
        elif method == 'nearmiss':
            kwargs['nearmiss_version'] = st.session_state.db_nearmiss_version
            kwargs['nearmiss_n_neighbors'] = st.session_state.db_nearmiss_n_neighbors

        # Apply balancing
        X_res, y_res = apply_balancing(X, y, method, **kwargs)

        if X_res is not None and y_res is not None:
            # Store results
            st.session_state.db_balanced_X = X_res
            st.session_state.db_balanced_y = y_res
            st.session_state.db_class_counts_after = y_res.value_counts()
            st.success("数据平衡处理完成！请前往“平衡结果展示”选项卡查看。")
        else:
            # Error occurred during balancing (handled in apply_balancing)
            st.session_state.db_balanced_X = None
            st.session_state.db_balanced_y = None
            st.session_state.db_class_counts_after = None


def create_balancing_results_section():
    """UI section to display the data after balancing."""
    st.header("3. 平衡结果展示")

    if st.session_state.db_balanced_X is None or st.session_state.db_balanced_y is None:
        st.info("请先在“平衡方法选择”选项卡中应用平衡方法。")
        return

    st.subheader("平衡后类别分布")
    if st.session_state.db_class_counts_after is not None:
        st.dataframe(st.session_state.db_class_counts_after.reset_index().rename(columns={'index': '类别', st.session_state.db_target_col: '数量'}))
        try:
            fig = plot_class_distribution(st.session_state.db_balanced_y, title="平衡后类别分布")
            st.pyplot(fig)
        except Exception as plot_e:
            st.error(f"绘制平衡后类别分布图时出错: {plot_e}")
    else:
        st.warning("未能获取平衡后的类别计数。")


    st.subheader("平衡后的数据预览 (前5行)")
    # Combine X and y for preview
    preview_df = st.session_state.db_balanced_X.head().copy()
    preview_df[st.session_state.db_target_col] = st.session_state.db_balanced_y.head()
    st.dataframe(preview_df)

    st.info(f"平衡后的数据集包含 {len(st.session_state.db_balanced_X)} 行。")

    # Download button for balanced data
    st.subheader("下载平衡后的数据")
    if st.button("准备下载文件", key="db_prep_download"):
        try:
            # Combine X and y for download
            df_to_download = st.session_state.db_balanced_X.copy()
            df_to_download[st.session_state.db_target_col] = st.session_state.db_balanced_y
            method_name = st.session_state.db_balancing_method.replace('_', '-')
            filename = f"balanced_data_{method_name}.csv"
            download_link = get_download_link_csv(df_to_download, filename, f"点击下载 {filename}")
            st.markdown(download_link, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"准备下载文件时出错: {e}")

# --- Main function entry point (for potential direct script run) ---
if __name__ == "__main__":
    # This part is optional, allows running this module directly for testing
    st.set_page_config(layout="wide", page_title="数据平衡处理")
    show_balancing_page()
