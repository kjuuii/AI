# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer
# Consider adding MICE if needed, it requires: pip install fancyimpute or iterativeimputer
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer
import io
import base64
import platform
import matplotlib.font_manager as fm
import os

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

def plot_missing_values_heatmap(df):
    """Create a heatmap visualization of missing values."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=80)
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax)
    ax.set_title("数据缺失值热力图", fontproperties=FONT_PROP if FONT_PROP else None)
    ax.set_xlabel("列名", fontproperties=FONT_PROP if FONT_PROP else None)
    ax.set_ylabel("样本索引", fontproperties=FONT_PROP if FONT_PROP else None)
    # Apply Chinese font to tick labels if needed
    if FONT_PROP:
        plt.setp(ax.get_xticklabels(), fontproperties=FONT_PROP, rotation=45, ha="right")
        plt.setp(ax.get_yticklabels(), fontproperties=FONT_PROP)
    else:
         plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    return fig

def get_download_link_csv(df, filename, text):
    """Generates a link to download a DataFrame as a CSV file."""
    csv = df.to_csv(index=False, encoding='utf-8-sig')
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# --- Imputation Functions ---
def impute_with_strategy(df, columns_to_impute, strategy='mean'):
    """Impute missing values using SimpleImputer (mean, median, most_frequent)."""
    imputer = SimpleImputer(strategy=strategy)
    df_imputed = df.copy()
    # Fit on selected columns and transform
    df_imputed[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])
    return df_imputed

def impute_with_knn(df, columns_to_impute, n_neighbors=5):
    """Impute missing values using KNNImputer."""
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_imputed = df.copy()
    # KNNImputer works best with scaled data, but for simplicity, apply directly here.
    # Consider adding scaling as an option.
    # Fit on selected columns and transform
    df_imputed[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])
    return df_imputed

# --- New Imputation Functions ---
def impute_with_zero(df, columns_to_impute):
    """Impute missing values with 0."""
    df_imputed = df.copy()
    df_imputed[columns_to_impute] = df_imputed[columns_to_impute].fillna(0)
    return df_imputed

def impute_with_ffill(df, columns_to_impute):
    """Impute missing values using forward fill."""
    df_imputed = df.copy()
    df_imputed[columns_to_impute] = df_imputed[columns_to_impute].ffill()
    # Note: ffill might leave NaNs at the beginning if the first value is NaN.
    # Consider adding a subsequent bfill or zero fill for those cases if needed.
    # For simplicity, we'll leave it as is for now.
    return df_imputed

def impute_with_bfill(df, columns_to_impute):
    """Impute missing values using backward fill."""
    df_imputed = df.copy()
    df_imputed[columns_to_impute] = df_imputed[columns_to_impute].bfill()
    # Note: bfill might leave NaNs at the end if the last value is NaN.
    # Consider adding a subsequent ffill or zero fill for those cases if needed.
    return df_imputed
# --- End New Imputation Functions ---

# --- Streamlit UI Functions ---
def initialize_missing_value_state():
    """Initialize session state variables for the missing value page."""
    defaults = {
        'mv_data': None,            # Original uploaded data
        'mv_data_imputed': None,    # Data after imputation
        'mv_missing_info': None,    # DataFrame with missing value stats
        'mv_imputation_method': 'mean', # Default imputation method
        'mv_columns_to_impute': [], # Columns selected for imputation
        'mv_knn_neighbors': 5,      # Parameter for KNN
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def show_missing_value_page():
    """Main function to display the missing value handling page."""
    initialize_missing_value_state()

    st.title("🧩 数据缺失值处理")
    st.markdown("---")
    st.info("上传数据，分析缺失值，并选择合适的方法进行填充。")

    # --- Create Tabs ---
    tab1, tab2, tab3 = st.tabs(["📁 数据导入与分析", "🛠️ 缺失值填充", "📊 结果展示"])

    with tab1:
        create_mv_data_import_analysis_section()

    with tab2:
        create_mv_imputation_section()

    with tab3:
        create_mv_results_section()

def create_mv_data_import_analysis_section():
    """UI section for data import and missing value analysis."""
    st.header("1. 数据导入与缺失值分析")

    uploaded_file = st.file_uploader("上传包含数据的文件 (CSV/Excel)", type=["csv", "xlsx", "xls"], key="mv_uploader")

    if uploaded_file:
        if st.button("加载并分析数据", key="mv_load_analyze_btn"):
            with st.spinner("正在加载和分析数据..."):
                try:
                    # Load data
                    data = pd.read_csv(uploaded_file) if uploaded_file.name.lower().endswith('.csv') else pd.read_excel(uploaded_file)

                    # Store original data
                    st.session_state.mv_data = data
                    st.session_state.mv_data_imputed = None # Clear previous imputed data

                    # Analyze missing values
                    missing_counts = data.isnull().sum()
                    missing_percentages = (missing_counts / len(data)) * 100
                    missing_info = pd.DataFrame({
                        '缺失数量': missing_counts,
                        '缺失比例 (%)': missing_percentages
                    })
                    # Filter to show only columns with missing values
                    missing_info = missing_info[missing_info['缺失数量'] > 0].sort_values(by='缺失比例 (%)', ascending=False)
                    st.session_state.mv_missing_info = missing_info

                    st.success(f"成功加载文件: {uploaded_file.name} ({len(data)} 行, {len(data.columns)} 列)")

                except Exception as e:
                    st.error(f"加载或分析数据时出错: {e}")
                    st.session_state.mv_data = None
                    st.session_state.mv_missing_info = None

    # Display analysis results if data is loaded
    if st.session_state.mv_data is not None:
        st.subheader("数据预览 (前5行)")
        st.dataframe(st.session_state.mv_data.head())

        st.subheader("缺失值统计")
        if st.session_state.mv_missing_info is not None and not st.session_state.mv_missing_info.empty:
            st.dataframe(st.session_state.mv_missing_info)

            st.subheader("缺失值热力图")
            try:
                fig = plot_missing_values_heatmap(st.session_state.mv_data)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"绘制热力图时出错: {e}")
        elif st.session_state.mv_missing_info is not None:
            st.success("🎉 数据中没有发现缺失值！")
        else:
            st.info("请点击“加载并分析数据”按钮以查看缺失值信息。")

def create_mv_imputation_section():
    """UI section for selecting imputation method and columns."""
    st.header("2. 缺失值填充方法选择")

    if st.session_state.mv_data is None:
        st.info("请先在“数据导入与分析”选项卡中加载数据。")
        return

    if st.session_state.mv_missing_info is None or st.session_state.mv_missing_info.empty:
        st.info("当前数据没有检测到缺失值，无需填充。")
        return

    # --- Select Columns for Imputation ---
    st.subheader("选择要填充的列")
    # Default to columns with missing values
    default_cols = st.session_state.mv_missing_info.index.tolist()
    # Ensure columns still exist in the dataframe
    valid_default_cols = [col for col in default_cols if col in st.session_state.mv_data.columns]

    # Get all columns suitable for imputation (numeric or potentially categorical)
    # For simplicity, let's focus on numeric for now, but allow selecting others
    all_columns = st.session_state.mv_data.columns.tolist()

    st.session_state.mv_columns_to_impute = st.multiselect(
        "选择需要填充缺失值的列",
        options=all_columns,
        default=valid_default_cols,
        key="mv_col_select"
    )

    if not st.session_state.mv_columns_to_impute:
        st.warning("请至少选择一列进行填充。")
        return

    # --- Select Imputation Method ---
    st.subheader("选择填充方法")
    # --- Updated: Added 'zero', 'ffill', 'bfill' ---
    imputation_options = {
        'zero': "填充为 0",
        'mean': "均值填充 (适用于数值型)",
        'median': "中位数填充 (适用于数值型，对异常值不敏感)",
        'most_frequent': "众数填充 (适用于数值型或类别型)",
        'ffill': "前向填充 (用前一个非缺失值填充)",
        'bfill': "后向填充 (用后一个非缺失值填充)",
        'knn': "K-近邻填充 (基于相似样本填充，计算量较大)",
        # 'mice': "多重插补 (MICE - 高级方法，暂未实现)"
    }
    # --- End Update ---

    st.session_state.mv_imputation_method = st.radio(
        "选择填充策略:",
        options=list(imputation_options.keys()),
        format_func=lambda x: imputation_options[x],
        key="mv_method_radio"
    )

    # --- Display Method Descriptions and Parameters ---
    method = st.session_state.mv_imputation_method
    with st.expander(f"关于 **{imputation_options[method]}** 的说明"):
        # --- Updated: Added descriptions for new methods ---
        if method == 'zero':
            st.markdown("""
            - **简介**: 将所有选定列中的缺失值（NaN）替换为0。
            - **适用范围**: 适用于缺失值本身就代表0或者可以合理地视为0的情况（例如，某些计数或金额）。也适用于某些机器学习模型（如基于树的模型）可以处理0值的情况。
            - **优点**: 实现非常简单快速。
            - **缺点**: 可能会显著改变列的分布（均值、方差等），引入偏差；如果0不是一个有意义的替换值，可能会误导模型。
            """)
        elif method == 'mean':
            st.markdown("""
            - **简介**: 使用该列所有非缺失值的平均数来填充缺失值。
            - **适用范围**: 数值型数据，且数据分布近似正态分布，对异常值较敏感。
            - **优点**: 计算简单快速。
            - **缺点**: 会改变数据的方差；对异常值敏感。
            """)
        elif method == 'median':
            st.markdown("""
            - **简介**: 使用该列所有非缺失值的中位数来填充缺失值。
            - **适用范围**: 数值型数据，特别是当数据存在偏态分布或包含异常值时。
            - **优点**: 对异常值不敏感；计算相对简单。
            - **缺点**: 可能不如均值填充保留数据的某些统计特性。
            """)
        elif method == 'most_frequent':
            st.markdown("""
            - **简介**: 使用该列中出现频率最高的值（众数）来填充缺失值。
            - **适用范围**: 类别型数据，也可用于数值型数据（特别是离散数值）。
            - **优点**: 适用于类别数据；实现简单。
            - **缺点**: 可能引入偏差，特别是当众数占比很高时；不适用于连续数值数据。
            """)
        elif method == 'ffill':
            st.markdown("""
            - **简介**: 前向填充 (Forward Fill)。使用该列中缺失值之前的最后一个有效观测值来填充。
            - **适用范围**: 时间序列数据或有序数据，假设缺失值与紧邻的前一个值相似。
            - **优点**: 实现简单；在时间序列中能保持一定的连续性。
            - **缺点**: 如果缺失值之前的有效值距离很远或不相关，填充效果可能不好；如果列的开头就有缺失值，则无法填充。
            """)
        elif method == 'bfill':
            st.markdown("""
            - **简介**: 后向填充 (Backward Fill)。使用该列中缺失值之后的第一个有效观测值来填充。
            - **适用范围**: 时间序列数据或有序数据，假设缺失值与紧邻的后一个值相似。
            - **优点**: 实现简单；可以填充列开头的缺失值（如果后面有值）。
            - **缺点**: 如果缺失值之后的有效值距离很远或不相关，填充效果可能不好；如果列的末尾有缺失值，则无法填充。
            """)
        # --- End Update ---
        elif method == 'knn':
            st.markdown("""
            - **简介**: 使用K个最相似（最近邻）样本的特征值的加权平均（或众数）来估计缺失值。
            - **适用范围**: 数值型数据，能捕捉特征间的复杂关系。
            - **优点**: 通常比简单填充方法更准确；能处理非线性关系。
            - **缺点**: 计算成本较高，特别是对于大数据集；对K值的选择敏感；需要确定合适的距离度量。
            """)
            # KNN specific parameter
            st.session_state.mv_knn_neighbors = st.slider(
                "K值 (邻居数量)", min_value=1, max_value=20,
                value=st.session_state.mv_knn_neighbors, step=1, key="mv_knn_k"
            )
        # elif method == 'mice':
        #     st.markdown("""
        #     - **简介**: 多重插补法 (Multivariate Imputation by Chained Equations)。通过迭代回归模型来估计缺失值，考虑了变量间的不确定性。
        #     - **适用范围**: 数值型和类别型数据，能处理复杂的缺失模式。
        #     - **优点**: 最准确的方法之一；能较好地保留数据的统计特性和不确定性。
        #     - **缺点**: 计算非常密集；实现和理解相对复杂。
        #     - **注意**: 此方法当前未在此应用中实现。
        #     """)

    # --- Apply Imputation Button ---
    if st.button("应用填充方法", key="mv_apply_btn", type="primary"):
        run_imputation()

def run_imputation():
    """Runs the selected imputation method."""
    df = st.session_state.mv_data
    columns_to_impute = st.session_state.mv_columns_to_impute
    method = st.session_state.mv_imputation_method

    if df is None or not columns_to_impute:
        st.error("无法执行填充：数据未加载或未选择要填充的列。")
        return

    # Check if selected columns exist
    missing_in_df = [col for col in columns_to_impute if col not in df.columns]
    if missing_in_df:
        st.error(f"选择的列在数据中不存在: {', '.join(missing_in_df)}")
        return

    # Filter to only impute columns that actually have missing values
    actual_cols_with_missing = df[columns_to_impute].isnull().sum()
    actual_cols_to_impute = actual_cols_with_missing[actual_cols_with_missing > 0].index.tolist()

    if not actual_cols_to_impute:
        st.warning("选择的列中没有缺失值，无需填充。")
        st.session_state.mv_data_imputed = df.copy() # Store original as imputed
        return

    st.info(f"将对以下列应用 '{imputation_options.get(method, method)}' 填充: {', '.join(actual_cols_to_impute)}")

    with st.spinner(f"正在使用 {imputation_options.get(method, method)} 方法填充缺失值..."):
        try:
            df_imputed = None
            # Select only numeric columns for mean, median, knn for now
            numeric_cols_selected = df[actual_cols_to_impute].select_dtypes(include=np.number).columns.tolist()

            # --- Updated: Handle new methods ---
            if method == 'zero':
                # Zero fill can apply to any selected column type
                df_imputed = impute_with_zero(df, actual_cols_to_impute)
            elif method == 'ffill':
                # ffill can apply to any selected column type
                df_imputed = impute_with_ffill(df, actual_cols_to_impute)
            elif method == 'bfill':
                # bfill can apply to any selected column type
                df_imputed = impute_with_bfill(df, actual_cols_to_impute)
            # --- End Update ---
            elif method in ['mean', 'median', 'most_frequent']:
                strategy = method
                cols_for_simple_impute = actual_cols_to_impute # Allow most_frequent for non-numeric
                if method in ['mean', 'median']:
                     cols_for_simple_impute = numeric_cols_selected
                     if not cols_for_simple_impute:
                          st.error(f"选择的方法 '{method}' 仅适用于数值列，但所选列中没有数值列。")
                          return
                df_imputed = impute_with_strategy(df, cols_for_simple_impute, strategy)

            elif method == 'knn':
                if not numeric_cols_selected:
                    st.error("KNN填充仅适用于数值列，但所选列中没有数值列。")
                    return
                n_neighbors = st.session_state.mv_knn_neighbors
                df_imputed = impute_with_knn(df, numeric_cols_selected, n_neighbors)

            # elif method == 'mice':
            #     st.error("MICE方法暂未实现。")
            #     return

            else:
                st.error("选择了无效的填充方法。")
                return

            # Store the imputed data
            st.session_state.mv_data_imputed = df_imputed
            st.success("缺失值填充完成！请前往“结果展示”选项卡查看。")

        except Exception as e:
            st.error(f"填充过程中发生错误: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.session_state.mv_data_imputed = None

def create_mv_results_section():
    """UI section to display the data after imputation."""
    st.header("3. 填充结果展示")

    if st.session_state.mv_data_imputed is None:
        st.info("请先在“缺失值填充”选项卡中应用填充方法。")
        return

    st.subheader("填充后的数据预览 (前5行)")
    st.dataframe(st.session_state.mv_data_imputed.head())

    # Verify if missing values are filled in the selected columns
    st.subheader("填充后缺失值检查")
    # Check only the columns that were *intended* for imputation
    cols_checked = st.session_state.mv_columns_to_impute
    if not cols_checked: # If somehow selection is empty, check all columns
         cols_checked = st.session_state.mv_data_imputed.columns.tolist()

    missing_after = st.session_state.mv_data_imputed[cols_checked].isnull().sum()
    missing_after_df = pd.DataFrame({'填充后缺失数量': missing_after})
    missing_after_df = missing_after_df[missing_after_df['填充后缺失数量'] > 0]

    if missing_after_df.empty:
        st.success("原先选择填充的列中的缺失值已成功处理！")
        # Optionally show heatmap of the whole dataset after imputation
        st.subheader("填充后数据整体缺失值热力图")
        try:
            fig_after = plot_missing_values_heatmap(st.session_state.mv_data_imputed)
            st.pyplot(fig_after)
        except Exception as e:
            st.error(f"绘制填充后热力图时出错: {e}")

    else:
        st.warning("填充后，以下原先选择的列仍存在缺失值 (可能是因为选择了不适用的方法、列类型，或者ffill/bfill无法填充开头/结尾的NaN):")
        st.dataframe(missing_after_df)
        # Show heatmap anyway to see the overall picture
        st.subheader("填充后数据整体缺失值热力图")
        try:
            fig_after = plot_missing_values_heatmap(st.session_state.mv_data_imputed)
            st.pyplot(fig_after)
        except Exception as e:
            st.error(f"绘制填充后热力图时出错: {e}")


    # Download button for imputed data
    st.subheader("下载填充后的数据")
    if st.button("准备下载文件", key="mv_prep_download"):
        try:
            df_to_download = st.session_state.mv_data_imputed
            # Use a more descriptive filename including the method
            method_name = st.session_state.mv_imputation_method.replace('_', '-')
            filename = f"imputed_data_{method_name}.csv"
            download_link = get_download_link_csv(df_to_download, filename, f"点击下载 {filename}")
            st.markdown(download_link, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"准备下载文件时出错: {e}")

# --- Main function entry point (for potential direct script run) ---
if __name__ == "__main__":
    # This part is optional, allows running this module directly for testing
    st.set_page_config(layout="wide", page_title="缺失值处理")
    show_missing_value_page()
