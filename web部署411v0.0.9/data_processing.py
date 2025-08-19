# -*- coding: utf-8 -*-
"""
Data Processing Module for Streamlit App
Includes Data Visualization and Segmentation functionalities.
Allows direct data upload within the module.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
import io # For download link

# --- 尝试导入字体工具 ---
try:
    from font_utils import apply_plot_style, FONT_PROP, create_figure_with_safe_dimensions
    print("字体工具从 font_utils 成功加载 (in data_processing)")
except ImportError:
    print("警告: 无法从 font_utils 导入，将在 data_processing 中使用备用绘图设置。")
    FONT_PROP = None
    def apply_plot_style(ax):
        ax.grid(True, linestyle='--', alpha=0.6)
        return ax
    def create_figure_with_safe_dimensions(w, h, dpi=80):
        fig, ax = plt.subplots(figsize=(w, h), dpi=dpi)
        return fig, ax
# --- 结束字体导入 ---

# --- 状态初始化 ---
def initialize_processing_state():
    """Initialize session state variables specific to data processing."""
    defaults = {
        'dp_uploaded_data': None,   # Data uploaded specifically in this module
        'dp_active_data': None,     # The dataframe currently being processed (uploaded or from main app)
        'dp_active_data_source': None, # 'uploaded' or 'main_app' or None
        'dp_selected_vis_col': None,
        'dp_plot_type': 'Histogram',
        'dp_segment_col': None,
        'dp_segment_value': None,
        'dp_segment_mode': 'value',
        'dp_segment_min': None,
        'dp_segment_max': None,
        'dp_segment_results': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- 绘图函数 (保持不变) ---
def plot_selected_visualization(df, column, plot_type):
    """Generates the selected plot for the specified column."""
    if df is None or column not in df.columns:
        st.warning("请选择有效的数据列进行可视化。")
        return None

    fig, ax = create_figure_with_safe_dimensions(8, 5)
    font_kwargs = {'fontproperties': FONT_PROP} if FONT_PROP else {}

    try:
        data_to_plot = df[column].dropna() # Drop NaNs for plotting

        if data_to_plot.empty:
            st.warning(f"列 '{column}' 没有有效数据可供绘制。")
            plt.close(fig)
            return None

        if plot_type == 'Histogram':
            if pd.api.types.is_numeric_dtype(data_to_plot):
                sns.histplot(data_to_plot, kde=True, ax=ax)
                ax.set_title(f"'{column}' 的直方图分布", **font_kwargs)
            else:
                st.warning("直方图仅适用于数值型数据。请尝试条形图。")
                plt.close(fig)
                return None
        elif plot_type == 'Box Plot':
            if pd.api.types.is_numeric_dtype(data_to_plot):
                sns.boxplot(y=data_to_plot, ax=ax)
                ax.set_title(f"'{column}' 的箱线图", **font_kwargs)
                ax.set_ylabel(column, **font_kwargs) # Set y-axis label
                ax.tick_params(axis='x', bottom=False, labelbottom=False) # Hide x-axis ticks/labels
            else:
                st.warning("箱线图仅适用于数值型数据。")
                plt.close(fig)
                return None
        elif plot_type == 'Bar Chart':
            # Suitable for categorical or low-cardinality numeric
            if data_to_plot.nunique() <= 50: # Limit cardinality for bar charts
                counts = data_to_plot.value_counts().nlargest(30) # Show top 30 categories
                sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, palette="viridis")
                ax.set_title(f"'{column}' 的频率条形图 (Top {len(counts)})", **font_kwargs)
                ax.tick_params(axis='x', rotation=45, labelsize=8)
            else:
                st.warning("条形图适用于类别型或低基数数值型数据 (最多50个唯一值)。")
                plt.close(fig)
                return None
        else:
            st.error(f"未知的绘图类型: {plot_type}")
            plt.close(fig)
            return None

        apply_plot_style(ax)
        plt.tight_layout()
        return fig

    except Exception as e:
        st.error(f"绘制图表时出错: {e}")
        print(traceback.format_exc())
        plt.close(fig) # Ensure figure is closed on error
        return None

# --- 数据分割函数 (保持不变) ---
def segment_data(df, column, mode, value=None, min_val=None, max_val=None):
    """Segments the dataframe based on the selected column and criteria."""
    if df is None or column not in df.columns:
        st.error("无效的数据或列用于分割。")
        return None

    try:
        if mode == 'value':
            if value is None or value == "": # Check for empty string too
                st.error("请为'按值分割'模式输入一个值。")
                return None
            # Attempt to convert value to column type if possible
            try:
                col_type = df[column].dtype
                if pd.api.types.is_numeric_dtype(col_type):
                    # Handle potential commas or other non-numeric chars if input is string
                    if isinstance(value, str): value = value.replace(',', '')
                    value_typed = pd.to_numeric(value)
                elif pd.api.types.is_datetime64_any_dtype(col_type):
                     value_typed = pd.to_datetime(value)
                elif pd.api.types.is_bool_dtype(col_type):
                     # Handle boolean conversion carefully
                     if str(value).lower() in ['true', '1', 'yes', 't']: value_typed = True
                     elif str(value).lower() in ['false', '0', 'no', 'f']: value_typed = False
                     else: raise ValueError("无法转换为布尔值")
                else: # Treat as string otherwise
                     value_typed = str(value)
            except (ValueError, TypeError) as conv_err:
                st.error(f"无法将输入值 '{value}' 转换为列 '{column}' 的类型 ({col_type}): {conv_err}")
                return None

            segment1 = df[df[column] == value_typed].copy()
            segment2 = df[df[column] != value_typed].copy()
            desc1 = f"等于 {value}"
            desc2 = f"不等于 {value}"

        elif mode == 'range':
            if min_val is None or max_val is None or min_val == "" or max_val == "": # Check empty
                st.error("请为'按范围分割'模式输入最小值和最大值。")
                return None
            # Attempt to convert range values
            try:
                col_type = df[column].dtype
                if not pd.api.types.is_numeric_dtype(col_type) and not pd.api.types.is_datetime64_any_dtype(col_type):
                    st.error("范围分割仅适用于数值或日期时间类型列。")
                    return None
                if pd.api.types.is_numeric_dtype(col_type):
                    # Handle potential commas
                    if isinstance(min_val, str): min_val = min_val.replace(',', '')
                    if isinstance(max_val, str): max_val = max_val.replace(',', '')
                    min_typed = pd.to_numeric(min_val)
                    max_typed = pd.to_numeric(max_val)
                else: # Datetime
                    min_typed = pd.to_datetime(min_val)
                    max_typed = pd.to_datetime(max_val)

                if min_typed >= max_typed:
                     st.error("范围分割的最小值必须小于最大值。")
                     return None

            except (ValueError, TypeError) as conv_err:
                st.error(f"无法将输入范围值 '{min_val}', '{max_val}' 转换为列 '{column}' 的类型 ({col_type}): {conv_err}")
                return None

            segment1 = df[(df[column] >= min_typed) & (df[column] < max_typed)].copy()
            segment2 = df[(df[column] < min_typed) | (df[column] >= max_typed)].copy()
            desc1 = f"在 [{min_val}, {max_val}) 范围内"
            desc2 = f"不在 [{min_val}, {max_val}) 范围内"

        else:
            st.error(f"未知的分割模式: {mode}")
            return None

        return {
            'segment1': {'df': segment1, 'desc': desc1},
            'segment2': {'df': segment2, 'desc': desc2}
        }

    except Exception as e:
        st.error(f"数据分割时出错: {e}")
        print(traceback.format_exc())
        return None

# --- 主 UI 函数 ---
def show_data_processing_page():
    """Displays the Data Processing page UI."""
    initialize_processing_state()
    st.title("🔧 数据处理与可视化")
    st.markdown("---")

    # --- 数据源选择与上传 ---
    st.header("1. 数据源")

    # Option to use data from other modules OR upload new data
    data_source_option = st.radio(
        "选择数据来源:",
        ["使用主应用已加载数据", "在此处上传新数据"],
        key="dp_data_source_radio",
        horizontal=True
    )

    df_to_process = None
    source_description = ""
    all_columns = []

    if data_source_option == "在此处上传新数据":
        uploaded_file_dp = st.file_uploader(
            "上传用于处理的 CSV 或 Excel 文件",
            type=["csv", "xlsx", "xls"],
            key="dp_file_uploader"
        )
        if uploaded_file_dp:
            # Check if it's a new upload or the same file
            # This simple check might not be perfect but helps avoid reloading unnecessarily
            if 'dp_last_uploaded_name' not in st.session_state or st.session_state.dp_last_uploaded_name != uploaded_file_dp.name:
                with st.spinner("加载上传的数据..."):
                    try:
                        data = pd.read_csv(uploaded_file_dp) if uploaded_file_dp.name.lower().endswith('.csv') else pd.read_excel(uploaded_file_dp)
                        data.dropna(axis=1, how='all', inplace=True)
                        if data.empty:
                            st.error("上传的文件为空。")
                            st.session_state.dp_uploaded_data = None
                        else:
                            st.session_state.dp_uploaded_data = data
                            st.session_state.dp_last_uploaded_name = uploaded_file_dp.name
                            # Reset selections when new data is uploaded
                            st.session_state.dp_selected_vis_col = None
                            st.session_state.dp_segment_col = None
                            st.session_state.dp_segment_results = None
                            st.success(f"已加载上传文件: {uploaded_file_dp.name}")
                    except Exception as e:
                        st.error(f"加载上传文件时出错: {e}")
                        st.session_state.dp_uploaded_data = None

            # Use the uploaded data if available
            if st.session_state.dp_uploaded_data is not None:
                df_to_process = st.session_state.dp_uploaded_data
                st.session_state.dp_active_data = df_to_process
                st.session_state.dp_active_data_source = 'uploaded'
                source_description = f"来源: 上传的文件 ({uploaded_file_dp.name})"
                all_columns = df_to_process.columns.tolist()

    else: # Use data from main app
        # Clear uploaded data state if switching back
        st.session_state.dp_uploaded_data = None
        if 'dp_last_uploaded_name' in st.session_state: del st.session_state.dp_last_uploaded_name

        # Find data loaded in other modules (adjust keys as needed)
        data_key_found = None
        potential_keys = ['classification_data', 'regression_data', 'clustering_data', 'mv_data', 'db_data', 'outlier_data', 'de_data'] # Add more keys if needed
        for key in potential_keys:
            if key in st.session_state and st.session_state[key] is not None:
                loaded_data = st.session_state[key]
                # Handle both DataFrame and dict cases
                if isinstance(loaded_data, pd.DataFrame):
                    df_to_process = loaded_data
                    data_key_found = key
                    break
                elif isinstance(loaded_data, dict) and 'X' in loaded_data and isinstance(loaded_data['X'], pd.DataFrame):
                    df_temp = loaded_data['X'].copy()
                    if 'y' in loaded_data and isinstance(loaded_data['y'], pd.Series):
                         df_temp[loaded_data['y'].name] = loaded_data['y']
                    if 'groups' in loaded_data and isinstance(loaded_data['groups'], pd.Series):
                         df_temp[loaded_data['groups'].name] = loaded_data['groups']
                    df_to_process = df_temp
                    data_key_found = key
                    break

        if df_to_process is not None:
            st.session_state.dp_active_data = df_to_process
            st.session_state.dp_active_data_source = 'main_app'
            source_description = f"来源: 主应用数据 (来自 '{data_key_found}')"
            all_columns = df_to_process.columns.tolist()
        else:
            st.session_state.dp_active_data = None
            st.session_state.dp_active_data_source = None
            st.warning("主应用中没有找到已加载的数据。请先在其他模块加载数据，或在此处上传新数据。")
            return # Stop further processing if no data is active

    # Display active data source and preview
    st.markdown(f"**当前处理数据:** {source_description}")
    if df_to_process is not None and not df_to_process.empty:
        with st.expander("预览当前数据 (前5行)"):
            st.dataframe(df_to_process.head())
    elif df_to_process is not None and df_to_process.empty:
         st.warning("当前活动数据为空。")
         return
    else:
         # This case should be handled by the return above, but as a fallback:
         st.error("无法确定要处理的数据，请导入正确的数据。")
         return

    st.markdown("---")

    # --- Create Tabs for Processing ---
    st.header("2. 数据处理操作")
    tab_vis, tab_seg = st.tabs(["📊 数据可视化", "✂️ 数据分割"])

    # --- Visualization Tab ---
    with tab_vis:
        st.subheader("数据可视化")
        st.markdown("选择一个列和图表类型来探索数据分布。")

        col1, col2 = st.columns(2)
        with col1:
            # Select column for visualization
            vis_col_options = [None] + all_columns
            # Reset selection if previous selection is no longer valid
            if st.session_state.dp_selected_vis_col not in all_columns:
                 st.session_state.dp_selected_vis_col = None
            vis_col_index = 0
            if st.session_state.dp_selected_vis_col:
                try: vis_col_index = vis_col_options.index(st.session_state.dp_selected_vis_col)
                except ValueError: pass
            st.session_state.dp_selected_vis_col = st.selectbox(
                "选择要可视化的列:",
                options=vis_col_options,
                index=vis_col_index,
                key="dp_vis_col_select"
            )

        with col2:
            # Select plot type
            plot_options = ['Histogram', 'Box Plot', 'Bar Chart']
            plot_type_index = plot_options.index(st.session_state.dp_plot_type) if st.session_state.dp_plot_type in plot_options else 0
            st.session_state.dp_plot_type = st.selectbox(
                "选择图表类型:",
                options=plot_options,
                index=plot_type_index,
                key="dp_plot_type_select"
            )

        # Generate and display plot
        if st.session_state.dp_selected_vis_col:
            fig = plot_selected_visualization(df_to_process, st.session_state.dp_selected_vis_col, st.session_state.dp_plot_type)
            if fig:
                st.pyplot(fig)
                plt.close(fig) # Close figure after displaying

    # --- Segmentation Tab ---
    with tab_seg:
        st.subheader("数据分割")
        st.markdown("根据选定列的值或范围将数据分割成两部分。")

        col1, col2 = st.columns([2, 1])
        with col1:
            # Select column for segmentation
            seg_col_options = [None] + all_columns
            # Reset selection if previous selection is no longer valid
            if st.session_state.dp_segment_col not in all_columns:
                 st.session_state.dp_segment_col = None
            seg_col_index = 0
            if st.session_state.dp_segment_col:
                 try: seg_col_index = seg_col_options.index(st.session_state.dp_segment_col)
                 except ValueError: pass
            st.session_state.dp_segment_col = st.selectbox(
                "选择用于分割的列:",
                options=seg_col_options,
                index=seg_col_index,
                key="dp_seg_col_select"
            )

        with col2:
            # Select segmentation mode
            seg_mode_options = {'value': '按特定值分割', 'range': '按数值/日期范围分割'}
            st.session_state.dp_segment_mode = st.radio(
                "选择分割模式:",
                options=list(seg_mode_options.keys()),
                format_func=lambda x: seg_mode_options[x],
                key="dp_seg_mode_radio",
                horizontal=True
            )

        # Input for segmentation criteria
        if st.session_state.dp_segment_col:
            # Display column type info
            try:
                 col_dtype = df_to_process[st.session_state.dp_segment_col].dtype
                 st.info(f"所选列 '{st.session_state.dp_segment_col}' 的数据类型: {col_dtype}")
            except KeyError:
                 st.error("所选列不存在。")


            if st.session_state.dp_segment_mode == 'value':
                st.session_state.dp_segment_value = st.text_input(
                    f"输入 '{st.session_state.dp_segment_col}' 列中用于分割的值:",
                    value=st.session_state.dp_segment_value if st.session_state.dp_segment_value is not None else "",
                    key="dp_seg_value_input",
                    help="输入要匹配的确切值（区分大小写）。对于布尔值，可输入 True/False 或 1/0。"
                )
            elif st.session_state.dp_segment_mode == 'range':
                col_range1, col_range2 = st.columns(2)
                with col_range1:
                    st.session_state.dp_segment_min = st.text_input(
                        "输入范围最小值 (包含):",
                        value=st.session_state.dp_segment_min if st.session_state.dp_segment_min is not None else "",
                        key="dp_seg_min_input",
                        help="对于日期时间，格式如 'YYYY-MM-DD' 或 'YYYY-MM-DD HH:MM:SS'"
                    )
                with col_range2:
                    st.session_state.dp_segment_max = st.text_input(
                        "输入范围最大值 (不包含):",
                        value=st.session_state.dp_segment_max if st.session_state.dp_segment_max is not None else "",
                        key="dp_seg_max_input",
                        help="对于日期时间，格式如 'YYYY-MM-DD' 或 'YYYY-MM-DD HH:MM:SS'"
                    )

            # Button to perform segmentation
            if st.button("执行分割", key="dp_run_segment_btn"):
                st.session_state.dp_segment_results = segment_data(
                    df_to_process, # Use the currently active dataframe
                    st.session_state.dp_segment_col,
                    st.session_state.dp_segment_mode,
                    st.session_state.dp_segment_value,
                    st.session_state.dp_segment_min,
                    st.session_state.dp_segment_max
                )
                # Rerun to display results immediately
                if st.session_state.dp_segment_results:
                     st.rerun()


        # Display segmentation results
        if st.session_state.dp_segment_results:
            st.markdown("---")
            st.subheader("分割结果预览")
            res = st.session_state.dp_segment_results
            seg1_info = res.get('segment1', {})
            seg2_info = res.get('segment2', {})
            df1 = seg1_info.get('df', pd.DataFrame())
            df2 = seg2_info.get('df', pd.DataFrame())

            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.markdown(f"**部分 1: {seg1_info.get('desc', '')} ({len(df1)} 行)**")
                if not df1.empty:
                    st.dataframe(df1.head())
                    # Download Button for Segment 1
                    csv1 = df1.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label="下载部分 1 (CSV)",
                        data=csv1,
                        file_name="segment_1.csv",
                        mime="text/csv",
                        key="dp_download_seg1"
                    )
                else:
                    st.info("此部分无数据。")
            with col_res2:
                st.markdown(f"**部分 2: {seg2_info.get('desc', '')} ({len(df2)} 行)**")
                if not df2.empty:
                    st.dataframe(df2.head())
                     # Download Button for Segment 2
                    csv2 = df2.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label="下载部分 2 (CSV)",
                        data=csv2,
                        file_name="segment_2.csv",
                        mime="text/csv",
                        key="dp_download_seg2"
                    )
                else:
                    st.info("此部分无数据。")

# --- Entry point for testing (optional) ---
# if __name__ == "__main__":
#     st.set_page_config(layout="wide", page_title="数据处理测试")
#     # Manually set some dummy data in session state for standalone testing
#     if 'classification_data' not in st.session_state:
#         st.session_state.classification_data = pd.DataFrame({
#             'NumericCol': np.random.rand(100) * 10,
#             'CategoryCol': np.random.choice(['A', 'B', 'C', 'D'], 100),
#             'DateCol': pd.to_datetime(pd.date_range(start='2023-01-01', periods=100, freq='D')),
#             'BoolCol': np.random.choice([True, False], 100)
#         })
#     show_data_processing_page()

