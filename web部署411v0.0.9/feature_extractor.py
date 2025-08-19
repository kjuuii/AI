# -*- coding: utf-8 -*-
"""
特征提取模块 - Streamlit界面
提供数据上传、特征提取、可视化和导出功能
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import io
import base64

# 尝试导入字体设置
try:
    from font_utils import FONT_PROP, apply_plot_style

    FONT_AVAILABLE = True
except ImportError:
    print("警告: 无法从 font_utils 导入，将使用备用绘图设置。")
    FONT_AVAILABLE = False
    FONT_PROP = None


    def apply_plot_style(ax):
        return ax

# 导入特征提取器
try:
    from feature_extractor import (
        process_data,
        get_feature_names,
        save_features_to_file,
        export_features_for_analysis,
        create_analysis_script,
        extract_features_from_segment,
        SAMPLE_RATE
    )

    EXTRACTOR_LOADED = True
except ImportError as e:
    st.error(f"无法导入 feature_extractor 模块: {e}")
    EXTRACTOR_LOADED = False


def show_feature_extraction_page():
    """显示特征提取页面的主函数"""
    st.title("🔍 特征提取")

    if not EXTRACTOR_LOADED:
        st.error("特征提取器模块未能正确加载。请确保 feature_extractor.py 文件存在。")
        return

    # 使用标签页组织不同功能
    tab1, tab2, tab3, tab4 = st.tabs(["📁 数据上传", "⚙️ 特征提取", "📊 特征分析", "💾 导出结果"])

    with tab1:
        show_data_upload_section()

    with tab2:
        show_feature_extraction_section()

    with tab3:
        show_feature_analysis_section()

    with tab4:
        show_export_section()


def show_data_upload_section():
    """数据上传部分"""
    st.header("📁 数据上传")

    # 数据源选择
    data_source = st.radio(
        "选择数据源：",
        ["上传文件", "使用示例数据", "从其他模块加载"],
        key="fe_data_source"
    )

    if data_source == "上传文件":
        uploaded_file = st.file_uploader(
            "上传数据文件",
            type=['csv', 'xlsx', 'xls'],
            help="支持CSV和Excel格式，需包含电压和电流数据列",
            key="fe_file_upload"
        )

        if uploaded_file is not None:
            # 读取文件
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                st.success(f"✅ 成功加载文件: {uploaded_file.name}")
                st.write(f"数据形状: {df.shape}")

                # 显示数据预览
                st.subheader("数据预览")
                st.dataframe(df.head())

                # 保存到session state
                st.session_state['fe_data'] = df
                st.session_state['fe_filename'] = uploaded_file.name

                # 列选择
                show_column_selection(df)

            except Exception as e:
                st.error(f"读取文件时出错: {e}")

    elif data_source == "使用示例数据":
        if st.button("生成示例数据", key="fe_generate_example"):
            df = generate_example_data()
            st.session_state['fe_data'] = df
            st.session_state['fe_filename'] = "example_data.csv"

            st.success("✅ 已生成示例数据")
            st.dataframe(df.head())

            # 自动设置列名
            st.session_state['fe_voltage_col'] = 'Voltage'
            st.session_state['fe_current_col'] = 'Current'

    else:  # 从其他模块加载
        st.info("可以从分类、回归或聚类模块加载已处理的数据")

        # 检查其他模块的数据
        available_data = []
        if 'classification_data' in st.session_state and st.session_state.classification_data is not None:
            available_data.append("分类模块数据")
        if 'regression_data' in st.session_state and st.session_state.regression_data is not None:
            available_data.append("回归模块数据")
        if 'clustering_data' in st.session_state and st.session_state.clustering_data is not None:
            available_data.append("聚类模块数据")

        if available_data:
            selected_data = st.selectbox("选择要加载的数据：", available_data)

            if st.button("加载数据", key="fe_load_from_module"):
                if selected_data == "分类模块数据":
                    df = st.session_state.classification_data
                elif selected_data == "回归模块数据":
                    df = st.session_state.regression_data
                else:
                    df = st.session_state.clustering_data

                st.session_state['fe_data'] = df
                st.session_state['fe_filename'] = f"{selected_data}.csv"

                st.success(f"✅ 已加载{selected_data}")
                st.dataframe(df.head())

                show_column_selection(df)
        else:
            st.warning("暂无可用的模块数据")


def show_column_selection(df):
    """显示列选择界面"""
    st.subheader("列选择")

    col1, col2 = st.columns(2)

    with col1:
        voltage_col = st.selectbox(
            "选择电压列：",
            df.columns,
            key="fe_voltage_col_select"
        )
        st.session_state['fe_voltage_col'] = voltage_col

    with col2:
        current_col = st.selectbox(
            "选择电流列：",
            df.columns,
            key="fe_current_col_select"
        )
        st.session_state['fe_current_col'] = current_col


def generate_example_data():
    """生成示例数据"""
    # 创建3秒的模拟数据
    time = np.linspace(0, 3, 3 * SAMPLE_RATE)

    # 模拟启动电流信号
    current_signal = (1 - np.exp(-time * 3)) * 1.5  # 指数上升
    current_signal += 0.1 * np.sin(2 * np.pi * 5 * time)  # 添加5Hz振荡
    current_signal += np.random.randn(len(time)) * 0.05  # 添加噪声

    # 模拟电压信号
    voltage_signal = np.full_like(time, 220)  # 基准电压220V
    voltage_signal += 2 * np.sin(2 * np.pi * 0.5 * time)  # 轻微波动
    voltage_signal += np.random.randn(len(time)) * 0.5  # 添加噪声

    return pd.DataFrame({
        'Time': time,
        'Voltage': voltage_signal,
        'Current': current_signal
    })


def show_feature_extraction_section():
    """特征提取部分"""
    st.header("⚙️ 特征提取")

    if 'fe_data' not in st.session_state:
        st.warning("请先上传数据")
        return

    df = st.session_state['fe_data']

    # 提取参数设置
    st.subheader("提取参数")

    col1, col2, col3 = st.columns(3)

    with col1:
        window_size = st.number_input(
            "窗口大小（样本数）",
            min_value=100,
            max_value=10000,
            value=1875,
            step=100,
            help="数据窗口的大小，默认为3秒的数据（625Hz * 3）"
        )

    with col2:
        current_threshold = st.number_input(
            "电流阈值",
            min_value=0.0,
            max_value=5.0,
            value=0.5,
            step=0.1,
            help="用于检测信号起始点的电流阈值"
        )

    with col3:
        batch_process = st.checkbox(
            "批量处理",
            help="如果数据包含多个样本，可以批量提取特征"
        )

    # 提取按钮
    if st.button("🚀 开始提取特征", key="fe_extract_button", type="primary"):
        try:
            with st.spinner("正在提取特征..."):
                # 获取列名
                voltage_col = st.session_state.get('fe_voltage_col', 'Voltage')
                current_col = st.session_state.get('fe_current_col', 'Current')

                # 提取特征
                features_df = process_data(
                    df,
                    voltage_col=voltage_col,
                    current_col=current_col,
                    window_size=int(window_size),
                    current_threshold=current_threshold
                )

                # 保存结果
                st.session_state['fe_features'] = features_df
                st.session_state['fe_feature_names'] = get_feature_names()

                st.success(f"✅ 特征提取完成！提取了 {len(features_df.columns)} 个特征")

                # 显示特征预览
                st.subheader("特征预览")
                st.dataframe(features_df)

                # 显示特征统计
                with st.expander("特征统计信息", expanded=True):
                    st.dataframe(features_df.describe())

        except Exception as e:
            st.error(f"特征提取失败: {e}")
            import traceback
            st.code(traceback.format_exc())


def show_feature_analysis_section():
    """特征分析部分"""
    st.header("📊 特征分析")

    if 'fe_features' not in st.session_state:
        st.warning("请先提取特征")
        return

    features_df = st.session_state['fe_features']
    feature_names = st.session_state['fe_feature_names']

    # 分析选项
    analysis_type = st.selectbox(
        "选择分析类型：",
        ["特征分布", "特征相关性", "特征重要性估计", "特征详情"]
    )

    if analysis_type == "特征分布":
        show_feature_distribution(features_df)

    elif analysis_type == "特征相关性":
        show_feature_correlation(features_df)

    elif analysis_type == "特征重要性估计":
        show_feature_importance(features_df)

    else:  # 特征详情
        show_feature_details(features_df, feature_names)


def show_feature_distribution(features_df):
    """显示特征分布"""
    st.subheader("特征分布图")

    # 选择要显示的特征
    selected_features = st.multiselect(
        "选择要显示的特征：",
        features_df.columns,
        default=list(features_df.columns[:6])  # 默认显示前6个
    )

    if selected_features:
        # 计算子图布局
        n_features = len(selected_features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_features == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for idx, feature in enumerate(selected_features):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]

            # 绘制直方图
            ax.hist(features_df[feature], bins=20, alpha=0.7, color='steelblue', edgecolor='black')
            ax.set_title(feature, fontsize=10)
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)

            if FONT_AVAILABLE:
                apply_plot_style(ax)

        # 隐藏多余的子图
        for idx in range(n_features, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


def show_feature_correlation(features_df):
    """显示特征相关性"""
    st.subheader("特征相关性矩阵")

    # 计算相关性矩阵
    corr_matrix = features_df.corr()

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(12, 10))

    # 使用mask只显示下三角
    mask = np.triu(np.ones_like(corr_matrix), k=1)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )

    ax.set_title('Feature Correlation Matrix', fontsize=16, pad=20)

    if FONT_AVAILABLE:
        apply_plot_style(ax)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # 显示高相关性特征对
    st.subheader("高相关性特征对")

    # 获取相关性阈值
    threshold = st.slider("相关性阈值：", 0.5, 1.0, 0.8, 0.05)

    # 找出高相关性的特征对
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) >= threshold:
                high_corr_pairs.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })

    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs)
        high_corr_df = high_corr_df.sort_values('Correlation', ascending=False)
        st.dataframe(high_corr_df)
    else:
        st.info(f"没有发现相关性高于 {threshold} 的特征对")


def show_feature_importance(features_df):
    """显示特征重要性估计"""
    st.subheader("特征重要性估计")

    st.info("这是基于特征方差的简单重要性估计。对于更准确的重要性评估，请在具体的机器学习任务中使用。")

    # 计算各种统计量作为重要性指标
    importance_metrics = pd.DataFrame({
        'Feature': features_df.columns,
        'Variance': features_df.var(),
        'Coefficient of Variation': features_df.std() / (features_df.mean() + 1e-10),
        'Range': features_df.max() - features_df.min(),
        'IQR': features_df.quantile(0.75) - features_df.quantile(0.25)
    })

    # 标准化各指标
    for col in ['Variance', 'Coefficient of Variation', 'Range', 'IQR']:
        importance_metrics[f'{col}_normalized'] = (
                importance_metrics[col] / importance_metrics[col].max()
        )

    # 计算综合重要性分数
    importance_metrics['Importance Score'] = (
            importance_metrics['Variance_normalized'] * 0.3 +
            importance_metrics['Coefficient of Variation_normalized'] * 0.3 +
            importance_metrics['Range_normalized'] * 0.2 +
            importance_metrics['IQR_normalized'] * 0.2
    )

    # 排序
    importance_metrics = importance_metrics.sort_values('Importance Score', ascending=False)

    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 条形图
    top_n = min(15, len(importance_metrics))
    ax1.barh(
        importance_metrics['Feature'].head(top_n),
        importance_metrics['Importance Score'].head(top_n),
        color='steelblue'
    )
    ax1.set_xlabel('Importance Score')
    ax1.set_title(f'Top {top_n} Important Features')
    ax1.grid(True, alpha=0.3)

    # 各指标对比
    metrics_to_plot = ['Variance_normalized', 'Coefficient of Variation_normalized',
                       'Range_normalized', 'IQR_normalized']
    importance_metrics[metrics_to_plot].head(top_n).plot(
        kind='bar',
        ax=ax2,
        width=0.8
    )
    ax2.set_xticklabels(importance_metrics['Feature'].head(top_n), rotation=45, ha='right')
    ax2.set_ylabel('Normalized Score')
    ax2.set_title('Feature Importance by Different Metrics')
    ax2.legend(['Variance', 'CV', 'Range', 'IQR'], loc='upper right')
    ax2.grid(True, alpha=0.3)

    if FONT_AVAILABLE:
        apply_plot_style(ax1)
        apply_plot_style(ax2)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # 显示详细数据
    with st.expander("查看详细重要性指标"):
        display_df = importance_metrics[['Feature', 'Variance', 'Coefficient of Variation',
                                         'Range', 'IQR', 'Importance Score']]
        st.dataframe(display_df)


def show_feature_details(features_df, feature_names):
    """显示特征详细说明"""
    st.subheader("特征详细说明")

    # 特征描述
    feature_descriptions = {
        'startup_peak': "启动峰值电流 - 设备启动时的最大电流值",
        'startup_peak_time': "启动峰值时间 - 达到启动峰值的时间（秒）",
        'startup_acceleration': "启动加速度 - 启动阶段电流变化的加速度",
        'startup_slope': "启动斜率 - 启动阶段电流上升的斜率",
        'steady_mean': "稳态平均值 - 稳定运行时的平均电流",
        'steady_std': "稳态标准差 - 稳定运行时电流的标准差",
        'steady_amplitude': "稳态幅度 - 稳定运行时电流的最大最小值之差",
        'steady_iqr': "稳态四分位距 - 稳定运行时电流的四分位距",
        'steady_skew': "稳态偏度 - 稳定运行时电流分布的偏度",
        'steady_kurtosis': "稳态峰度 - 稳定运行时电流分布的峰度",
        'peaks_per_sec': "每秒峰值数 - 稳定运行时每秒出现的峰值数量",
        'valleys_per_sec': "每秒谷值数 - 稳定运行时每秒出现的谷值数量",
        'mean_peak_prominence': "平均峰值突出度 - 峰值的平均突出程度",
        'std_peak_prominence': "峰值突出度标准差 - 峰值突出度的标准差",
        'main_freq': "主频率 - 信号的主要频率成分",
        'main_freq_power': "主频率功率 - 主频率的功率大小",
        'low_freq_ratio': "低频比例 - 0-10Hz频率成分的能量比例",
        'mid_freq_ratio': "中频比例 - 10-30Hz频率成分的能量比例",
        'high_freq_ratio': "高频比例 - 30-50Hz频率成分的能量比例",
        'spectral_centroid': "频谱质心 - 频谱的重心位置",
        'window_mean_std': "窗口均值标准差 - 滑动窗口均值的标准差",
        'window_mean_range': "窗口均值范围 - 滑动窗口均值的范围",
        'mean_window_std': "平均窗口标准差 - 滑动窗口标准差的平均值",
        'std_window_std': "窗口标准差的标准差 - 滑动窗口标准差的标准差",
        'autocorr_lag10': "10滞后自相关 - 信号与其10个采样点延迟的相关性",
        'first_min_time': "第一个最小值时间 - 自相关函数第一个最小值的时间",
        'voltage_mean': "电压平均值 - 电压信号的平均值",
        'voltage_std': "电压标准差 - 电压信号的标准差",
        'voltage_amplitude': "电压幅度 - 电压信号的最大最小值之差",
        'power_mean': "功率平均值 - 瞬时功率的平均值",
        'power_std': "功率标准差 - 瞬时功率的标准差",
        'power_max': "功率最大值 - 瞬时功率的最大值",
        'voltage_current_corr': "电压电流相关性 - 电压和电流信号的相关系数"
    }

    # 创建特征说明表格
    feature_info = []
    for i, name in enumerate(feature_names):
        feature_info.append({
            '序号': i + 1,
            '特征名称': name,
            '当前值': features_df[name].iloc[0] if len(features_df) > 0 else 'N/A',
            '说明': feature_descriptions.get(name, '暂无说明')
        })

    feature_info_df = pd.DataFrame(feature_info)

    # 显示表格
    st.dataframe(
        feature_info_df,
        use_container_width=True,
        hide_index=True
    )

    # 特征分类展示
    st.subheader("特征分类")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🚀 启动特征**")
        startup_features = ['startup_peak', 'startup_peak_time', 'startup_acceleration', 'startup_slope']
        for f in startup_features:
            if f in features_df.columns:
                st.write(f"- {f}: {features_df[f].iloc[0]:.4f}")

    with col2:
        st.markdown("**📊 稳态特征**")
        steady_features = ['steady_mean', 'steady_std', 'steady_amplitude', 'steady_iqr']
        for f in steady_features:
            if f in features_df.columns:
                st.write(f"- {f}: {features_df[f].iloc[0]:.4f}")

    with col3:
        st.markdown("**🌊 频域特征**")
        freq_features = ['main_freq', 'main_freq_power', 'spectral_centroid']
        for f in freq_features:
            if f in features_df.columns:
                st.write(f"- {f}: {features_df[f].iloc[0]:.4f}")


def show_export_section():
    """导出结果部分"""
    st.header("💾 导出结果")

    if 'fe_features' not in st.session_state:
        st.warning("请先提取特征")
        return

    features_df = st.session_state['fe_features']

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("导出特征数据")

        # 文件格式选择
        file_format = st.selectbox(
            "选择导出格式：",
            ["CSV", "Excel", "JSON"]
        )

        # 添加元数据
        include_metadata = st.checkbox("包含元数据", value=True)

        if st.button("导出特征数据", key="fe_export_features"):
            try:
                if file_format == "CSV":
                    csv = features_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="extracted_features.csv">下载CSV文件</a>'
                    st.markdown(href, unsafe_allow_html=True)

                elif file_format == "Excel":
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        features_df.to_excel(writer, sheet_name='Features', index=False)

                        if include_metadata:
                            metadata = {
                                'Extraction Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'Feature Count': len(features_df.columns),
                                'Sample Count': len(features_df),
                                'Source File': st.session_state.get('fe_filename', 'Unknown')
                            }
                            metadata_df = pd.DataFrame(list(metadata.items()), columns=['Property', 'Value'])
                            metadata_df.to_excel(writer, sheet_name='Metadata', index=False)

                    b64 = base64.b64encode(output.getvalue()).decode()
                    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="extracted_features.xlsx">下载Excel文件</a>'
                    st.markdown(href, unsafe_allow_html=True)

                else:  # JSON
                    json_str = features_df.to_json(orient='records', indent=2)
                    b64 = base64.b64encode(json_str.encode()).decode()
                    href = f'<a href="data:file/json;base64,{b64}" download="extracted_features.json">下载JSON文件</a>'
                    st.markdown(href, unsafe_allow_html=True)

                st.success("✅ 导出准备完成，点击链接下载")

            except Exception as e:
                st.error(f"导出失败: {e}")

    with col2:
        st.subheader("导出到其他模块")

        # 选择目标模块
        target_module = st.selectbox(
            "选择目标模块：",
            ["分类模块", "回归模块", "聚类模块"]
        )

        # 添加标签（如果需要）
        if target_module in ["分类模块", "回归模块"]:
            st.info("分类和回归任务需要添加目标标签")

            label_source = st.radio(
                "标签来源：",
                ["手动输入", "从文件加载"]
            )

            if label_source == "手动输入":
                if target_module == "分类模块":
                    label_value = st.text_input("输入类别标签：", "Class_A")
                else:
                    label_value = st.number_input("输入目标值：", value=1.0)

                # 为所有样本添加相同标签
                features_with_label = features_df.copy()
                features_with_label['target'] = label_value
            else:
                st.info("请确保标签文件与特征数据的样本数量一致")
        else:
            features_with_label = features_df.copy()

        if st.button("导出到模块", key="fe_export_to_module"):
            if target_module == "分类模块":
                st.session_state['classification_data'] = features_with_label
                st.success("✅ 特征数据已导出到分类模块")
            elif target_module == "回归模块":
                st.session_state['regression_data'] = features_with_label
                st.success("✅ 特征数据已导出到回归模块")
            else:
                st.session_state['clustering_data'] = features_with_label
                st.success("✅ 特征数据已导出到聚类模块")

            st.info(f"请前往{target_module}继续分析")

    # 生成分析脚本
    st.subheader("生成分析脚本")

    script_type = st.selectbox(
        "选择脚本类型：",
        ["分类分析", "回归分析", "聚类分析", "通用分析"]
    )

    if st.button("生成Python脚本", key="fe_generate_script"):
        try:
            # 先保存特征文件
            temp_file = "temp_features.csv"
            features_df.to_csv(temp_file, index=False)

            # 创建脚本
            script_types_map = {
                "分类分析": "classification",
                "回归分析": "regression",
                "聚类分析": "clustering",
                "通用分析": "general"
            }

            script_path = create_analysis_script(
                temp_file,
                script_types_map[script_type],
                output_dir="generated_scripts"
            )

            # 读取并显示脚本
            with open(script_path, 'r', encoding='utf-8') as f:
                script_content = f.read()

            st.code(script_content, language='python')

            # 提供下载
            b64 = base64.b64encode(script_content.encode()).decode()
            href = f'<a href="data:file/python;base64,{b64}" download="{os.path.basename(script_path)}">下载Python脚本</a>'
            st.markdown(href, unsafe_allow_html=True)

            st.success(f"✅ 脚本生成成功: {script_path}")

            # 清理临时文件
            if os.path.exists(temp_file):
                os.remove(temp_file)

        except Exception as e:
            st.error(f"生成脚本失败: {e}")
            import traceback
            st.code(traceback.format_exc())

# 主函数已经在最上面定义了 show_feature_extraction_page()