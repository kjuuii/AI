# -*- coding: utf-8 -*-
"""
Feature Extraction Module for Streamlit App
Enhanced version with batch processing, feature selection, and data merging
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
import io
import os
import glob
import pickle
import warnings
from datetime import datetime
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import find_peaks, welch, butter, filtfilt
from scipy.fft import fft, fftfreq

# 屏蔽警告
warnings.filterwarnings('ignore')

# --- 尝试导入字体工具 ---
try:
    from font_utils import apply_plot_style, FONT_PROP, create_figure_with_safe_dimensions

    print("字体工具从 font_utils 成功加载 (in feature_extraction)")
except ImportError:
    print("警告: 无法从 font_utils 导入，将在 feature_extraction 中使用备用绘图设置。")
    FONT_PROP = None


    def apply_plot_style(ax):
        ax.grid(True, linestyle='--', alpha=0.6)
        return ax


    def create_figure_with_safe_dimensions(w, h, dpi=80):
        fig, ax = plt.subplots(figsize=(w, h), dpi=dpi)
        return fig, ax

# --- 全局参数 ---
SAMPLE_RATE = 625
RANDOM_SEED = 42

# --- 特征定义 ---
FEATURE_DEFINITIONS = {
    "Basic Statistical Features": {
        "description": "基础统计特征提取",
        "features": {
            "mean": "均值 - 数据的平均值",
            "std": "标准差 - 数据的离散程度",
            "min": "最小值 - 数据的最小值",
            "max": "最大值 - 数据的最大值",
            "median": "中位数 - 数据的中间值",
            "q25": "下四分位数 - 25%分位点",
            "q75": "上四分位数 - 75%分位点",
            "skew": "偏度 - 分布的不对称性",
            "kurtosis": "峰度 - 分布的尖锐程度"
        }
    },
    "Advanced Statistical Features": {
        "description": "高级统计特征提取",
        "features": {
            "cv": "变异系数 - 标准差与均值的比率",
            "energy": "能量 - 信号的总能量",
            "entropy": "熵 - 信号的复杂度",
            "peak_to_peak": "峰峰值 - 最大值与最小值之差",
            "rms": "均方根 - 有效值",
            "abs_mean": "绝对值均值 - 绝对值的平均"
        }
    },
    "Time Domain Features": {
        "description": "时域特征提取",
        "features": {
            "num_peaks": "峰值数量 - 局部最大值个数",
            "num_valleys": "谷值数量 - 局部最小值个数",
            "zero_crossing_rate": "过零率 - 信号穿越零点的频率",
            "waveform_factor": "波形因子 - 有效值与平均值之比",
            "crest_factor": "峰值因子 - 峰值与有效值之比",
            "impulse_factor": "脉冲因子 - 峰值与平均值之比"
        }
    },
    "Frequency Domain Features": {
        "description": "频域特征提取",
        "features": {
            "main_freq": "主频率 - 最强频率成分",
            "main_freq_amp": "主频幅值 - 主频的振幅",
            "spectral_centroid": "频谱质心 - 频谱的重心",
            "spectral_energy": "频谱能量 - 频域总能量",
            "spectral_entropy": "频谱熵 - 频谱复杂度"
        }
    },
    "Rolling Window Features": {
        "description": "滑动窗口特征提取",
        "features": {
            "rolling_mean_avg": "滑动均值的平均 - 局部均值的整体平均",
            "rolling_mean_std": "滑动均值的标准差 - 局部均值的变化程度",
            "rolling_std_avg": "滑动标准差的平均 - 局部变化的平均程度",
            "rolling_std_std": "滑动标准差的标准差 - 局部变化的稳定性",
            "rolling_min_avg": "滑动最小值的平均 - 局部最小值的平均",
            "rolling_max_avg": "滑动最大值的平均 - 局部最大值的平均",
            "rolling_range_avg": "滑动范围的平均 - 局部波动范围的平均"
        }
    },
    "Enhanced Signal Features": {
        "description": "增强信号特征",
        "features": {
            "startup_max": "启动峰值 - 启动阶段的最大值",
            "startup_time": "启动时间 - 达到峰值的时间",
            "steady_mean": "稳态均值 - 稳定阶段的平均值",
            "steady_std": "稳态标准差 - 稳定阶段的波动",
            "autocorr_lag10": "自相关系数 - 10个延迟的相关性"
        }
    }
}

# --- 特征提取方法优缺点 ---
FEATURE_METHODS_INFO = {
    "Basic Statistical Features": {
        "pros": [
            "✅ 计算速度快，效率高",
            "✅ 通用性强，适用于所有数值数据",
            "✅ 结果稳定，可解释性强"
        ],
        "cons": [
            "❌ 可能忽略时序信息",
            "❌ 无法捕捉复杂的模式"
        ],
        "suitable_for": "适用于快速数据探索、基础分析"
    },
    "Advanced Statistical Features": {
        "pros": [
            "✅ 包含更多统计信息",
            "✅ 能捕捉数据分布特性"
        ],
        "cons": [
            "❌ 计算复杂度增加",
            "❌ 某些特征可能存在相关性"
        ],
        "suitable_for": "适用于分布分析、异常检测"
    },
    "Time Domain Features": {
        "pros": [
            "✅ 保留时序信息",
            "✅ 能识别趋势和模式"
        ],
        "cons": [
            "❌ 需要时序数据",
            "❌ 对采样率敏感"
        ],
        "suitable_for": "适用于时间序列、信号处理"
    },
    "Frequency Domain Features": {
        "pros": [
            "✅ 揭示周期性模式",
            "✅ 适合振动信号分析"
        ],
        "cons": [
            "❌ 需要足够的数据长度",
            "❌ 丢失时间信息"
        ],
        "suitable_for": "适用于振动分析、故障诊断"
    },
    "Rolling Window Features": {
        "pros": [
            "✅ 捕捉局部变化",
            "✅ 适合非平稳信号"
        ],
        "cons": [
            "❌ 窗口大小选择敏感",
            "❌ 特征数量多"
        ],
        "suitable_for": "适用于变化信号、趋势检测"
    },
    "Enhanced Signal Features": {
        "pros": [
            "✅ 综合时频域信息",
            "✅ 物理意义明确"
        ],
        "cons": [
            "❌ 计算复杂度高",
            "❌ 需要预定义参数"
        ],
        "suitable_for": "适用于电机信号、设备监控"
    }
}


# --- 状态初始化 ---
def initialize_feature_extraction_state():
    """Initialize session state variables specific to feature extraction."""
    defaults = {
        'fe_uploaded_data': None,
        'fe_batch_data': {},  # 存储批量数据
        'fe_active_data': None,
        'fe_active_data_source': None,
        'fe_selected_columns': [],
        'fe_selected_methods': [],
        'fe_selected_features': {},  # 存储每个方法选中的特征
        'fe_extracted_features': None,
        'fe_extraction_params': {},
        'fe_original_data': None,  # 存储原始数据用于合并
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# --- 特征提取函数（可选特征版本）---
def extract_basic_statistical_features(data, columns, selected_features=None):
    """提取基础统计特征（可选特征）"""
    if selected_features is None:
        selected_features = list(FEATURE_DEFINITIONS["Basic Statistical Features"]["features"].keys())

    features = []
    feature_names = []

    for col in columns:
        col_data = data[col].dropna()
        if len(col_data) > 0:
            col_features = {}

            if "mean" in selected_features:
                col_features[f'{col}_mean'] = col_data.mean()
            if "std" in selected_features:
                col_features[f'{col}_std'] = col_data.std()
            if "min" in selected_features:
                col_features[f'{col}_min'] = col_data.min()
            if "max" in selected_features:
                col_features[f'{col}_max'] = col_data.max()
            if "median" in selected_features:
                col_features[f'{col}_median'] = col_data.median()
            if "q25" in selected_features:
                col_features[f'{col}_q25'] = col_data.quantile(0.25)
            if "q75" in selected_features:
                col_features[f'{col}_q75'] = col_data.quantile(0.75)
            if "skew" in selected_features:
                col_features[f'{col}_skew'] = skew(col_data)
            if "kurtosis" in selected_features:
                col_features[f'{col}_kurtosis'] = kurtosis(col_data)

            features.append(col_features)

    # 合并所有特征
    if features:
        combined_features = {}
        for feature_dict in features:
            combined_features.update(feature_dict)
        return pd.DataFrame([combined_features])
    else:
        return pd.DataFrame()


def extract_advanced_statistical_features(data, columns, selected_features=None):
    """提取高级统计特征（可选特征）"""
    if selected_features is None:
        selected_features = list(FEATURE_DEFINITIONS["Advanced Statistical Features"]["features"].keys())

    features = []

    for col in columns:
        col_data = data[col].dropna()
        if len(col_data) > 0:
            col_features = {}
            mean_val = col_data.mean()
            std_val = col_data.std()

            if "cv" in selected_features:
                col_features[f'{col}_cv'] = std_val / (mean_val + 1e-10)
            if "energy" in selected_features:
                col_features[f'{col}_energy'] = np.sum(col_data ** 2)
            if "entropy" in selected_features:
                hist, _ = np.histogram(col_data, bins=10)
                hist = hist / hist.sum()
                col_features[f'{col}_entropy'] = entropy(hist + 1e-10)
            if "peak_to_peak" in selected_features:
                col_features[f'{col}_peak_to_peak'] = col_data.max() - col_data.min()
            if "rms" in selected_features:
                col_features[f'{col}_rms'] = np.sqrt(np.mean(col_data ** 2))
            if "abs_mean" in selected_features:
                col_features[f'{col}_abs_mean'] = np.mean(np.abs(col_data))

            features.append(col_features)

    if features:
        combined_features = {}
        for feature_dict in features:
            combined_features.update(feature_dict)
        return pd.DataFrame([combined_features])
    else:
        return pd.DataFrame()


def extract_time_domain_features(data, columns, sample_rate=625, selected_features=None):
    """提取时域特征（可选特征）"""
    if selected_features is None:
        selected_features = list(FEATURE_DEFINITIONS["Time Domain Features"]["features"].keys())

    features = []

    for col in columns:
        col_data = data[col].dropna().values
        if len(col_data) > 10:
            col_features = {}

            if "num_peaks" in selected_features or "num_valleys" in selected_features:
                peaks, _ = find_peaks(col_data, distance=sample_rate // 10)
                valleys, _ = find_peaks(-col_data, distance=sample_rate // 10)

                if "num_peaks" in selected_features:
                    col_features[f'{col}_num_peaks'] = len(peaks)
                if "num_valleys" in selected_features:
                    col_features[f'{col}_num_valleys'] = len(valleys)

            if "zero_crossing_rate" in selected_features:
                zero_crossings = np.sum(np.diff(np.sign(col_data)) != 0)
                col_features[f'{col}_zero_crossing_rate'] = zero_crossings / len(col_data)

            if "waveform_factor" in selected_features:
                col_features[f'{col}_waveform_factor'] = np.sqrt(np.mean(col_data ** 2)) / (
                            np.mean(np.abs(col_data)) + 1e-10)

            if "crest_factor" in selected_features:
                col_features[f'{col}_crest_factor'] = np.max(np.abs(col_data)) / (
                            np.sqrt(np.mean(col_data ** 2)) + 1e-10)

            if "impulse_factor" in selected_features:
                col_features[f'{col}_impulse_factor'] = np.max(np.abs(col_data)) / (np.mean(np.abs(col_data)) + 1e-10)

            features.append(col_features)

    if features:
        combined_features = {}
        for feature_dict in features:
            combined_features.update(feature_dict)
        return pd.DataFrame([combined_features])
    else:
        return pd.DataFrame()


def extract_frequency_domain_features(data, columns, sample_rate=625, selected_features=None):
    """提取频域特征（可选特征）"""
    if selected_features is None:
        selected_features = list(FEATURE_DEFINITIONS["Frequency Domain Features"]["features"].keys())

    features = []

    for col in columns:
        col_data = data[col].dropna().values
        if len(col_data) > 10:
            col_features = {}

            # FFT
            fft_vals = np.abs(fft(col_data))
            freqs = fftfreq(len(col_data), 1 / sample_rate)

            # 只取正频率部分
            positive_freq_idx = freqs > 0
            fft_vals = fft_vals[positive_freq_idx]
            freqs = freqs[positive_freq_idx]

            if len(fft_vals) > 0:
                if "main_freq" in selected_features or "main_freq_amp" in selected_features:
                    main_freq_idx = np.argmax(fft_vals)
                    if "main_freq" in selected_features:
                        col_features[f'{col}_main_freq'] = freqs[main_freq_idx]
                    if "main_freq_amp" in selected_features:
                        col_features[f'{col}_main_freq_amp'] = fft_vals[main_freq_idx]

                if "spectral_centroid" in selected_features:
                    col_features[f'{col}_spectral_centroid'] = np.sum(freqs * fft_vals) / (np.sum(fft_vals) + 1e-10)

                if "spectral_energy" in selected_features:
                    col_features[f'{col}_spectral_energy'] = np.sum(fft_vals ** 2)

                if "spectral_entropy" in selected_features:
                    normalized_fft = fft_vals / (np.sum(fft_vals) + 1e-10)
                    col_features[f'{col}_spectral_entropy'] = entropy(normalized_fft + 1e-10)

            features.append(col_features)

    if features:
        combined_features = {}
        for feature_dict in features:
            combined_features.update(feature_dict)
        return pd.DataFrame([combined_features])
    else:
        return pd.DataFrame()


def extract_rolling_window_features(data, columns, window_size=20, selected_features=None):
    """提取滑动窗口特征（可选特征）"""
    if selected_features is None:
        selected_features = list(FEATURE_DEFINITIONS["Rolling Window Features"]["features"].keys())

    features = []

    for col in columns:
        col_data = data[col].dropna()
        if len(col_data) > window_size:
            col_features = {}

            rolling_mean = col_data.rolling(window=window_size).mean().dropna()
            rolling_std = col_data.rolling(window=window_size).std().dropna()
            rolling_min = col_data.rolling(window=window_size).min().dropna()
            rolling_max = col_data.rolling(window=window_size).max().dropna()

            if len(rolling_mean) > 0:
                if "rolling_mean_avg" in selected_features:
                    col_features[f'{col}_rolling_mean_avg'] = rolling_mean.mean()
                if "rolling_mean_std" in selected_features:
                    col_features[f'{col}_rolling_mean_std'] = rolling_mean.std()
                if "rolling_std_avg" in selected_features:
                    col_features[f'{col}_rolling_std_avg'] = rolling_std.mean()
                if "rolling_std_std" in selected_features:
                    col_features[f'{col}_rolling_std_std'] = rolling_std.std()
                if "rolling_min_avg" in selected_features:
                    col_features[f'{col}_rolling_min_avg'] = rolling_min.mean()
                if "rolling_max_avg" in selected_features:
                    col_features[f'{col}_rolling_max_avg'] = rolling_max.mean()
                if "rolling_range_avg" in selected_features:
                    col_features[f'{col}_rolling_range_avg'] = (rolling_max - rolling_min).mean()

            features.append(col_features)

    if features:
        combined_features = {}
        for feature_dict in features:
            combined_features.update(feature_dict)
        return pd.DataFrame([combined_features])
    else:
        return pd.DataFrame()


def extract_enhanced_signal_features(data, columns, sample_rate=625, selected_features=None):
    """提取增强信号特征（可选特征）"""
    if selected_features is None:
        selected_features = list(FEATURE_DEFINITIONS["Enhanced Signal Features"]["features"].keys())

    features = []

    for col in columns:
        col_data = data[col].dropna().values
        if len(col_data) > 100:
            col_features = {}

            # 启动检测（前20%数据）
            startup_samples = min(int(0.2 * len(col_data)), len(col_data))
            startup_data = col_data[:startup_samples]
            steady_data = col_data[startup_samples:]

            if "startup_max" in selected_features:
                col_features[f'{col}_startup_max'] = np.max(np.abs(startup_data))

            if "startup_time" in selected_features:
                col_features[f'{col}_startup_time'] = np.argmax(np.abs(startup_data)) / sample_rate

            if "steady_mean" in selected_features:
                col_features[f'{col}_steady_mean'] = np.mean(steady_data)

            if "steady_std" in selected_features:
                col_features[f'{col}_steady_std'] = np.std(steady_data)

            if "autocorr_lag10" in selected_features:
                if len(steady_data) > 20:
                    autocorr = np.correlate(steady_data, steady_data, mode='same')
                    autocorr = autocorr[len(autocorr) // 2:]
                    autocorr = autocorr / (autocorr[0] + 1e-10)
                    col_features[f'{col}_autocorr_lag10'] = autocorr[min(10, len(autocorr) - 1)]
                else:
                    col_features[f'{col}_autocorr_lag10'] = 0

            features.append(col_features)

    if features:
        combined_features = {}
        for feature_dict in features:
            combined_features.update(feature_dict)
        return pd.DataFrame([combined_features])
    else:
        return pd.DataFrame()


# --- 批量处理函数 ---
def load_folder_data(folder_path, file_pattern="*.xlsx"):
    """加载文件夹中的所有Excel文件"""
    all_data = {}
    file_paths = glob.glob(os.path.join(folder_path, file_pattern))

    if not file_paths:
        file_paths = glob.glob(os.path.join(folder_path, "*.xls"))

    for file_path in file_paths:
        try:
            filename = os.path.basename(file_path)
            df = pd.read_excel(file_path)
            all_data[filename] = df
        except Exception as e:
            st.warning(f"无法读取文件 {filename}: {e}")

    return all_data


# --- 主页面函数 ---
def show_feature_extraction_page():
    """显示特征提取页面的主函数 - 优化版"""
    # 初始化状态
    initialize_feature_extraction_state()

    # 使用缓存装饰器优化数据加载
    @st.cache_data
    def load_data_cached(file_path):
        """缓存文件加载"""
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        else:
            return pd.read_excel(file_path)

    # 页面标题
    st.title("🔍 高级特征提取")
    st.markdown("---")

    # 页面说明
    with st.expander("ℹ️ 功能说明", expanded=False):
        st.markdown("""
        **高级特征提取模块**支持：

        🎯 **核心功能**:
        - 批量处理文件夹中的所有Excel文件
        - 自定义选择每个方法中的具体特征
        - 支持与原始数据合并导出
        - 多种特征提取方法组合使用
        """)

    # 主要内容区域
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🎛️ 控制面板")

        # 数据上传部分使用form减少刷新
        with st.container():
            st.markdown("### 📁 数据源")

            data_source = st.radio(
                "选择数据源:",
                ["上传单个文件", "批量导入文件夹", "使用示例数据"],
                key="fe_data_source",
                label_visibility="collapsed"
            )

            if data_source == "上传单个文件":
                uploaded_file = st.file_uploader(
                    "上传数据文件",
                    type=['csv', 'xlsx', 'xls'],
                    help="支持CSV和Excel格式文件",
                    key="fe_file_uploader"
                )

                if uploaded_file is not None:
                    # 使用唯一键避免重复处理
                    file_key = f"{uploaded_file.name}_{uploaded_file.size}"
                    if 'last_processed_file' not in st.session_state or st.session_state.last_processed_file != file_key:
                        try:
                            if uploaded_file.name.endswith('.csv'):
                                df = pd.read_csv(uploaded_file)
                            else:
                                df = pd.read_excel(uploaded_file)

                            st.session_state.fe_active_data = df
                            st.session_state.fe_original_data = df.copy()
                            st.session_state.last_processed_file = file_key
                            st.success(f"✅ 成功加载: {df.shape}")
                        except Exception as e:
                            st.error(f"文件加载失败: {e}")

            elif data_source == "批量导入文件夹":
                with st.form("batch_import_form"):
                    st.markdown("#### 📂 批量导入设置")

                    folder_path = st.text_input(
                        "输入文件夹路径:",
                        placeholder="例如: C:/Users/data/excel_files",
                        help="输入包含Excel文件的文件夹路径"
                    )

                    file_pattern = st.selectbox(
                        "文件类型:",
                        ["*.xlsx", "*.xls", "*.csv", "*.*"],
                        help="选择要导入的文件类型"
                    )

                    submitted = st.form_submit_button("🔄 扫描文件夹")

                    if submitted and folder_path and os.path.exists(folder_path):
                        with st.spinner("正在加载文件..."):
                            batch_data = load_folder_data(folder_path, file_pattern)
                            if batch_data:
                                st.session_state.fe_batch_data = batch_data
                                st.success(f"✅ 成功加载 {len(batch_data)} 个文件")

            elif data_source == "使用示例数据":
                if st.button("生成示例数据", key="gen_sample_data"):
                    np.random.seed(42)
                    n_samples = 1000
                    time = np.linspace(0, 10, n_samples)

                    df = pd.DataFrame({
                        'Time': time,
                        'Signal_1': np.sin(2 * np.pi * time) + np.random.randn(n_samples) * 0.1,
                        'Signal_2': np.cos(2 * np.pi * time * 0.5) + np.random.randn(n_samples) * 0.1,
                        'Signal_3': np.sin(2 * np.pi * time * 2) * np.exp(-time / 5) + np.random.randn(
                            n_samples) * 0.05,
                        'Voltage': 220 + 5 * np.sin(2 * np.pi * time * 0.1) + np.random.randn(n_samples),
                        'Current': 10 * (1 - np.exp(-time / 2)) + np.random.randn(n_samples) * 0.5
                    })

                    st.session_state.fe_active_data = df
                    st.session_state.fe_original_data = df.copy()
                    st.success("✅ 示例数据已生成")

    with col2:
        st.subheader("📊 特征提取设置")

        if st.session_state.fe_active_data is not None:
            data = st.session_state.fe_active_data

            # 数据预览
            with st.expander("数据预览", expanded=False):
                st.dataframe(data.head())

                col1_info, col2_info, col3_info = st.columns(3)
                with col1_info:
                    st.metric("数据行数", len(data))
                with col2_info:
                    st.metric("数据列数", len(data.columns))
                with col3_info:
                    numeric_cols = data.select_dtypes(include=[np.number]).columns
                    st.metric("数值列数", len(numeric_cols))

            # 列选择 - 使用form避免刷新
            st.markdown("### 🎯 选择要处理的列")
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

            if numeric_columns:
                with st.form("column_selection_form"):
                    quick_select = st.radio(
                        "快速选择:",
                        ["自定义选择", "选择全部数值列", "选择前5列"],
                        horizontal=True
                    )

                    if quick_select == "选择全部数值列":
                        selected_columns = numeric_columns
                    elif quick_select == "选择前5列":
                        selected_columns = numeric_columns[:min(5, len(numeric_columns))]
                    else:
                        default_cols = st.session_state.fe_selected_columns if st.session_state.fe_selected_columns else numeric_columns[
                                                                                                                         :min(
                                                                                                                             3,
                                                                                                                             len(numeric_columns))]
                        selected_columns = st.multiselect(
                            "选择列:",
                            numeric_columns,
                            default=default_cols
                        )

                    submitted_cols = st.form_submit_button("确认列选择")
                    if submitted_cols:
                        st.session_state.fe_selected_columns = selected_columns
                        if selected_columns:
                            st.success(f"✅ 已选择 {len(selected_columns)} 列")
                        else:
                            st.error("请至少选择一列！")
            else:
                st.error("数据中没有数值列！")
                selected_columns = []

    # 特征提取方法和特征选择
    if st.session_state.fe_active_data is not None and st.session_state.fe_selected_columns:
        st.markdown("---")
        st.subheader("🔧 特征提取方法和特征选择")

        # 使用单个大form处理所有选择
        with st.form("feature_extraction_form", clear_on_submit=False):
            # 方法选择
            st.markdown("### 选择特征提取方法")
            selected_methods = st.multiselect(
                "选择特征提取方法:",
                list(FEATURE_METHODS_INFO.keys()),
                default=st.session_state.fe_selected_methods if st.session_state.fe_selected_methods else [
                    "Basic Statistical Features"],
                help="可以选择多个方法组合使用"
            )

            st.markdown("---")
            st.markdown("### 📋 选择具体特征")

            # 存储每个方法的选择
            feature_selections = {}

            if selected_methods:
                # 为每个方法创建特征选择
                for method in selected_methods:
                    st.markdown(f"#### 🔸 {method}")

                    # 显示方法说明
                    method_info = FEATURE_METHODS_INFO[method]
                    col_desc, col_suit = st.columns(2)
                    with col_desc:
                        st.info(f"**{FEATURE_DEFINITIONS[method]['description']}**")
                    with col_suit:
                        st.success(f"*{method_info['suitable_for']}*")

                    # 获取特征列表
                    available_features = list(FEATURE_DEFINITIONS[method]["features"].keys())
                    feature_descriptions = FEATURE_DEFINITIONS[method]["features"]

                    # 获取默认选择
                    default_features = st.session_state.get(
                        f"confirmed_features_{method}",
                        available_features  # 默认全选
                    )

                    # 创建两列布局显示特征选项
                    col_left, col_right = st.columns(2)

                    selected_features = []
                    for idx, (feat_key, feat_desc) in enumerate(feature_descriptions.items()):
                        # 交替使用左右列
                        with col_left if idx % 2 == 0 else col_right:
                            is_selected = st.checkbox(
                                f"{feat_key}",
                                value=feat_key in default_features,
                                key=f"feat_{method}_{feat_key}",
                                help=feat_desc
                            )
                            if is_selected:
                                selected_features.append(feat_key)

                    feature_selections[method] = selected_features

                    # 显示统计
                    st.markdown(f"✅ 已选择 **{len(selected_features)}/{len(available_features)}** 个特征")
                    st.markdown("---")

                # 参数设置部分
                st.markdown("### ⚙️ 参数设置")
                params = {}

                for method in selected_methods:
                    if method in ["Rolling Window Features"]:
                        params[method] = {
                            'window_size': st.slider(
                                f"{method} - 窗口大小",
                                min_value=5,
                                max_value=100,
                                value=20,
                                key=f"param_window_{method}"
                            )
                        }
                    elif method in ["Time Domain Features", "Frequency Domain Features", "Enhanced Signal Features"]:
                        params[method] = {
                            'sample_rate': st.number_input(
                                f"{method} - 采样率 (Hz)",
                                min_value=1,
                                max_value=10000,
                                value=625,
                                key=f"param_rate_{method}"
                            )
                        }

            # 提交按钮
            col_submit1, col_submit2 = st.columns([3, 1])
            with col_submit1:
                submitted = st.form_submit_button(
                    "🚀 确认设置并开始特征提取",
                    type="primary",
                    use_container_width=True
                )
            with col_submit2:
                # 显示总特征数预估
                if selected_methods and feature_selections:
                    total_features = sum(len(features) for features in feature_selections.values())
                    st.metric("预计特征数", total_features * len(st.session_state.fe_selected_columns))

            # 处理提交
            if submitted:
                # 验证是否有选择
                if not selected_methods:
                    st.error("请至少选择一个特征提取方法！")
                elif all(len(features) == 0 for features in feature_selections.values()):
                    st.error("请至少选择一个特征！")
                else:
                    # 保存选择
                    st.session_state.fe_selected_methods = selected_methods
                    st.session_state.fe_extraction_params = params

                    for method, features in feature_selections.items():
                        st.session_state[f"confirmed_features_{method}"] = features

                    st.session_state.fe_selected_features = feature_selections

                    # 执行特征提取
                    with st.spinner("正在提取特征..."):
                        try:
                            all_features = []
                            feature_summary = []

                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            for idx, method in enumerate(selected_methods):
                                status_text.text(f"正在处理: {method}")

                                method_features = None
                                method_selected_features = feature_selections[method]

                                if not method_selected_features:
                                    continue

                                # 根据方法调用相应的特征提取函数
                                data = st.session_state.fe_active_data
                                selected_columns = st.session_state.fe_selected_columns

                                if method == "Basic Statistical Features":
                                    method_features = extract_basic_statistical_features(
                                        data, selected_columns, method_selected_features
                                    )
                                elif method == "Advanced Statistical Features":
                                    method_features = extract_advanced_statistical_features(
                                        data, selected_columns, method_selected_features
                                    )
                                elif method == "Time Domain Features":
                                    sample_rate = params.get(method, {}).get('sample_rate', 625)
                                    method_features = extract_time_domain_features(
                                        data, selected_columns, sample_rate, method_selected_features
                                    )
                                elif method == "Frequency Domain Features":
                                    sample_rate = params.get(method, {}).get('sample_rate', 625)
                                    method_features = extract_frequency_domain_features(
                                        data, selected_columns, sample_rate, method_selected_features
                                    )
                                elif method == "Rolling Window Features":
                                    window_size = params.get(method, {}).get('window_size', 20)
                                    method_features = extract_rolling_window_features(
                                        data, selected_columns, window_size, method_selected_features
                                    )
                                elif method == "Enhanced Signal Features":
                                    sample_rate = params.get(method, {}).get('sample_rate', 625)
                                    method_features = extract_enhanced_signal_features(
                                        data, selected_columns, sample_rate, method_selected_features
                                    )

                                if method_features is not None and not method_features.empty:
                                    all_features.append(method_features)
                                    feature_summary.append({
                                        'Method': method,
                                        'Features': len(method_features.columns),
                                        'Selected Features': len(method_selected_features),
                                        'Columns': len(selected_columns)
                                    })

                                progress_bar.progress((idx + 1) / len(selected_methods))

                            status_text.empty()

                            # 合并所有特征
                            if all_features:
                                combined_features = pd.concat(all_features, axis=1)
                                st.session_state.fe_extracted_features = combined_features
                                st.session_state.fe_extraction_success = True  # 添加成功标志

                                st.success(f"✅ 特征提取完成！共提取 {len(combined_features.columns)} 个特征")

                                # 显示特征汇总
                                st.markdown("### 📊 特征提取汇总")
                                summary_df = pd.DataFrame(feature_summary)
                                st.dataframe(summary_df, use_container_width=True)

                                # 显示特征结果
                                with st.expander("查看提取的特征", expanded=True):
                                    # 限制显示行数以提高性能
                                    st.dataframe(combined_features.head(100))
                                    if len(combined_features) > 100:
                                        st.info(f"显示前100行，共{len(combined_features)}行")
                            else:
                                st.warning("没有成功提取任何特征")
                                st.session_state.fe_extraction_success = False

                        except Exception as e:
                            st.error(f"特征提取失败: {e}")
                            st.code(traceback.format_exc())
                            st.session_state.fe_extraction_success = False

    # 导出功能 - 移到form外部
    if st.session_state.get('fe_extracted_features') is not None:
        st.markdown("---")
        st.subheader("💾 导出特征")
        export_features_enhanced(st.session_state.fe_extracted_features)

    # 批量处理导出 - 也移到form外部
    if st.session_state.fe_batch_data and st.session_state.get('fe_selected_methods'):
        st.markdown("---")
        st.subheader("🔄 批量处理")

        col1_batch, col2_batch = st.columns([2, 1])
        with col1_batch:
            st.info(f"检测到 {len(st.session_state.fe_batch_data)} 个批量文件")
        with col2_batch:
            if st.button("🚀 批量处理所有文件", type="primary"):
                process_batch_files()


def process_batch_files():
    """批量处理文件 - 独立函数"""
    with st.spinner("正在批量处理..."):
        batch_results = {}
        progress_bar = st.progress(0)

        for i, (filename, df) in enumerate(st.session_state.fe_batch_data.items()):
            try:
                # 获取数值列
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

                if numeric_cols:
                    # 提取特征
                    file_features = []
                    for method in st.session_state.fe_selected_methods:
                        method_selected_features = st.session_state.fe_selected_features.get(method, [])
                        if not method_selected_features:
                            continue

                        # 根据方法提取特征
                        params = st.session_state.fe_extraction_params.get(method, {})

                        if method == "Basic Statistical Features":
                            features = extract_basic_statistical_features(
                                df, numeric_cols, method_selected_features
                            )
                        elif method == "Advanced Statistical Features":
                            features = extract_advanced_statistical_features(
                                df, numeric_cols, method_selected_features
                            )
                        elif method == "Time Domain Features":
                            sample_rate = params.get('sample_rate', 625)
                            features = extract_time_domain_features(
                                df, numeric_cols, sample_rate, method_selected_features
                            )
                        elif method == "Frequency Domain Features":
                            sample_rate = params.get('sample_rate', 625)
                            features = extract_frequency_domain_features(
                                df, numeric_cols, sample_rate, method_selected_features
                            )
                        elif method == "Rolling Window Features":
                            window_size = params.get('window_size', 20)
                            features = extract_rolling_window_features(
                                df, numeric_cols, window_size, method_selected_features
                            )
                        elif method == "Enhanced Signal Features":
                            sample_rate = params.get('sample_rate', 625)
                            features = extract_enhanced_signal_features(
                                df, numeric_cols, sample_rate, method_selected_features
                            )

                        if features is not None and not features.empty:
                            file_features.append(features)

                    if file_features:
                        combined = pd.concat(file_features, axis=1)
                        batch_results[filename] = combined

            except Exception as e:
                st.warning(f"处理 {filename} 失败: {e}")

            progress_bar.progress((i + 1) / len(st.session_state.fe_batch_data))

        # 导出批量结果
        if batch_results:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                for filename, features in batch_results.items():
                    sheet_name = filename.replace('.xlsx', '').replace('.xls', '')[:31]
                    features.to_excel(writer, sheet_name=sheet_name, index=False)

            st.download_button(
                label="📥 下载批量处理结果",
                data=buffer.getvalue(),
                file_name=f"batch_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            st.success(f"✅ 批量处理完成！处理了 {len(batch_results)} 个文件")


def export_features_enhanced(features_df):
    """增强的导出功能，支持与原始数据合并"""
    st.markdown("---")
    st.subheader("💾 导出特征")

    col1_export, col2_export = st.columns(2)

    with col1_export:
        st.markdown("### 📄 文件导出")

        # 选择导出模式
        export_mode = st.radio(
            "导出模式:",
            ["仅导出特征", "与原始数据合并导出"],
            help="选择是只导出特征还是与原始数据合并"
        )

        # 准备导出数据
        if export_mode == "与原始数据合并导出" and st.session_state.fe_original_data is not None:
            # 将特征扩展到原始数据的长度
            original_data = st.session_state.fe_original_data.copy()

            # 如果特征只有一行，扩展到原始数据的长度
            if len(features_df) == 1:
                features_expanded = pd.concat([features_df] * len(original_data), ignore_index=True)
            else:
                features_expanded = features_df

            # 合并数据
            export_data = pd.concat([original_data.reset_index(drop=True),
                                     features_expanded.reset_index(drop=True)], axis=1)
            st.info(f"将导出 {len(export_data)} 行, {len(export_data.columns)} 列数据")
        else:
            export_data = features_df
            st.info(f"将导出 {len(export_data)} 行, {len(export_data.columns)} 列特征")

        # 选择导出格式
        export_format = st.selectbox("选择导出格式:", ["CSV", "Excel"])

        if export_format == "CSV":
            csv = export_data.to_csv(index=False)
            st.download_button(
                label="📥 下载CSV文件",
                data=csv,
                file_name=f"features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:  # Excel
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                # 导出主数据
                export_data.to_excel(writer, sheet_name='Features', index=False)

                # 添加特征说明表
                if st.session_state.fe_selected_features:
                    feature_info = []
                    for method, features in st.session_state.fe_selected_features.items():
                        for feat in features:
                            feature_info.append({
                                'Method': method,
                                'Feature': feat,
                                'Description': FEATURE_DEFINITIONS[method]['features'].get(feat, '')
                            })

                    if feature_info:
                        pd.DataFrame(feature_info).to_excel(writer, sheet_name='Feature_Info', index=False)

                # 添加方法说明表
                if st.session_state.fe_selected_methods:
                    method_info = []
                    for method in st.session_state.fe_selected_methods:
                        method_info.append({
                            'Method': method,
                            'Description': FEATURE_DEFINITIONS[method]['description'],
                            'Suitable For': FEATURE_METHODS_INFO[method]['suitable_for']
                        })
                    pd.DataFrame(method_info).to_excel(writer, sheet_name='Method_Info', index=False)

            st.download_button(
                label="📥 下载Excel文件",
                data=buffer.getvalue(),
                file_name=f"features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    with col2_export:
        st.markdown("### 🔄 批量处理导出")

        if st.session_state.fe_batch_data:
            st.info(f"检测到 {len(st.session_state.fe_batch_data)} 个批量文件")

            if st.button("🚀 批量处理所有文件", type="primary"):
                with st.spinner("正在批量处理..."):
                    batch_results = {}
                    progress_bar = st.progress(0)

                    for i, (filename, df) in enumerate(st.session_state.fe_batch_data.items()):
                        try:
                            # 获取数值列
                            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

                            if numeric_cols:
                                # 提取特征
                                file_features = []
                                for method in st.session_state.fe_selected_methods:
                                    method_selected_features = st.session_state.fe_selected_features.get(method, [])
                                    if not method_selected_features:
                                        continue

                                    # 根据方法提取特征
                                    if method == "Basic Statistical Features":
                                        features = extract_basic_statistical_features(
                                            df, numeric_cols, method_selected_features
                                        )
                                    # ... 其他方法类似

                                    if features is not None and not features.empty:
                                        file_features.append(features)

                                if file_features:
                                    combined = pd.concat(file_features, axis=1)
                                    batch_results[filename] = combined

                        except Exception as e:
                            st.warning(f"处理 {filename} 失败: {e}")

                        progress_bar.progress((i + 1) / len(st.session_state.fe_batch_data))

                    # 导出批量结果
                    if batch_results:
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                            for filename, features in batch_results.items():
                                sheet_name = filename.replace('.xlsx', '').replace('.xls', '')[:31]
                                features.to_excel(writer, sheet_name=sheet_name, index=False)

                        st.download_button(
                            label="📥 下载批量处理结果",
                            data=buffer.getvalue(),
                            file_name=f"batch_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                        st.success(f"✅ 批量处理完成！处理了 {len(batch_results)} 个文件")