# -*- coding: utf-8 -*-
"""
Regression Tutorial Module for Streamlit App (Self-Contained Version)
Provides an interactive interface to demonstrate regression algorithms.
Includes necessary plotting functions directly.
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm # Added import
import platform # Added import
import os # Added import
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import traceback

# --- Font Setup (Included directly in this file) ---
def setup_better_chinese_font():
    """设置更好的中文字体支持"""
    system = platform.system()
    font_candidates = []
    if system == 'Windows':
        font_candidates = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']
    elif system == 'Darwin':  # macOS
        font_candidates = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Apple LiGothic Medium']
    else:  # Linux 和其他系统
        font_candidates = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'Droid Sans Fallback']
    font_candidates.extend(['DejaVu Sans', 'Arial', 'Helvetica'])
    font_prop = None
    for font_name in font_candidates:
        try:
            font_path = fm.findfont(fm.FontProperties(family=font_name))
            if os.path.exists(font_path) and 'DejaVuSans' not in font_path:
                print(f"字体日志 (Tutorial): 使用字体 '{font_name}' 在路径: {font_path}")
                plt.rcParams['font.family'] = 'sans-serif'
                plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
                font_prop = fm.FontProperties(family=font_name)
                break
        except Exception as e:
            print(f"字体日志 (Tutorial): 尝试字体 {font_name} 失败: {e}")
    if not font_prop:
        print("字体日志 (Tutorial): 未找到合适的中文字体。")
    plt.rcParams['axes.unicode_minus'] = False
    return font_prop

FONT_PROP = setup_better_chinese_font() # Define FONT_PROP globally in this module

# --- Plotting Helper Functions (Included directly) ---
def apply_plot_style(ax):
    """应用统一的绘图样式"""
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
    font_kwargs = {'fontproperties': FONT_PROP} if FONT_PROP else {}
    if font_kwargs:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(FONT_PROP)
        ax.xaxis.label.set_fontproperties(FONT_PROP)
        ax.yaxis.label.set_fontproperties(FONT_PROP)
        ax.title.set_fontproperties(FONT_PROP)
    return ax

def create_figure_with_safe_dimensions(width_inches, height_inches, dpi=80):
    """创建不会超出Matplotlib限制的图形尺寸"""
    max_pixels = 65000
    width_dpi = max_pixels / width_inches if width_inches > 0 else dpi
    height_dpi = max_pixels / height_inches if height_inches > 0 else dpi
    safe_dpi = min(width_dpi, height_dpi, dpi)
    fig, ax = plt.subplots(figsize=(width_inches, height_inches), dpi=safe_dpi)
    return fig, ax

# --- Core Plotting Functions (Copied from regression_validation.py logic) ---
def plot_validation_results(true_values, predictions, indices=None, model_name="Model"):
    """绘制模型预测值 vs 真实值 (回归)"""
    fig, ax = create_figure_with_safe_dimensions(10, 6) # Use the local helper
    apply_plot_style(ax) # Use the local helper
    font_kwargs = {'fontproperties': FONT_PROP} if FONT_PROP else {}

    # Ensure inputs are numpy arrays
    true_values_np = np.asarray(true_values)
    predictions_np = np.asarray(predictions)

    if indices is None:
        indices = np.arange(len(true_values_np))
    else:
        indices = np.asarray(indices)

    if len(true_values_np) != len(predictions_np) or len(indices) != len(true_values_np):
        ax.text(0.5, 0.5, '绘图错误：数据长度不匹配', ha='center', va='center', transform=ax.transAxes, color='red', **font_kwargs)
        return fig

    try:
        # Sort data by indices for proper line plotting if indices are meaningful
        if len(indices) > 1 and np.any(np.diff(indices) < 0): # Check if sorting is needed
             sort_indices_order = np.argsort(indices)
             sorted_indices = indices[sort_indices_order]
             sorted_true = true_values_np[sort_indices_order]
             sorted_pred = predictions_np[sort_indices_order]
        else: # If indices are sequential or single point, no need to sort
             sorted_indices = indices
             sorted_true = true_values_np
             sorted_pred = predictions_np

        # Plot lines or scatter depending on data size
        marker_true = 'o'
        marker_pred = 'x'
        linestyle_true = '-'
        linestyle_pred = '--'
        markersize_true = 3
        markersize_pred = 4
        if len(sorted_indices) > 200: # Use smaller markers for large datasets
             marker_true = '.'
             marker_pred = '.'
             markersize_true = 2
             markersize_pred = 2
             linestyle_true = 'None' # Use scatter plot for large data
             linestyle_pred = 'None'


        ax.plot(sorted_indices, sorted_true, color='#2ecc71', label='真实值',
                linewidth=1.5, linestyle=linestyle_true, marker=marker_true, markersize=markersize_true, alpha=0.8)
        ax.plot(sorted_indices, sorted_pred, color='#e74c3c', label=f'{model_name}预测值',
                linewidth=1.5, linestyle=linestyle_pred, marker=marker_pred, markersize=markersize_pred, alpha=0.8)

        ax.set_title('模型预测值 vs 真实值', **font_kwargs)
        ax.set_xlabel('样本索引' if len(indices) <= 1 or np.all(np.diff(indices) > 0) else '索引值', **font_kwargs) # Adjust x-label based on indices
        ax.set_ylabel('值', **font_kwargs)
        legend = ax.legend(prop=FONT_PROP, fontsize=9)
        if FONT_PROP:
            for text in legend.get_texts():
                text.set_fontproperties(FONT_PROP)
        plt.tight_layout()
    except Exception as e:
        ax.text(0.5, 0.5, f'绘图错误：{str(e)}', ha='center', va='center', transform=ax.transAxes, color='red', **font_kwargs)

    return fig


def plot_residuals(true_values, predictions, indices=None, model_name="Model"):
    """绘制残差图 (真实值 - 预测值)"""
    fig, ax = create_figure_with_safe_dimensions(10, 6) # Use the local helper
    apply_plot_style(ax) # Use the local helper
    font_kwargs = {'fontproperties': FONT_PROP} if FONT_PROP else {}

    # Ensure inputs are numpy arrays
    true_values_np = np.asarray(true_values)
    predictions_np = np.asarray(predictions)

    if indices is None:
        indices = np.arange(len(true_values_np))
    else:
        indices = np.asarray(indices)

    if len(true_values_np) != len(predictions_np) or len(indices) != len(true_values_np):
        ax.text(0.5, 0.5, '绘图错误：数据长度不匹配', ha='center', va='center', transform=ax.transAxes, color='red', **font_kwargs)
        return fig

    try:
        residuals = true_values_np - predictions_np

        # Sort by indices if needed
        if len(indices) > 1 and np.any(np.diff(indices) < 0): # Check if sorting is needed
            sort_indices_order = np.argsort(indices)
            sorted_indices = indices[sort_indices_order]
            sorted_residuals = residuals[sort_indices_order]
        else:
            sorted_indices = indices
            sorted_residuals = residuals

        # Plot residuals
        marker = '.'
        linestyle = '-'
        if len(sorted_indices) > 200: # Use scatter for large data
             linestyle = 'None'

        ax.plot(sorted_indices, sorted_residuals, color='#3498db', label=f'{model_name}残差',
                linewidth=1.5, linestyle=linestyle, marker=marker, markersize=3, alpha=0.8)
        ax.axhline(y=0, color='#e74c3c', linestyle='--', alpha=0.7, linewidth=1.0)

        ax.set_title('残差分析 (真实值 - 预测值)', **font_kwargs)
        ax.set_xlabel('样本索引' if len(indices) <= 1 or np.all(np.diff(indices) > 0) else '索引值', **font_kwargs)
        ax.set_ylabel('残差', **font_kwargs)
        legend = ax.legend(prop=FONT_PROP, fontsize=9)
        if FONT_PROP:
            for text in legend.get_texts():
                text.set_fontproperties(FONT_PROP)
        plt.tight_layout()
    except Exception as e:
        ax.text(0.5, 0.5, f'绘图错误：{str(e)}', ha='center', va='center', transform=ax.transAxes, color='red', **font_kwargs)

    return fig


# --- 教学状态初始化 ---
def initialize_regression_tutorial_state():
    """初始化回归教学模块专用的会话状态变量"""
    defaults = {
        'reg_tut_dataset_name': 'Synthetic', 'reg_tut_n_samples': 200, 'reg_tut_n_features': 1,
        'reg_tut_n_informative': 1, 'reg_tut_noise': 15.0, 'reg_tut_bias': 0.0,
        'reg_tut_method_select': 'Linear Regression', # Use key name for state
        'reg_tut_ridge_alpha': 1.0,
        'reg_tut_lasso_alpha': 1.0,
        'reg_tut_svr_c': 1.0, 'reg_tut_svr_kernel': 'rbf', 'reg_tut_svr_epsilon': 0.1,
        'reg_tut_rf_n_estimators': 100, 'reg_tut_rf_max_depth': 0, # Use 0 for None
        'reg_tut_data_X_raw': None,
        'reg_tut_data_X': None, # 标准化后的特征数据
        'reg_tut_data_y': None, # 真实目标值
        'reg_tut_X_train': None, 'reg_tut_X_test': None,
        'reg_tut_y_train': None, 'reg_tut_y_test': None,
        'reg_tut_model': None, # 训练好的模型
        'reg_tut_y_pred': None, # 测试集预测值
        'reg_tut_scaler': StandardScaler(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
        if key == 'reg_tut_rf_max_depth' and st.session_state.get(key) is None:
             st.session_state[key] = 0 # Ensure it's 0 if None

# --- 教学 UI 函数 ---
def show_regression_tutorial_page():
    """创建交互式回归教学演示的用户界面"""
    initialize_regression_tutorial_state()

    st.header("🎓 回归教学演示")
    st.markdown("""
    欢迎来到回归教学模块！在这里，你可以：
    1.  生成 **合成回归数据**。
    2.  调整生成数据的 **参数**，例如特征数量、噪声水平等。
    3.  选择 **回归算法**（如线性回归, Ridge, Lasso, SVR, 随机森林回归）并调整其关键参数。
    4.  **训练模型** 并 **可视化** 结果（例如预测值 vs 真实值、残差图）。
    5.  查看 **评估指标**（如 R², MSE, MAE）和结果解读。

    通过互动操作，直观理解不同回归算法如何拟合数据以及参数变化对模型性能的影响。
    """)
    st.markdown("---")

    # --- 1. 生成合成数据集 ---
    st.subheader("1. 生成合成数据集")
    col_data1, col_data2 = st.columns(2)
    with col_data1:
        st.slider(
            "样本数量:", min_value=50, max_value=1000,
            value=st.session_state.reg_tut_n_samples, step=50, key="reg_tut_samples",
            help="生成数据点的总数。"
        )
        st.slider(
            "特征数量:", min_value=1, max_value=5,
            value=st.session_state.reg_tut_n_features, step=1, key="reg_tut_n_features",
            help="生成数据的特征维度。"
        )

        # --- Conditional Control for n_informative ---
        current_n_features = st.session_state.reg_tut_n_features

        if current_n_features <= 1:
            st.session_state.reg_tut_n_informative = 1
            st.markdown("**信息特征数:** 1 (当总特征数为1时固定)")
        else:
            max_informative = current_n_features
            current_informative_val = st.session_state.get('reg_tut_n_informative', 1)
            valid_informative_val = max(1, min(current_informative_val, max_informative))
            st.slider(
                "信息特征数:", min_value=1, max_value=max_informative,
                value=valid_informative_val, step=1, key="reg_tut_n_informative",
                help="真正与目标值相关的特征数量。"
            )

    with col_data2:
        st.slider(
            "噪声水平 (noise):", min_value=0.0, max_value=50.0,
            value=st.session_state.reg_tut_noise, step=1.0, format="%.1f", key="reg_tut_noise",
            help="添加到目标值上的高斯噪声的标准差。"
        )
        st.slider(
            "偏置 (bias):", min_value=-50.0, max_value=50.0,
            value=st.session_state.reg_tut_bias, step=5.0, format="%.1f", key="reg_tut_bias",
            help="添加到目标值上的固定偏移量。"
        )

    # --- 生成数据集按钮 ---
    if st.button("🔄 生成/更新数据集", key="reg_tut_generate_data"):
        # ... (生成数据集逻辑保持不变，读取 state) ...
        X_raw, y_true = None, None
        with st.spinner("正在生成数据..."):
            try:
                random_state_data = 42
                n_samples = st.session_state.reg_tut_n_samples
                n_features = st.session_state.reg_tut_n_features
                n_informative = 1 if n_features <= 1 else st.session_state.reg_tut_n_informative
                n_informative = min(n_informative, n_features)
                noise = st.session_state.reg_tut_noise
                bias = st.session_state.reg_tut_bias

                X_raw, y_true = make_regression(
                    n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                    noise=noise, bias=bias, random_state=random_state_data
                )
                if n_features == 1: X_raw = X_raw.reshape(-1, 1)

                st.session_state.reg_tut_data_X_raw = X_raw
                st.session_state.reg_tut_data_X = st.session_state.reg_tut_scaler.fit_transform(X_raw)
                st.session_state.reg_tut_data_y = y_true

                st.session_state.reg_tut_X_train, st.session_state.reg_tut_X_test, \
                st.session_state.reg_tut_y_train, st.session_state.reg_tut_y_test = train_test_split(
                    st.session_state.reg_tut_data_X, st.session_state.reg_tut_data_y,
                    test_size=0.3, random_state=random_state_data
                )
                st.session_state.reg_tut_model = None
                st.session_state.reg_tut_y_pred = None
                st.success("合成回归数据集已生成并分割。")
            except Exception as data_err:
                st.error(f"生成数据集时出错: {data_err}")
                print(traceback.format_exc())
                st.session_state.reg_tut_data_X = None


    # --- 显示生成的数据集 (仅当单特征时) ---
    # 使用本地定义的绘图函数
    if st.session_state.reg_tut_data_X is not None and st.session_state.reg_tut_n_features == 1:
        st.write("---")
        st.markdown("#### 数据集预览（单特征 vs 目标值）")
        try:
            fig_data, ax_data = create_figure_with_safe_dimensions(8, 5)
            ax_data.scatter(
                st.session_state.reg_tut_data_X,
                st.session_state.reg_tut_data_y,
                s=30, alpha=0.6, edgecolors='k', linewidth=0.5, label="数据点"
            )
            apply_plot_style(ax_data) # Use local function
            title_str = "合成回归数据预览"
            ax_data.set_title(title_str, fontproperties=FONT_PROP if FONT_PROP else None)
            ax_data.set_xlabel("特征 (标准化后)", fontproperties=FONT_PROP if FONT_PROP else None)
            ax_data.set_ylabel("目标值", fontproperties=FONT_PROP if FONT_PROP else None)
            legend = ax_data.legend(prop=FONT_PROP)
            if FONT_PROP:
                 for text in legend.get_texts(): text.set_fontproperties(FONT_PROP)
            st.pyplot(fig_data)
            plt.close(fig_data)
        except Exception as plot_err:
            st.warning(f"绘制数据集图表时出错: {plot_err}")
            print(traceback.format_exc())

    elif st.session_state.reg_tut_data_X is None:
        st.info("请点击 **“🔄 生成/更新数据集”** 按钮来创建数据。")
        return

    st.markdown("---")

    # --- 2. 选择回归方法与参数 ---
    st.subheader("2. 选择回归方法与参数")
    reg_tut_method_options = ["Linear Regression", "Ridge", "Lasso", "SVR (支持向量回归)", "Random Forest Regressor"]
    st.selectbox(
        "选择回归算法:", options=reg_tut_method_options, key="reg_tut_method_select",
        help="选择要应用于上方数据的回归算法。"
    )
    current_method = st.session_state.reg_tut_method_select

    # --- 参数设置 (代码与上一版本相同，直接读取session state即可) ---
    if current_method == "Linear Regression":
        st.markdown("**算法说明:** 线性回归尝试找到一条直线（或超平面）来最佳拟合数据点。")
    elif current_method == "Ridge":
        st.slider("正则化强度 Alpha (α):", min_value=0.01, max_value=10.0, value=st.session_state.reg_tut_ridge_alpha, step=0.1, format="%.2f", key="reg_tut_ridge_alpha", help="控制L2正则化的强度。")
        st.markdown("**算法说明:**岭回归是线性回归的一种变体，增加了L2正则化项。")
    elif current_method == "Lasso":
        st.slider("正则化强度 Alpha (α):", min_value=0.01, max_value=10.0, value=st.session_state.reg_tut_lasso_alpha, step=0.1, format="%.2f", key="reg_tut_lasso_alpha", help="控制L1正则化的强度。")
        st.markdown("**算法说明:** Lasso 回归是线性回归的变体，增加了L1正则化项。")
    elif current_method == "SVR (支持向量回归)":
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1: st.slider("正则化强度 C:", min_value=0.1, max_value=10.0, value=st.session_state.reg_tut_svr_c, step=0.1, format="%.1f", key="reg_tut_svr_c", help="控制违反间隔边界的惩罚程度。")
        with col_p2: st.selectbox("核函数 (kernel):", options=['rbf', 'linear', 'poly'], key="reg_tut_svr_kernel", help="'rbf'和'poly'可以处理非线性关系。")
        with col_p3: st.slider("Epsilon (ε):", min_value=0.01, max_value=1.0, value=st.session_state.reg_tut_svr_epsilon, step=0.01, format="%.2f", key="reg_tut_svr_epsilon", help="定义间隔带的宽度。")
        st.markdown("**算法说明:** 支持向量回归试图找到一个函数，使得尽可能多的样本点落在间隔带内。")
    elif current_method == "Random Forest Regressor":
        col_p1, col_p2 = st.columns(2)
        with col_p1: st.slider("树的数量 (n_estimators):", min_value=10, max_value=200, value=st.session_state.reg_tut_rf_n_estimators, step=10, key="reg_tut_rf_n_estimators", help="森林中决策树的数量。")
        with col_p2: st.slider("树的最大深度 (max_depth, 0表示无限制):", min_value=0, max_value=20, value=st.session_state.reg_tut_rf_max_depth, step=1, key="reg_tut_rf_max_depth", help="限制单棵决策树的最大深度。0表示不限制。")
        st.markdown("**算法说明:** 随机森林回归器通过构建多棵决策树并对其预测结果进行平均来工作。")

    # --- 训练模型按钮 ---
    if st.button("🧠 训练并评估模型", key="reg_tut_run_training", help="使用当前选择的算法和参数训练回归模型"):
        # ... (训练模型的逻辑保持不变，读取 state) ...
        if st.session_state.reg_tut_X_train is None:
             st.error("请先生成数据集！")
        else:
            X_train_tut = st.session_state.reg_tut_X_train
            y_train_tut = st.session_state.reg_tut_y_train
            X_test_tut = st.session_state.reg_tut_X_test
            y_test_tut = st.session_state.reg_tut_y_test
            method_tut = st.session_state.reg_tut_method_select # Read selected method from state
            model_tut = None
            success_flag = False
            try:
                with st.spinner(f"正在训练 {method_tut}..."):
                    if method_tut == "Linear Regression": model_tut = LinearRegression()
                    elif method_tut == "Ridge": model_tut = Ridge(alpha=st.session_state.reg_tut_ridge_alpha)
                    elif method_tut == "Lasso": model_tut = Lasso(alpha=st.session_state.reg_tut_lasso_alpha)
                    elif method_tut == "SVR (支持向量回归)": model_tut = SVR(C=st.session_state.reg_tut_svr_c, kernel=st.session_state.reg_tut_svr_kernel, epsilon=st.session_state.reg_tut_svr_epsilon)
                    elif method_tut == "Random Forest Regressor":
                        rf_depth_state = st.session_state.reg_tut_rf_max_depth
                        max_depth_param = None if rf_depth_state == 0 else rf_depth_state
                        model_tut = RandomForestRegressor(n_estimators=st.session_state.reg_tut_rf_n_estimators, max_depth=max_depth_param, random_state=42, n_jobs=-1)

                    if model_tut:
                         model_tut.fit(X_train_tut, y_train_tut)
                         st.session_state.reg_tut_model = model_tut
                         st.session_state.reg_tut_y_pred = model_tut.predict(X_test_tut)
                         success_flag = True
                if success_flag: st.success(f"{method_tut} 模型训练完成！请查看下方评估结果。")
                else: st.error("未能初始化所选模型。")
            except Exception as train_e:
                st.error(f"训练 {method_tut} 出错: {train_e}")
                print(traceback.format_exc())
                st.session_state.reg_tut_model = None
                st.session_state.reg_tut_y_pred = None

    st.markdown("---")

    # --- 3. 显示评估结果 ---
    if st.session_state.reg_tut_model is not None and st.session_state.reg_tut_y_pred is not None:
        st.subheader("3. 模型评估结果（基于测试集）")

        y_test = st.session_state.reg_tut_y_test
        y_pred = st.session_state.reg_tut_y_pred
        X_test = st.session_state.reg_tut_X_test

        # --- 显示评估指标 ---
        try:
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1: st.metric("R² 分数 (R-squared)", f"{r2:.3f}", help="解释目标变量方差的比例。越接近1越好。")
            with col_m2: st.metric("均方误差 (MSE)", f"{mse:.3f}", help="预测误差平方的平均值。越小越好。")
            with col_m3: st.metric("平均绝对误差 (MAE)", f"{mae:.3f}", help="预测误差绝对值的平均值。越小越好。")
        except Exception as metric_e:
            st.error(f"计算评估指标时出错: {metric_e}")

        # --- 显示可视化结果 (使用本文件内定义的绘图函数) ---
        col_viz1, col_viz2 = st.columns(2)
        with col_viz1:
            st.markdown("##### 预测值 vs. 真实值")
            st.markdown("理想情况下，点应聚集在对角线附近。")
            try:
                fig_pred = plot_validation_results(y_test, y_pred, model_name=st.session_state.reg_tut_method_select)
                st.pyplot(fig_pred)
                plt.close(fig_pred)
            except Exception as pred_err:
                 st.warning(f"绘制预测对比图时出错: {pred_err}")
                 print(traceback.format_exc())
        with col_viz2:
            st.markdown("##### 残差图 (真实值 - 预测值)")
            st.markdown("理想情况下，残差应随机分布在0线周围。")
            try:
                 fig_res = plot_residuals(y_test, y_pred, model_name=st.session_state.reg_tut_method_select)
                 st.pyplot(fig_res)
                 plt.close(fig_res)
            except Exception as res_err:
                 st.warning(f"绘制残差图时出错: {res_err}")
                 print(traceback.format_exc())

        # --- 结果解读 ---
        st.markdown("#### 结果解读提示")
        st.markdown("- **R² 分数:** 接近 1 表示模型拟合得很好。接近 0 表示效果类似预测平均值。负数表示效果很差。")
        st.markdown("- **MSE / MAE:** 值越小表示预测越接近真实值。MSE 对大误差更敏感。")
        st.markdown("- **预测值 vs. 真实值图:** 点越靠近对角线（y=x），说明预测越准确。")
        st.markdown("- **残差图:** 点随机分布在 0 线上下是好的迹象。如果呈现模式（曲线、喇叭形），可能表示模型有问题。")

# --- 允许直接运行此脚本进行测试 ---
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="回归教学演示（独立运行）")
    st.sidebar.info("这是回归教学模块的独立测试运行。")
    show_regression_tutorial_page()