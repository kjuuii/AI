import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from datetime import datetime
import base64
from io import BytesIO
import matplotlib.font_manager as fm
import platform
import joblib

# For regression algorithms
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler


# Font setup for visualization (reusing from classification_validation)
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


# Helper functions for visualization
def apply_plot_style(ax):
    """应用统一的绘图样式"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.7)
    return ax


def create_figure_with_safe_dimensions(width_inches, height_inches, max_dpi=80):
    """创建不会超出Matplotlib限制的图形尺寸"""
    # 确保尺寸不会超过2^16限制
    max_pixels = 65000  # 略低于2^16

    # 计算保持尺寸在限制内的DPI
    width_dpi = max_pixels / width_inches
    height_dpi = max_pixels / height_inches

    # 使用较小的计算DPI值以确保两个维度都是安全的
    safe_dpi = min(width_dpi, height_dpi, max_dpi)

    # 使用安全DPI创建图形
    fig, ax = plt.subplots(figsize=(width_inches, height_inches), dpi=safe_dpi)
    return fig, ax


# Plot functions for regression validation results
def plot_predictions_vs_actual(y_true, y_pred, model_name="Model"):
    """绘制预测值与真实值对比图"""
    fig, ax = create_figure_with_safe_dimensions(10, 6)
    apply_plot_style(ax)

    try:
        # 创建索引
        indices = np.arange(len(y_true))

        # 绘制真实值和预测值
        ax.plot(indices, y_true,
                color='#2ecc71', label='真实值',
                linewidth=1.5, marker='o', markersize=3, alpha=0.7)
        ax.plot(indices, y_pred,
                color='#e74c3c', label=f'{model_name}预测值',
                linewidth=1.5, marker='x', markersize=4, alpha=0.7)

        font_kwargs = {'fontproperties': FONT_PROP} if 'FONT_PROP' in globals() else {}
        ax.set_xlabel('样本索引', fontsize=10, fontweight='bold', **font_kwargs)
        ax.set_ylabel('值', fontsize=10, fontweight='bold', **font_kwargs)
        ax.set_title('预测值 vs 真实值', fontsize=12, fontweight='bold', **font_kwargs)
        ax.legend(frameon=True, framealpha=0.9, edgecolor='#3498db',
                  prop=FONT_PROP if 'FONT_PROP' in globals() else None, fontsize=9)

        plt.tight_layout()
        return fig

    except Exception as e:
        print(f"绘制预测对比图时出错: {e}")
        ax.text(0.5, 0.5, f'绘制预测对比图时出错: {e}',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12, color='red',
                fontproperties=FONT_PROP if 'FONT_PROP' in globals() else None)
        return fig


def plot_residuals(y_true, y_pred, model_name="Model"):
    """绘制残差图"""
    fig, ax = create_figure_with_safe_dimensions(10, 6)
    apply_plot_style(ax)

    try:
        # 计算残差
        residuals = y_true - y_pred
        indices = np.arange(len(y_true))

        # 绘制残差
        ax.scatter(indices, residuals, alpha=0.6, color='#3498db', s=20)
        ax.axhline(y=0, color='#e74c3c', linestyle='--', linewidth=1.5)

        # 添加残差的标准差线
        std_residuals = np.std(residuals)
        ax.axhline(y=std_residuals, color='#f39c12', linestyle=':', alpha=0.7, label=f'±1σ')
        ax.axhline(y=-std_residuals, color='#f39c12', linestyle=':', alpha=0.7)

        font_kwargs = {'fontproperties': FONT_PROP} if 'FONT_PROP' in globals() else {}
        ax.set_xlabel('样本索引', fontsize=10, fontweight='bold', **font_kwargs)
        ax.set_ylabel('残差', fontsize=10, fontweight='bold', **font_kwargs)
        ax.set_title(f'{model_name}残差分析', fontsize=12, fontweight='bold', **font_kwargs)
        ax.legend(frameon=True, framealpha=0.9, edgecolor='#3498db',
                  prop=FONT_PROP if 'FONT_PROP' in globals() else None, fontsize=9)

        plt.tight_layout()
        return fig

    except Exception as e:
        print(f"绘制残差图时出错: {e}")
        ax.text(0.5, 0.5, f'绘制残差图时出错: {e}',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12, color='red',
                fontproperties=FONT_PROP if 'FONT_PROP' in globals() else None)
        return fig


def plot_scatter_actual_vs_pred(y_true, y_pred, model_name="Model"):
    """绘制真实值vs预测值散点图"""
    fig, ax = create_figure_with_safe_dimensions(8, 8)
    apply_plot_style(ax)

    try:
        # 绘制散点图
        ax.scatter(y_true, y_pred, alpha=0.5, color='#3498db', s=20)

        # 添加理想预测线（45度线）
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val],
                'r--', linewidth=2, label='理想预测线')

        # 计算R²
        r2 = r2_score(y_true, y_pred)

        font_kwargs = {'fontproperties': FONT_PROP} if 'FONT_PROP' in globals() else {}
        ax.set_xlabel('真实值', fontsize=10, fontweight='bold', **font_kwargs)
        ax.set_ylabel('预测值', fontsize=10, fontweight='bold', **font_kwargs)
        ax.set_title(f'{model_name} - 真实值 vs 预测值 (R²={r2:.4f})',
                     fontsize=12, fontweight='bold', **font_kwargs)
        ax.legend(frameon=True, framealpha=0.9, edgecolor='#3498db',
                  prop=FONT_PROP if 'FONT_PROP' in globals() else None, fontsize=9)

        plt.tight_layout()
        return fig

    except Exception as e:
        print(f"绘制散点图时出错: {e}")
        ax.text(0.5, 0.5, f'绘制散点图时出错: {e}',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12, color='red',
                fontproperties=FONT_PROP if 'FONT_PROP' in globals() else None)
        return fig


def plot_model_metrics(metrics, model_name="Model"):
    """绘制模型指标柱形图"""
    fig, ax = create_figure_with_safe_dimensions(10, 6)
    apply_plot_style(ax)

    try:
        # 准备指标数据
        labels = ['MSE', 'MAE', 'RMSE', 'R²']
        values = [
            metrics.get('mse', 0),
            metrics.get('mae', 0),
            np.sqrt(metrics.get('mse', 0)),
            metrics.get('r2', 0)
        ]

        x = np.arange(len(labels))
        bars = ax.bar(x, values, color=['#e74c3c', '#3498db', '#9b59b6', '#2ecc71'], alpha=0.7)

        # 应用字体到标题和标签
        font_kwargs = {'fontproperties': FONT_PROP} if 'FONT_PROP' in globals() else {}
        ax.set_title(f'{model_name}性能指标', fontsize=12, fontweight='bold', **font_kwargs)
        ax.set_ylabel('值', fontsize=10, **font_kwargs)

        # 设置刻度和标签
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, **font_kwargs)

        # 添加值标签
        for i, v in enumerate(values):
            ax.text(i, v + 0.01 * max(values), f'{v:.4f}', ha='center', fontsize=9)

        plt.tight_layout()
        return fig

    except Exception as e:
        print(f"绘制模型指标时出错: {e}")
        ax.text(0.5, 0.5, f'绘制模型指标时出错: {e}',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12, color='red',
                fontproperties=FONT_PROP if 'FONT_PROP' in globals() else None)
        return fig


def plot_multi_model_metrics(metrics_list, model_names=None):
    """绘制多模型指标对比柱形图"""
    fig, ax = create_figure_with_safe_dimensions(10, 6)
    apply_plot_style(ax)

    try:
        metrics_labels = ['MSE', 'MAE', 'RMSE', 'R²']

        # 获取每个模型的指标
        all_metrics = []
        for i, metrics in enumerate(metrics_list):
            model_metrics = [
                metrics.get('mse', 0),
                metrics.get('mae', 0),
                np.sqrt(metrics.get('mse', 0)),
                metrics.get('r2', 0)
            ]
            all_metrics.append(model_metrics)

        # 设置x轴位置
        x = np.arange(len(metrics_labels))
        width = 0.8 / len(metrics_list)  # 根据模型数量调整柱形宽度

        # 为每个模型设置不同颜色
        colors = ['#e74c3c', '#3498db', '#9b59b6']

        # 绘制每个模型的指标柱形图
        for i, model_metrics in enumerate(all_metrics):
            model_name = model_names[i] if model_names and i < len(model_names) else f"模型 {i + 1}"
            offset = (i - len(metrics_list) / 2 + 0.5) * width
            bars = ax.bar(x + offset, model_metrics,
                          width, label=model_name,
                          color=colors[i % len(colors)], alpha=0.7)

            # 在柱形上方添加值标签
            for bar, value in zip(bars, model_metrics):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                        f'{value:.3f}', ha='center', fontsize=7)

        # 应用字体到标题和标签
        font_kwargs = {'fontproperties': FONT_PROP} if 'FONT_PROP' in globals() else {}
        ax.set_title('多模型性能指标对比', fontsize=12, fontweight='bold', **font_kwargs)
        ax.set_ylabel('值', fontsize=10, **font_kwargs)

        # 设置刻度和标签
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_labels, rotation=0, **font_kwargs)

        # 创建图例
        legend = ax.legend(loc='upper right', fontsize=8, prop=FONT_PROP)

        plt.tight_layout()
        return fig

    except Exception as e:
        print(f"绘制多模型指标时出错: {e}")
        ax.text(0.5, 0.5, f'绘制多模型指标时出错: {e}',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12, color='red',
                fontproperties=FONT_PROP if 'FONT_PROP' in globals() else None)
        return fig


def plot_multi_model_predictions(y_true, all_predictions, model_names=None):
    """绘制多模型预测对比图"""
    fig, ax = create_figure_with_safe_dimensions(12, 6)
    apply_plot_style(ax)

    try:
        indices = np.arange(len(y_true))

        # 绘制真实值
        ax.plot(indices, y_true,
                color='#2ecc71', label='真实值',
                linewidth=2, marker='o', markersize=3, alpha=0.8)

        # 颜色和标记样式
        colors = ['#e74c3c', '#3498db', '#9b59b6']
        markers = ['x', 's', '^']

        # 绘制每个模型的预测
        for i, predictions in enumerate(all_predictions):
            model_name = model_names[i] if model_names and i < len(model_names) else f"模型 {i + 1}"
            ax.plot(indices, predictions,
                    color=colors[i % len(colors)],
                    label=f'{model_name}预测值',
                    linewidth=1.5,
                    linestyle='--',
                    marker=markers[i % len(markers)],
                    markersize=4,
                    alpha=0.7)

        font_kwargs = {'fontproperties': FONT_PROP} if 'FONT_PROP' in globals() else {}
        ax.set_xlabel('样本索引', fontsize=10, fontweight='bold', **font_kwargs)
        ax.set_ylabel('值', fontsize=10, fontweight='bold', **font_kwargs)
        ax.set_title('多模型预测对比', fontsize=12, fontweight='bold', **font_kwargs)
        ax.legend(frameon=True, framealpha=0.9, edgecolor='#3498db',
                  prop=FONT_PROP if 'FONT_PROP' in globals() else None, fontsize=9)

        plt.tight_layout()
        return fig

    except Exception as e:
        print(f"绘制多模型预测时出错: {e}")
        ax.text(0.5, 0.5, f'绘制多模型预测时出错: {e}',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12, color='red',
                fontproperties=FONT_PROP if 'FONT_PROP' in globals() else None)
        return fig


# 处理文件夹数据（虽然回归任务可能不太需要，但保持一致性）
def process_folder_data(folder_path, progress_callback=None):
    """处理文件夹数据用于回归任务"""
    try:
        # 获取所有CSV/Excel文件
        all_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.csv', '.xlsx', '.xls')):
                    all_files.append(os.path.join(root, file))

        if not all_files:
            return None, "所选文件夹不包含CSV或Excel文件。"

        # 初始化数据收集
        all_data = []

        # 处理每个文件
        file_count = len(all_files)
        for i, file_path in enumerate(all_files):
            try:
                # 加载数据
                if file_path.lower().endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)

                # 跳过空文件
                if df.empty:
                    continue

                # 处理潜在的混合类型列或空值
                for col in df.select_dtypes(include=['object']).columns:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='ignore')
                    except Exception:
                        pass

                # 删除全为NaN的列
                df.dropna(axis=1, how='all', inplace=True)

                # 删除包含任何NaN的行
                df.dropna(inplace=True)

                # 清理后如果为空则跳过
                if df.empty:
                    continue

                # 确保所有数据都是数值型
                numeric_df = df.select_dtypes(include=['number'])

                # 如果没有数值列，则跳过
                if numeric_df.empty:
                    continue

                # 添加到数据集合
                all_data.append(numeric_df)

            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")

            # 更新进度
            if progress_callback:
                progress_percent = int((i + 1) / file_count * 100)
                progress_callback(progress_percent)

        # 合并所有数据
        if not all_data:
            return None, "未找到有效的数据文件或所有文件处理失败"

        # 确保所有数据框有相同的列
        common_columns = set.intersection(*[set(df.columns) for df in all_data])
        if not common_columns:
            return None, "文件之间没有公共的数值列，无法合并数据"

        # 过滤到公共列并连接
        all_data = [df[list(common_columns)] for df in all_data]
        combined_data = pd.concat(all_data, ignore_index=True)

        return combined_data, None

    except Exception as e:
        import traceback
        return None, f"处理文件夹时出错: {str(e)}\n{traceback.format_exc()}"


# 验证单个模型
def validate_model(X, y, model_path, use_sliding_window=False, window_size=3,
                   use_column_features=True, use_raw_columns=False, progress_callback=None):
    """验证单个回归模型并返回结果"""
    try:
        # 初始化进度
        if progress_callback:
            progress_callback(10)

        # 加载模型
        try:
            model_info = joblib.load(model_path)
            model = model_info.get('model')
            if model is None:
                raise ValueError("模型对象为空")

            # 获取模型信息
            model_type = model_info.get('model_type', 'unknown')
            feature_names = model_info.get('feature_names', [])
            model_feature_names = model_info.get('model_feature_names', feature_names)
            scaler = model_info.get('scaler')
            model_params = model_info.get('best_params', {})

            if progress_callback:
                progress_callback(30)

            print(f"加载的模型类型: {model_type}")

        except Exception as e:
            raise ValueError(f"加载模型时出错: {str(e)}")

        # 确保特征顺序正确
        if model_feature_names and set(model_feature_names).issubset(set(X.columns)):
            X = X[model_feature_names]
        elif feature_names and set(feature_names).issubset(set(X.columns)):
            X = X[feature_names]
        else:
            print(f"警告: 特征名称不匹配。模型期望: {model_feature_names or feature_names}, 数据提供: {list(X.columns)}")

        # 数据预处理
        X_processed = X.copy()

        # 应用滑动窗口（如果需要）
        if use_sliding_window and window_size > 0:
            X_windowed = create_sliding_window_data(X_processed, window_size)
            if X_windowed is not None and not X_windowed.empty:
                # 对齐y值
                y = y.iloc[window_size:]
                X_processed = X_windowed

        # 标准化（如果训练时使用了）
        if scaler is not None:
            X_processed = pd.DataFrame(
                scaler.transform(X_processed),
                columns=X_processed.columns,
                index=X_processed.index
            )

        if progress_callback:
            progress_callback(50)

        # 进行预测
        y_pred = model.predict(X_processed)

        if progress_callback:
            progress_callback(70)

        # 确保y和y_pred长度一致
        min_len = min(len(y), len(y_pred))
        y = y[:min_len]
        y_pred = y_pred[:min_len]

        # 计算评估指标
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mse)

        # 获取特征重要性（如果模型支持）
        feature_importance = get_feature_importance(model, X_processed.columns.tolist(), model_type)

        if progress_callback:
            progress_callback(90)

        # 准备结果
        results = {
            'model_info': {
                'model_path': model_path,
                'model_type': model_type,
                'feature_names': feature_names,
                'model_feature_names': model_feature_names,
                'params': model_params,
                'use_sliding_window': use_sliding_window,
                'window_size': window_size,
                'use_column_features': use_column_features,
                'use_raw_columns': use_raw_columns
            },
            'predictions': y_pred,
            'metrics': {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': rmse
            },
            'feature_importance': feature_importance,
            'y': y
        }

        if progress_callback:
            progress_callback(100)

        return results, None

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"验证模型时出错: {e}\n{error_details}")
        return None, f"验证模型时出错: {str(e)}"


def create_sliding_window_data(X, window_size):
    """创建滑动窗口数据"""
    if not isinstance(X, pd.DataFrame):
        return None

    X_windowed = X.copy()

    # 添加滞后特征
    for col in X.columns:
        for i in range(1, window_size + 1):
            lag_name = f"{col}_lag_{i}"
            X_windowed[lag_name] = X[col].shift(i)

    # 删除包含NaN的行
    X_windowed = X_windowed.dropna()

    return X_windowed


def get_feature_importance(model, feature_names, model_type):
    """获取特征重要性（根据模型类型）"""
    feature_importance = {}

    try:
        if hasattr(model, 'feature_importances_'):
            # 树模型（随机森林、GBDT、CatBoost等）
            importance = model.feature_importances_
            if len(feature_names) == len(importance):
                feature_importance = dict(zip(feature_names, importance))

        elif hasattr(model, 'coef_'):
            # 线性模型
            importance = np.abs(model.coef_)
            if len(feature_names) == len(importance):
                feature_importance = dict(zip(feature_names, importance))

    except Exception as e:
        print(f"获取特征重要性时出错: {e}")

    return feature_importance


# 验证多个模型
def validate_multiple_models(X, y, model_paths, use_sliding_window=False, window_size=3,
                             use_column_features=True, use_raw_columns=False, progress_callback=None):
    """验证多个回归模型并返回比较结果"""
    try:
        # 初始化结果
        all_results = {
            'model_infos': [],
            'predictions': [],
            'metrics': [],
            'feature_importances': [],
            'model_names': [],
            'model_paths': model_paths,
            'y': y
        }

        # 为每个模型计算进度
        model_count = len(model_paths)
        if model_count == 0:
            raise ValueError("没有提供模型路径")

        progress_per_model = 90 / model_count

        # 处理每个模型
        for i, model_path in enumerate(model_paths):
            if progress_callback:
                progress_callback(int(10 + i * progress_per_model))

            # 提取模型名称
            model_name = os.path.basename(model_path).replace('.joblib', '')

            # 加载和验证单个模型
            single_results, error = validate_model(
                X, y, model_path,
                use_sliding_window, window_size,
                use_column_features, use_raw_columns
            )

            if error:
                raise ValueError(f"验证模型 {model_name} 时出错: {error}")

            # 从单个结果中提取所需信息
            model_info = single_results['model_info']
            predictions = single_results['predictions']
            metrics = single_results['metrics']
            feature_importance = single_results['feature_importance']

            # 添加到总结果
            all_results['model_infos'].append(model_info)
            all_results['predictions'].append(predictions)
            all_results['metrics'].append(metrics)
            all_results['feature_importances'].append(feature_importance)
            all_results['model_names'].append(model_name)

            if progress_callback:
                progress_callback(int(10 + (i + 1) * progress_per_model))

        if progress_callback:
            progress_callback(100)

        return all_results, None

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"验证多个模型时出错: {e}\n{error_details}")
        return None, f"验证多个模型时出错: {str(e)}"


# 创建下载链接
def get_download_link(df, filename, text):
    """生成CSV下载链接"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href


# 导出单模型验证结果
def export_single_model_results(results, filename="regression_validation_results.xlsx"):
    """导出单模型验证结果到Excel"""
    try:
        y = results['y']
        pred = results['predictions']

        # 创建数据框
        df = pd.DataFrame({
            '样本索引': np.arange(len(y)),
            '真实值': y,
            '预测值': pred,
            '残差': y - pred,
            '相对误差(%)': np.abs((y - pred) / y) * 100
        })

        # 准备总体指标
        metrics = results['metrics']
        metrics_df = pd.DataFrame({
            '指标': ['MSE', 'MAE', 'RMSE', 'R²'],
            '值': [
                metrics.get('mse', 'N/A'),
                metrics.get('mae', 'N/A'),
                metrics.get('rmse', 'N/A'),
                metrics.get('r2', 'N/A')
            ]
        })

        # 保存到内存中的Excel
        buffer = BytesIO()

        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            # 写入元数据
            info_df = pd.DataFrame({
                '信息': [
                    '回归模型验证结果',
                    f'验证时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                    f'模型路径: {results["model_info"].get("model_path", "N/A")}'
                ],
                '值': ['', '', '']
            })
            info_df.to_excel(writer, sheet_name='概览', index=False)

            # 写入总体指标
            metrics_df.to_excel(writer, sheet_name='总体指标', index=False)

            # 写入样本预测结果
            df.to_excel(writer, sheet_name='样本预测结果', index=False)

        # 将buffer转为二进制以下载
        buffer.seek(0)
        excel_data = buffer.getvalue()

        b64 = base64.b64encode(excel_data).decode()
        return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">下载验证结果</a>'

    except Exception as e:
        print(f"导出结果时出错: {e}")
        return None


# 导出多模型验证结果
def export_multi_model_results(results, filename="multi_model_validation_results.xlsx"):
    """导出多模型验证结果到Excel"""
    try:
        y = results['y']
        all_predictions = results['predictions']
        model_names = results['model_names']
        all_metrics = results['metrics']

        # 创建样本预测数据框
        data = {'样本索引': np.arange(len(y)), '真实值': y}

        # 添加每个模型的预测
        for i, (pred, name) in enumerate(zip(all_predictions, model_names)):
            data[f'{name}预测值'] = pred
            data[f'{name}残差'] = y - pred

        df = pd.DataFrame(data)

        # 创建总体指标数据框
        metrics_data = {'指标': ['MSE', 'MAE', 'RMSE', 'R²']}

        for i, (metrics, name) in enumerate(zip(all_metrics, model_names)):
            metrics_data[name] = [
                metrics.get('mse', 'N/A'),
                metrics.get('mae', 'N/A'),
                metrics.get('rmse', 'N/A'),
                metrics.get('r2', 'N/A')
            ]

        metrics_df = pd.DataFrame(metrics_data)

        # 保存到内存中的Excel
        buffer = BytesIO()

        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            # 写入元数据
            info_data = [['多模型回归验证结果'], [f'验证时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}']]
            for i, name in enumerate(model_names):
                model_path = results["model_paths"][i] if i < len(results["model_paths"]) else "N/A"
                info_data.append([f'模型{i + 1}: {name}, 路径: {model_path}'])

            info_df = pd.DataFrame(info_data, columns=['信息'])
            info_df.to_excel(writer, sheet_name='概览', index=False)

            # 写入总体指标
            metrics_df.to_excel(writer, sheet_name='总体指标', index=False)

            # 写入样本预测结果
            df.to_excel(writer, sheet_name='样本预测结果', index=False)

        # 将buffer转为二进制以下载
        buffer.seek(0)
        excel_data = buffer.getvalue()

        b64 = base64.b64encode(excel_data).decode()
        return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">下载比较结果</a>'

    except Exception as e:
        print(f"导出多模型结果时出错: {e}")
        return None


# 主界面函数
def show_regression_validation_page():
    """显示回归验证页面"""
    st.title("回归模型验证")

    # 初始化会话状态变量
    if 'validation_tab' not in st.session_state:
        st.session_state.validation_tab = 0

    if 'validation_mode' not in st.session_state:
        st.session_state.validation_mode = "single"  # "single" 或 "multi"

    if 'model_paths' not in st.session_state:
        st.session_state.model_paths = []

    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False

    if 'data' not in st.session_state:
        st.session_state.data = None

    if 'column_names' not in st.session_state:
        st.session_state.column_names = []

    if 'selected_input_columns' not in st.session_state:
        st.session_state.selected_input_columns = []

    if 'selected_output_column' not in st.session_state:
        st.session_state.selected_output_column = None

    if 'validation_results' not in st.session_state:
        st.session_state.validation_results = None

    if 'multi_validation_results' not in st.session_state:
        st.session_state.multi_validation_results = None

    if 'validation_progress' not in st.session_state:
        st.session_state.validation_progress = 0

    if 'data_source_type' not in st.session_state:
        st.session_state.data_source_type = "file"  # "file" 或 "folder"

    # 新增的临时变量，用于避免刷新问题
    if 'temp_selected_input_columns' not in st.session_state:
        st.session_state.temp_selected_input_columns = []

    if 'temp_selected_output_column' not in st.session_state:
        st.session_state.temp_selected_output_column = None

    if 'columns_selected_flag' not in st.session_state:
        st.session_state.columns_selected_flag = False

    # 创建选项卡
    tabs = st.tabs(["1. 验证模式", "2. 模型导入", "3. 数据导入", "4. 列选择", "5. 验证结果"])

    # 1. 验证模式选项卡
    with tabs[0]:
        create_validation_mode_section()

    # 2. 模型导入选项卡
    with tabs[1]:
        create_model_import_section()

    # 3. 数据导入选项卡
    with tabs[2]:
        create_data_import_section()

    # 4. 列选择选项卡
    with tabs[3]:
        create_column_selection_section()

    # 5. 验证结果选项卡
    with tabs[4]:
        create_results_section()

    # 处理验证逻辑
    handle_validation()


def create_validation_mode_section():
    """创建验证模式选择部分"""
    st.header("验证模式选择")

    # 验证模式选择
    col1, col2 = st.columns(2)

    with col1:
        single_mode = st.button("单模型验证",
                                use_container_width=True,
                                type="primary" if st.session_state.validation_mode == "single" else "secondary")

        if single_mode:
            st.session_state.validation_mode = "single"
            # 清除现有结果
            st.session_state.validation_results = None
            st.session_state.multi_validation_results = None

    with col2:
        multi_mode = st.button("多模型比较",
                               use_container_width=True,
                               type="primary" if st.session_state.validation_mode == "multi" else "secondary")

        if multi_mode:
            st.session_state.validation_mode = "multi"
            # 清除现有结果
            st.session_state.validation_results = None
            st.session_state.multi_validation_results = None

    # 显示说明
    st.markdown("""
    **验证模式说明**

    - **单模型验证**: 验证一个回归模型在给定数据集上的性能
    - **多模型比较**: 比较多个回归模型在同一数据集上的性能（最多3个模型）
    """)


def create_model_import_section():
    """创建模型导入部分"""
    st.header("模型导入")

    # 根据验证模式显示不同UI
    if st.session_state.validation_mode == "single":
        create_single_model_import_enhanced()
    else:
        create_multi_model_import()


def create_single_model_import_enhanced():
    """创建单模型导入UI - 增强版"""
    uploaded_model = st.file_uploader("上传模型文件", type=["joblib"], key="single_model")

    if uploaded_model is not None:
        # 保存上传的模型文件
        with open(f"temp_model_{uploaded_model.name}", "wb") as f:
            f.write(uploaded_model.getvalue())

        model_path = f"temp_model_{uploaded_model.name}"
        st.session_state.model_paths = [model_path]

        # 显示成功消息
        st.success(f"已成功加载模型: {uploaded_model.name}")

        # 提供详细的模型信息
        try:
            model_info = joblib.load(model_path)
            model_type = model_info.get('model_type', '未知')
            feature_count = len(model_info.get('feature_names', []))

            # 显示模型信息
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("模型类型", model_type)
            with col2:
                st.metric("特征数量", feature_count)
            with col3:
                st.metric("输出变量", model_info.get('output_name', 'N/A'))

            # 显示模型参数
            if 'best_params' in model_info and model_info['best_params']:
                with st.expander("查看模型参数"):
                    params_df = pd.DataFrame(
                        list(model_info['best_params'].items()),
                        columns=['参数', '值']
                    )
                    st.dataframe(params_df, hide_index=True)

            # 显示特征名称
            if 'feature_names' in model_info and model_info['feature_names']:
                with st.expander("查看特征名称"):
                    st.write(", ".join(model_info['feature_names']))

        except Exception as e:
            st.warning(f"读取模型信息时出错: {str(e)}")

    elif len(st.session_state.model_paths) > 0 and os.path.exists(st.session_state.model_paths[0]):
        st.info(f"当前已加载模型: {os.path.basename(st.session_state.model_paths[0])}")


def create_multi_model_import():
    """创建多模型导入UI"""
    col1, col2, col3 = st.columns(3)

    # 最多可导入3个模型
    with col1:
        st.subheader("模型 1")
        uploaded_model1 = st.file_uploader("上传模型1", type=["joblib"], key="model1")

        if uploaded_model1 is not None:
            with open(f"temp_model1_{uploaded_model1.name}", "wb") as f:
                f.write(uploaded_model1.getvalue())

            model_path = f"temp_model1_{uploaded_model1.name}"

            # 更新模型路径列表
            model_paths = st.session_state.model_paths.copy()
            if len(model_paths) == 0:
                model_paths.append(model_path)
            else:
                model_paths[0] = model_path
            st.session_state.model_paths = model_paths

            st.success(f"已加载模型1: {uploaded_model1.name}")

    with col2:
        st.subheader("模型 2")
        uploaded_model2 = st.file_uploader("上传模型2", type=["joblib"], key="model2")

        if uploaded_model2 is not None:
            with open(f"temp_model2_{uploaded_model2.name}", "wb") as f:
                f.write(uploaded_model2.getvalue())

            model_path = f"temp_model2_{uploaded_model2.name}"

            # 更新模型路径列表
            model_paths = st.session_state.model_paths.copy()
            if len(model_paths) < 2:
                model_paths.append(model_path)
            else:
                model_paths[1] = model_path
            st.session_state.model_paths = model_paths

            st.success(f"已加载模型2: {uploaded_model2.name}")

    with col3:
        st.subheader("模型 3")
        uploaded_model3 = st.file_uploader("上传模型3", type=["joblib"], key="model3")

        if uploaded_model3 is not None:
            with open(f"temp_model3_{uploaded_model3.name}", "wb") as f:
                f.write(uploaded_model3.getvalue())

            model_path = f"temp_model3_{uploaded_model3.name}"

            # 更新模型路径列表
            model_paths = st.session_state.model_paths.copy()
            if len(model_paths) < 3:
                model_paths.append(model_path)
            else:
                model_paths[2] = model_path
            st.session_state.model_paths = model_paths

            st.success(f"已加载模型3: {uploaded_model3.name}")

    # 显示当前已加载的模型
    loaded_models = [path for path in st.session_state.model_paths if os.path.exists(path)]
    if loaded_models:
        st.info(f"当前已加载 {len(loaded_models)} 个模型")
        for i, path in enumerate(loaded_models):
            st.write(f"模型 {i + 1}: {os.path.basename(path)}")


def create_data_import_section():
    """创建数据导入部分"""
    st.header("数据导入")

    # 创建两列
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("文件导入")
        uploaded_file = st.file_uploader("上传CSV或Excel文件", type=["csv", "xlsx", "xls"], key="data_file")

        if uploaded_file is not None:
            try:
                # 处理上传的文件
                if uploaded_file.name.lower().endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                else:
                    data = pd.read_excel(uploaded_file)

                # 清理数据
                for col in data.select_dtypes(include=['object']).columns:
                    try:
                        data[col] = pd.to_numeric(data[col], errors='ignore')
                    except:
                        pass

                data.dropna(axis=1, how='all', inplace=True)

                if data.empty:
                    st.error("上传的文件为空或不包含有效数据。")
                else:
                    # 在会话状态中存储数据
                    st.session_state.data = data
                    st.session_state.column_names = list(data.columns)
                    st.session_state.data_source_type = "file"
                    st.session_state.data_loaded = True

                    # 成功消息
                    st.success(f"已成功加载: {uploaded_file.name} (包含 {len(data)} 行, {len(data.columns)} 列)")

                    # 清除之前的结果
                    st.session_state.validation_results = None
                    st.session_state.multi_validation_results = None

            except Exception as e:
                st.error(f"加载数据时出错: {str(e)}")

    with col2:
        st.subheader("文件夹导入")
        folder_path = st.text_input("输入文件夹路径", key="folder_path")

        if folder_path:
            if os.path.isdir(folder_path):
                if st.button("处理文件夹", key="process_folder"):
                    with st.spinner("正在处理文件夹..."):
                        progress_bar = st.progress(0)

                        def update_progress(p):
                            progress_bar.progress(p / 100)

                        results, error_msg = process_folder_data(folder_path, progress_callback=update_progress)

                        if results is not None:
                            # 在会话状态中存储数据
                            st.session_state.data = results
                            st.session_state.column_names = list(results.columns)
                            st.session_state.data_source_type = "folder"
                            st.session_state.data_loaded = True

                            # 成功消息
                            st.success(f"已成功加载文件夹数据: {len(results)} 行, {len(results.columns)} 列")

                            # 清除之前的结果
                            st.session_state.validation_results = None
                            st.session_state.multi_validation_results = None

                        else:
                            st.error(f"处理文件夹时出错: {error_msg}")
            else:
                st.error("请输入有效的文件夹路径")

    # 如果已加载数据，显示数据预览
    if st.session_state.data_loaded and st.session_state.data is not None:
        st.subheader("数据预览")
        st.dataframe(st.session_state.data.head())

        # 添加数据处理选项
        st.subheader("数据预处理选项")

        col1, col2 = st.columns(2)

        with col1:
            st.checkbox("使用滑动窗口", key="use_sliding_window")
            st.number_input("窗口大小", min_value=1, max_value=10, value=3, key="window_size",
                            disabled=not st.session_state.get("use_sliding_window", False))

        with col2:
            st.checkbox("使用列统计特征", value=True, key="use_column_features")
            st.checkbox("使用原始列数据", key="use_raw_columns")

        st.info("注: 滑动窗口适用于时间序列数据，将添加前几个时间点的特征。")


def create_column_selection_section():
    """创建列选择部分"""
    st.header("列选择")

    if not st.session_state.data_loaded:
        st.info("请先导入数据")
        return

    # 初始化临时变量
    if 'temp_selected_input_columns' not in st.session_state:
        st.session_state.temp_selected_input_columns = st.session_state.selected_input_columns.copy()

    if 'temp_selected_output_column' not in st.session_state:
        st.session_state.temp_selected_output_column = st.session_state.selected_output_column

    if 'columns_selected_flag' not in st.session_state:
        st.session_state.columns_selected_flag = False

    # 创建三列布局
    col1, col2, col3 = st.columns([3, 3, 2])

    with col1:
        st.subheader("可用列")
        if st.session_state.column_names:
            # 过滤掉已在临时选择中的列
            available_columns = [col for col in st.session_state.column_names
                                 if col not in st.session_state.temp_selected_input_columns
                                 and col != st.session_state.temp_selected_output_column]

            selected_available = st.multiselect("选择列", available_columns, key="available_cols")

            # 添加按钮，但不立即更新session_state
            if selected_available and st.button("添加为输入特征", key="add_input"):
                new_inputs = st.session_state.temp_selected_input_columns.copy()
                new_inputs.extend(selected_available)
                st.session_state.temp_selected_input_columns = new_inputs

            # 设为输出标签的按钮
            if len(selected_available) == 1 and st.button("设为输出目标", key="set_output"):
                st.session_state.temp_selected_output_column = selected_available[0]
        else:
            st.info("没有可用列")

    with col2:
        st.subheader("输入特征")
        if st.session_state.temp_selected_input_columns:
            selected_inputs = st.multiselect("已选特征", st.session_state.temp_selected_input_columns,
                                             default=st.session_state.temp_selected_input_columns,
                                             key="selected_inputs")

            # 如果有变化，更新临时选择
            if selected_inputs != st.session_state.temp_selected_input_columns:
                st.session_state.temp_selected_input_columns = selected_inputs

            # 移除选中的输入特征
            if st.button("移除选中的输入特征", key="remove_input"):
                st.session_state.temp_selected_input_columns = []
        else:
            st.info("未选择输入特征")

    with col3:
        st.subheader("输出目标")
        if st.session_state.temp_selected_output_column:
            st.info(f"已选择: {st.session_state.temp_selected_output_column}")

            # 清除输出列按钮
            if st.button("清除输出目标", key="clear_output"):
                st.session_state.temp_selected_output_column = None
        else:
            st.info("未选择输出目标")

    # 添加确认按钮区域
    st.markdown("---")
    st.subheader("确认选择")

    # 确认按钮
    if st.button("✅ 确认列选择", key="confirm_columns", use_container_width=True):
        # 只有在确认时才将临时选择更新到正式的session_state变量
        st.session_state.selected_input_columns = st.session_state.temp_selected_input_columns.copy()
        st.session_state.selected_output_column = st.session_state.temp_selected_output_column
        st.session_state.columns_selected_flag = True
        st.success("列选择已确认！")
        time.sleep(0.5)  # 短暂延迟以显示成功消息

    # 验证按钮区域
    st.markdown("---")
    st.subheader("开始验证")

    # 检查是否可以执行验证
    can_validate = (st.session_state.data_loaded and
                    len(st.session_state.model_paths) > 0 and
                    len(st.session_state.selected_input_columns) > 0 and
                    st.session_state.selected_output_column is not None)

    validate_col1, validate_col2 = st.columns([3, 1])

    with validate_col1:
        if st.button("验证模型", type="primary", disabled=not can_validate, key="validate_button"):
            st.session_state.is_validating = True
            st.session_state.validation_tab = 4
            st.rerun()

    with validate_col2:
        # 显示验证进度
        st.progress(st.session_state.validation_progress / 100)


def create_results_section():
    """创建结果展示部分"""
    st.header("验证结果")

    # 添加立即强制显示的按钮
    force_display = st.button("显示验证结果", key="force_display_results")

    # 显示结果的状态
    if force_display:
        if st.session_state.validation_mode == "single":
            if st.session_state.validation_results is not None:
                display_single_validation_results()
            else:
                st.warning("没有找到单模型验证结果")
        else:
            if st.session_state.multi_validation_results is not None:
                display_multi_validation_results()
            else:
                st.warning("没有找到多模型验证结果")

    # 如果正在验证，显示进度条
    if st.session_state.get("is_validating", False):
        with st.spinner("正在验证模型..."):
            progress_bar = st.progress(0)

            try:
                # 准备验证数据
                X = st.session_state.data[st.session_state.selected_input_columns].copy()
                y = st.session_state.data[st.session_state.selected_output_column].copy()

                # 处理可能的NaN值
                if X.isnull().values.any() or y.isnull().values.any():
                    combined = pd.concat([X, y], axis=1)
                    combined.dropna(inplace=True)

                    if combined.empty:
                        st.error("移除缺失值后数据为空，无法验证。")
                        st.session_state.is_validating = False
                        return

                    X = combined[X.columns]
                    y = combined[st.session_state.selected_output_column]

                # 执行验证
                use_sliding_window = st.session_state.get("use_sliding_window", False)
                window_size = st.session_state.get("window_size", 3)
                use_column_features = st.session_state.get("use_column_features", True)
                use_raw_columns = st.session_state.get("use_raw_columns", False)

                def update_progress(p):
                    progress_bar.progress(p / 100)
                    st.session_state.validation_progress = p

                if st.session_state.validation_mode == "single":
                    # 单模型验证
                    results, error = validate_model(
                        X, y, st.session_state.model_paths[0],
                        use_sliding_window, window_size,
                        use_column_features, use_raw_columns,
                        progress_callback=update_progress
                    )

                    if error:
                        st.error(f"验证失败: {error}")
                    else:
                        # 保存结果到会话状态
                        st.session_state["validation_results"] = results
                        st.session_state["validation_completed"] = True
                        st.success("单模型验证完成！")
                else:
                    # 多模型验证
                    results, error = validate_multiple_models(
                        X, y, st.session_state.model_paths,
                        use_sliding_window, window_size,
                        use_column_features, use_raw_columns,
                        progress_callback=update_progress
                    )

                    if error:
                        st.error(f"多模型验证失败: {error}")
                    else:
                        # 保存结果到会话状态
                        st.session_state["multi_validation_results"] = results
                        st.session_state["validation_completed"] = True
                        st.success("多模型验证完成！")

            except Exception as e:
                import traceback
                st.error(f"验证过程中发生错误: {str(e)}")
                st.code(traceback.format_exc())

            finally:
                # 无论如何都要重置验证状态
                st.session_state.is_validating = False

    # 显示验证结果
    if not st.session_state.get("is_validating", False):
        has_results = False

        if st.session_state.validation_mode == "single" and st.session_state.get("validation_results") is not None:
            has_results = True
            st.success("显示单模型验证结果")
            display_single_validation_results()
        elif st.session_state.validation_mode == "multi" and st.session_state.get(
                "multi_validation_results") is not None:
            has_results = True
            st.success("显示多模型验证结果")
            display_multi_validation_results()
        else:
            st.info("请先进行验证或导入验证结果")

            # 如果验证已完成但没有结果，提供一个重试按钮
            if st.session_state.get("validation_completed", False):
                if st.button("重新验证", key="retry_validation"):
                    st.session_state.is_validating = True
                    st.rerun()


def display_single_validation_results():
    """显示单模型验证结果"""
    if st.session_state.validation_results is None:
        st.info("请先进行验证或导入验证结果")
        return

    results = st.session_state.validation_results

    # 创建选项卡显示不同类型的结果
    result_tabs = st.tabs(["性能指标", "预测对比", "残差分析", "散点图"])

    with result_tabs[0]:
        st.subheader("验证性能指标")

        metrics = results['metrics']

        # 使用列布局显示指标
        metric_cols = st.columns(4)

        with metric_cols[0]:
            st.metric("MSE", f"{metrics['mse']:.4f}")

        with metric_cols[1]:
            st.metric("MAE", f"{metrics['mae']:.4f}")

        with metric_cols[2]:
            st.metric("RMSE", f"{metrics['rmse']:.4f}")

        with metric_cols[3]:
            r2_value = metrics['r2']
            # 根据R²值显示不同颜色
            if r2_value > 0.8:
                delta_color = "normal"
            elif r2_value > 0.6:
                delta_color = "normal"
            else:
                delta_color = "inverse"
            st.metric("R²", f"{r2_value:.4f}")

        # 显示模型信息
        st.subheader("模型信息")

        # 提取模型信息
        model_info = results['model_info']
        model_type = model_info.get('model_type', '未知')
        feature_names = model_info.get('feature_names', [])

        # 显示信息表格
        info_data = {
            "参数": ["模型类型", "特征数量", "使用滑动窗口", "窗口大小", "使用列特征", "使用原始列"],
            "值": [
                model_type,
                len(feature_names),
                "是" if model_info.get('use_sliding_window', False) else "否",
                model_info.get('window_size', 'N/A'),
                "是" if model_info.get('use_column_features', True) else "否",
                "是" if model_info.get('use_raw_columns', False) else "否"
            ]
        }

        st.dataframe(pd.DataFrame(info_data), use_container_width=True)

        # 导出结果按钮
        export_link = export_single_model_results(results, "regression_validation_results.xlsx")
        if export_link:
            st.markdown(export_link, unsafe_allow_html=True)

    with result_tabs[1]:
        st.subheader("预测值对比")

        # 绘制预测对比图
        y_true = results['y']
        y_pred = results['predictions']
        model_name = os.path.basename(results['model_info'].get('model_path', '')).replace('.joblib', '')

        fig = plot_predictions_vs_actual(y_true, y_pred, model_name)
        st.pyplot(fig)

    with result_tabs[2]:
        st.subheader("残差分析")

        # 绘制残差图
        fig = plot_residuals(y_true, y_pred, model_name)
        st.pyplot(fig)

    with result_tabs[3]:
        st.subheader("真实值 vs 预测值散点图")

        # 绘制散点图
        fig = plot_scatter_actual_vs_pred(y_true, y_pred, model_name)
        st.pyplot(fig)


def display_multi_validation_results():
    """显示多模型验证结果"""
    if st.session_state.multi_validation_results is None:
        st.info("请先进行多模型验证或导入验证结果")
        return

    results = st.session_state.multi_validation_results

    # 创建选项卡显示不同类型的结果
    result_tabs = st.tabs(["比较结果", "预测对比", "指标对比"])

    with result_tabs[0]:
        st.subheader("模型比较结果")

        # 提取模型名称和指标
        model_names = results['model_names']
        all_metrics = results['metrics']

        # 创建比较表格
        comparison_data = {
            "指标": ["MSE", "MAE", "RMSE", "R²"]
        }

        # 为每个模型添加列
        for i, (name, metrics) in enumerate(zip(model_names, all_metrics)):
            comparison_data[name] = [
                f"{metrics.get('mse', 0):.4f}",
                f"{metrics.get('mae', 0):.4f}",
                f"{metrics.get('rmse', 0):.4f}",
                f"{metrics.get('r2', 0):.4f}"
            ]

        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

        # 找出最佳模型
        best_mse_idx = np.argmin([m['mse'] for m in all_metrics])
        best_r2_idx = np.argmax([m['r2'] for m in all_metrics])

        col1, col2 = st.columns(2)
        with col1:
            st.info(f"最低MSE: {model_names[best_mse_idx]}")
        with col2:
            st.info(f"最高R²: {model_names[best_r2_idx]}")

        # 导出结果按钮
        export_link = export_multi_model_results(results, "multi_model_validation_results.xlsx")
        if export_link:
            st.markdown(export_link, unsafe_allow_html=True)

    with result_tabs[1]:
        st.subheader("多模型预测对比")

        # 绘制多模型预测对比图
        y_true = results['y']
        all_predictions = results['predictions']
        model_names = results['model_names']

        fig = plot_multi_model_predictions(y_true, all_predictions, model_names)
        st.pyplot(fig)

    with result_tabs[2]:
        st.subheader("模型指标对比")

        # 绘制多模型指标对比图
        all_metrics = results['metrics']
        model_names = results['model_names']

        fig = plot_multi_model_metrics(all_metrics, model_names)
        st.pyplot(fig)


def handle_validation():
    """处理验证过程的逻辑"""
    # 检查是否需要跳转到结果选项卡
    if st.session_state.get("validation_tab", 0) == 4:
        # 在侧边栏中添加跳转到结果选项卡的链接
        st.sidebar.success("验证已启动，请查看结果选项卡")

        # 关键: 不重置validation_tab，即使没有结果
        # 让用户留在结果选项卡