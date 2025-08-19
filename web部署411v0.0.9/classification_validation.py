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

# For classification algorithms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Font setup for visualization (reusing from classification_training)
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


# Plot functions for classification validation results
def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """绘制混淆矩阵"""
    fig, ax = create_figure_with_safe_dimensions(10, 8)
    apply_plot_style(ax)

    try:
        cm = confusion_matrix(y_true, y_pred)

        if class_names is None:
            class_names = [str(i) for i in range(cm.shape[0])]

        # Create heatmap with annotations
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)

        font_kwargs = {'fontproperties': FONT_PROP} if 'FONT_PROP' in globals() else {}
        ax.set_xlabel('预测类别', fontsize=10, fontweight='bold', **font_kwargs)
        ax.set_ylabel('真实类别', fontsize=10, fontweight='bold', **font_kwargs)
        ax.set_title('混淆矩阵', fontsize=12, fontweight='bold', **font_kwargs)

        # Rotate labels for better readability if needed
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", **font_kwargs)
        plt.setp(ax.get_yticklabels(), rotation=0, **font_kwargs)

        plt.tight_layout()
        return fig

    except Exception as e:
        print(f"绘制混淆矩阵时出错: {e}")
        ax.text(0.5, 0.5, f'绘制混淆矩阵时出错: {e}',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12, color='red',
                fontproperties=FONT_PROP if 'FONT_PROP' in globals() else None)
        return fig


def plot_class_distribution(y_true, y_pred, class_names=None, model_name="Model"):
    """绘制类别分布比较（真实值与预测值）"""
    fig, ax = create_figure_with_safe_dimensions(10, 8)
    apply_plot_style(ax)

    try:
        # 统计每个类别在真实和预测中的出现次数
        if class_names is None:
            class_names = sorted(list(set(list(y_true) + list(y_pred))))

        true_counts = pd.Series(y_true).value_counts().reindex(class_names, fill_value=0)
        pred_counts = pd.Series(y_pred).value_counts().reindex(class_names, fill_value=0)

        # 设置条形的位置
        x = np.arange(len(class_names))
        width = 0.35

        # 创建条形
        ax.bar(x - width / 2, true_counts, width, label='真实分布', color='#2ecc71', alpha=0.8)
        ax.bar(x + width / 2, pred_counts, width, label=f'{model_name}预测分布', color='#e74c3c', alpha=0.8)

        # 添加标签、标题和图例
        font_kwargs = {'fontproperties': FONT_PROP} if 'FONT_PROP' in globals() else {}
        ax.set_xlabel('类别', fontsize=10, fontweight='bold', **font_kwargs)
        ax.set_ylabel('样本数量', fontsize=10, fontweight='bold', **font_kwargs)
        ax.set_title('类别分布对比', fontsize=12, fontweight='bold', **font_kwargs)
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right', **font_kwargs)

        legend = ax.legend(frameon=True, framealpha=0.9, edgecolor='#3498db',
                           prop=FONT_PROP if 'FONT_PROP' in globals() else None, fontsize=9)

        # 在条形顶部添加计数数字
        for i, v in enumerate(true_counts):
            ax.text(i - width / 2, v + 0.1, str(int(v)), ha='center', fontsize=8)
        for i, v in enumerate(pred_counts):
            ax.text(i + width / 2, v + 0.1, str(int(v)), ha='center', fontsize=8)

        plt.tight_layout()
        return fig

    except Exception as e:
        print(f"绘制类别分布时出错: {e}")
        ax.text(0.5, 0.5, f'绘制类别分布时出错: {e}',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12, color='red',
                fontproperties=FONT_PROP if 'FONT_PROP' in globals() else None)
        return fig


def plot_multi_model_class_distribution(y_true, all_predictions, class_names, model_names=None):
    """绘制多模型类别分布比较"""
    fig, ax = create_figure_with_safe_dimensions(10, 8)
    apply_plot_style(ax)

    try:
        # 确保值计数正确
        true_counts = pd.Series(y_true).value_counts().reindex(class_names, fill_value=0)

        # 调整条形图位置参数
        num_models = len(all_predictions)
        x = np.arange(len(class_names))
        total_width = 0.8  # 调整总宽度
        bar_width = total_width / (num_models + 1)

        # 使用更明显的颜色区分
        colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']

        # 绘制真实值条形图
        true_bars = ax.bar(
            x - total_width / 2 + bar_width / 2,
            true_counts,
            bar_width,
            label='真实分布',
            color=colors[0],
            alpha=0.8
        )

        # 为每个条形添加数值标签
        for i, v in enumerate(true_counts):
            ax.text(
                x[i] - total_width / 2 + bar_width / 2,
                v + 0.1,
                str(int(v)),
                ha='center',
                fontsize=8
            )

        # 绘制每个模型的预测值
        for i, predictions in enumerate(all_predictions):
            model_name = model_names[i] if model_names and i < len(model_names) else f"模型 {i + 1}"
            pred_counts = pd.Series(predictions).value_counts().reindex(class_names, fill_value=0)

            # 计算条形位置
            position = x - total_width / 2 + bar_width / 2 + (i + 1) * bar_width

            pred_bars = ax.bar(
                position,
                pred_counts,
                bar_width,
                label=f'{model_name}预测',
                color=colors[(i + 1) % len(colors)],
                alpha=0.8
            )

            # 为每个条形添加数值标签
            for j, v in enumerate(pred_counts):
                ax.text(
                    position[j],
                    v + 0.1,
                    str(int(v)),
                    ha='center',
                    fontsize=8
                )

        # 设置坐标轴和标签
        font_kwargs = {'fontproperties': FONT_PROP} if 'FONT_PROP' in globals() else {}
        ax.set_xlabel('类别', fontsize=10, fontweight='bold', **font_kwargs)
        ax.set_ylabel('样本数量', fontsize=10, fontweight='bold', **font_kwargs)
        ax.set_title('多模型类别分布对比', fontsize=12, fontweight='bold', **font_kwargs)

        # 确保x轴刻度位置正确
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right', **font_kwargs)

        # 确保图例正确显示
        legend = ax.legend(
            frameon=True,
            framealpha=0.9,
            edgecolor='#3498db',
            prop=FONT_PROP,
            fontsize=8,
            loc='upper right'
        )

        plt.tight_layout()
        return fig

    except Exception as e:
        import traceback
        print(f"绘制多模型结果错误: {e}\n{traceback.format_exc()}")
        ax.text(0.5, 0.5, f'绘制多模型对比时出错: {e}',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12, color='red',
                fontproperties=FONT_PROP if 'FONT_PROP' in globals() else None)
        return fig


def plot_model_metrics(metrics, model_name="Model"):
    """绘制模型指标条形图"""
    fig, ax = create_figure_with_safe_dimensions(10, 6)
    apply_plot_style(ax)

    try:
        # 准备指标数据
        labels = ['准确率', '精确率', '召回率', 'F1分数']
        values = [
            metrics.get('accuracy', 0),
            metrics.get('precision', 0),
            metrics.get('recall', 0),
            metrics.get('f1', 0)
        ]

        x = np.arange(len(labels))
        bars = ax.bar(x, values, color='#9b59b6', alpha=0.7)  # 紫色条形图

        # 设置图表属性
        ax.set_ylim(0, 1.0)

        # 应用字体到标题和标签
        font_kwargs = {'fontproperties': FONT_PROP} if 'FONT_PROP' in globals() else {}
        ax.set_title(f'{model_name}性能指标', fontsize=12, fontweight='bold', **font_kwargs)
        ax.set_ylabel('分数', fontsize=10, **font_kwargs)

        # 设置刻度和标签
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, **font_kwargs)

        # 添加值标签
        for i, v in enumerate(values):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)

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
    """绘制多模型指标对比条形图"""
    fig, ax = create_figure_with_safe_dimensions(10, 6)
    apply_plot_style(ax)

    try:
        metrics_labels = ['准确率', '精确率', '召回率', 'F1分数']
        metrics_keys = ['accuracy', 'precision', 'recall', 'f1']

        # 获取每个模型的指标
        all_metrics = []
        for i, metrics in enumerate(metrics_list):
            model_metrics = []
            for key in metrics_keys:
                model_metrics.append(metrics.get(key, 0))
            all_metrics.append(model_metrics)

        # 设置x轴位置
        x = np.arange(len(metrics_labels))
        width = 0.8 / len(metrics_list)  # 根据模型数量调整条形宽度

        # 为每个模型设置不同颜色
        colors = ['#9b59b6', '#e74c3c', '#3498db']  # 紫色, 红色, 蓝色

        # 绘制每个模型的指标条形图
        for i, model_metrics in enumerate(all_metrics):
            model_name = model_names[i] if model_names and i < len(model_names) else f"模型 {i + 1}"
            offset = (i - len(metrics_list) / 2 + 0.5) * width
            bars = ax.bar(x + offset, model_metrics,
                          width, label=model_name,
                          color=colors[i % len(colors)], alpha=0.7)

            # 在条形上方添加值标签
            for bar, value in zip(bars, model_metrics):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                        f'{value:.2f}', ha='center', fontsize=7)

        # 设置图表属性
        ax.set_ylim(0, 1.0)

        # 应用字体到标题和标签
        font_kwargs = {'fontproperties': FONT_PROP} if 'FONT_PROP' in globals() else {}
        ax.set_title('多模型性能指标对比', fontsize=12, fontweight='bold', **font_kwargs)
        ax.set_ylabel('分数', fontsize=10, **font_kwargs)

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


def plot_multi_confusion_matrices(confusion_matrices, class_names, model_names=None):
    """绘制多个混淆矩阵进行比较"""
    if not confusion_matrices or len(confusion_matrices) == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, '没有混淆矩阵数据可用',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12, color='orange',
                fontproperties=FONT_PROP if 'FONT_PROP' in globals() else None)
        return fig

    num_models = len(confusion_matrices)
    if num_models > 4:  # 限制最多4个模型
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, f'模型数量({num_models})过多，无法同时显示所有混淆矩阵',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12, color='orange',
                fontproperties=FONT_PROP if 'FONT_PROP' in globals() else None)
        return fig

    try:
        # 计算子图布局
        if num_models == 1:
            nrows, ncols = 1, 1
        elif num_models == 2:
            nrows, ncols = 1, 2
        else:
            nrows, ncols = 2, 2

        # 创建具有安全尺寸的图形
        fig = plt.figure(figsize=(4 * ncols, 4 * nrows), dpi=80)

        # 绘制每个混淆矩阵
        for idx, cm in enumerate(confusion_matrices):
            if idx < num_models:
                ax = fig.add_subplot(nrows, ncols, idx + 1)
                apply_plot_style(ax)

                model_name = model_names[idx] if model_names and idx < len(model_names) else f"模型 {idx + 1}"

                # 确保混淆矩阵为数值类型
                cm_array = np.array(cm, dtype=np.float64)

                # 绘制热图
                im = sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues',
                                 xticklabels=class_names, yticklabels=class_names, ax=ax)

                # 设置标题和标签
                font_kwargs = {'fontproperties': FONT_PROP} if 'FONT_PROP' in globals() else {}
                ax.set_title(f'{model_name}', fontsize=10, fontweight='bold', **font_kwargs)
                ax.set_xlabel('预测类别', fontsize=8, fontweight='bold', **font_kwargs)
                ax.set_ylabel('真实类别', fontsize=8, fontweight='bold', **font_kwargs)

                # 设置刻度标签
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7, **font_kwargs)
                plt.setp(ax.get_yticklabels(), fontsize=7, **font_kwargs)

        # 添加总标题
        if FONT_PROP:
            fig.suptitle('模型混淆矩阵对比', fontproperties=FONT_PROP, fontsize=12, fontweight='bold')
        else:
            fig.suptitle('模型混淆矩阵对比', fontsize=12, fontweight='bold')

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为suptitle留出空间
        return fig

    except Exception as e:
        import traceback
        print(f"绘制多个混淆矩阵时出错: {e}\n{traceback.format_exc()}")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, f'绘制多个混淆矩阵时出错: {e}',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12, color='red',
                fontproperties=FONT_PROP if 'FONT_PROP' in globals() else None)
        return fig


# 处理文件夹数据，将其中的CSV/Excel文件按子文件夹作为类别进行处理
def process_folder_data(folder_path, progress_callback=None):
    """处理包含子文件夹作为类别的文件夹数据"""
    try:
        # 检查文件夹是否有子文件夹
        subfolders = [f for f in os.listdir(folder_path)
                      if os.path.isdir(os.path.join(folder_path, f))]

        if not subfolders:
            return None, "所选文件夹没有包含子文件夹。分类验证需要每个类别有一个单独的子文件夹。"

        # 初始化数据收集
        all_data = []
        labels = []
        file_names = []

        # 处理每个子文件夹
        folder_count = len(subfolders)
        for i, subfolder in enumerate(subfolders):
            subfolder_path = os.path.join(folder_path, subfolder)

            # 获取所有CSV/Excel文件
            files = []
            for ext in ['.csv', '.xlsx', '.xls']:
                files.extend([f for f in os.listdir(subfolder_path)
                              if f.lower().endswith(ext)])

            # 处理每个文件
            for file in files:
                file_path = os.path.join(subfolder_path, file)

                try:
                    # 加载数据
                    if file.lower().endswith('.csv'):
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
                    labels.extend([subfolder] * len(numeric_df))
                    file_names.extend([file] * len(numeric_df))

                except Exception as e:
                    print(f"处理文件 {file} 时出错: {e}")

            # 更新进度
            if progress_callback:
                progress_percent = int((i + 1) / folder_count * 100)
                progress_callback(progress_percent)

        # 合并所有数据
        if not all_data:
            return None, "未找到有效的数据文件或所有文件处理失败"

        # 确保所有数据框有相同的列
        common_columns = set.intersection(*[set(df.columns) for df in all_data])
        if not common_columns:
            return None, "文件之间没有共同的数值列，无法合并数据"

        # 过滤到公共列并连接
        all_data = [df[list(common_columns)] for df in all_data]
        X = pd.concat(all_data, ignore_index=True)

        # 进行标签编码
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(labels)
        y_series = pd.Series(y, name='label')

        # 保存类别映射
        class_mapping = dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))

        # 创建文件名Series
        file_paths = pd.Series(file_names, name='file')

        # 返回处理后的数据
        return {
            'X': X,
            'y': y_series,
            'file_names': file_paths,
            'class_mapping': class_mapping,
            'label_encoder': label_encoder,
            'raw_labels': labels
        }, None

    except Exception as e:
        import traceback
        return None, f"处理文件夹时出错: {str(e)}\n{traceback.format_exc()}"


# 验证单个模型
def validate_model(X, y, model_path, use_sliding_window=False, window_size=3,
                   use_column_features=True, use_raw_columns=False, progress_callback=None):
    """验证单个分类模型并返回结果 - 支持所有模型类型"""
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
            class_names = model_info.get('class_names', [])
            scaler = model_info.get('scaler')
            label_encoder = model_info.get('label_encoder')
            model_params = model_info.get('params', {})

            if progress_callback:
                progress_callback(30)

            print(f"加载的模型类型: {model_type}")

        except Exception as e:
            raise ValueError(f"加载模型时出错: {str(e)}")

        # 确保特征顺序正确
        if feature_names and set(feature_names).issubset(set(X.columns)):
            X = X[feature_names]
        else:
            print(f"警告: 特征名称不匹配。模型期望: {feature_names}, 数据提供: {list(X.columns)}")

        # 数据预处理
        X_processed = X.copy()

        # 标准化（如果训练时使用了）
        if scaler is not None:
            X_processed = pd.DataFrame(
                scaler.transform(X_processed),
                columns=X_processed.columns,
                index=X_processed.index
            )

        if progress_callback:
            progress_callback(50)

        # 根据模型类型进行特殊处理
        if model_type in ['bp_neural_network', 'rnn', 'cnn', 'lstm', 'gru']:
            # 深度学习模型需要特殊处理
            y_pred, y_proba = validate_deep_learning_model(
                model, X_processed, y, model_type, model_params, label_encoder
            )
        else:
            # 传统机器学习模型
            y_pred, y_proba = validate_traditional_model(
                model, X_processed, model_type, label_encoder
            )

        if progress_callback:
            progress_callback(70)

        # 确保y_pred和y的类型一致
        if label_encoder is not None:
            # 如果y是编码后的，需要解码
            if isinstance(y.iloc[0] if hasattr(y, 'iloc') else y[0], (int, np.integer)):
                # y是编码的，y_pred已经是解码的，需要统一
                try:
                    y_decoded = label_encoder.inverse_transform(y)
                    y = pd.Series(y_decoded, index=y.index if hasattr(y, 'index') else None)
                except:
                    # 如果解码失败，将y_pred编码
                    y_pred = label_encoder.transform(y_pred)

        # 计算评估指标
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)

        # 计算混淆矩阵
        conf_matrix = confusion_matrix(y, y_pred)

        # 获取特征重要性（如果模型支持）
        feature_importance = get_feature_importance(model, feature_names, model_type)

        if progress_callback:
            progress_callback(90)

        # 准备结果
        results = {
            'model_info': {
                'model_path': model_path,
                'model_type': model_type,
                'feature_names': feature_names,
                'class_names': class_names,
                'params': model_params,
                'use_sliding_window': use_sliding_window,
                'window_size': window_size,
                'use_column_features': use_column_features,
                'use_raw_columns': use_raw_columns
            },
            'predictions': y_pred,
            'probabilities': y_proba,
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': conf_matrix
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


def validate_traditional_model(model, X, model_type, label_encoder=None):
    """验证传统机器学习模型"""
    # 预测
    y_pred = model.predict(X)

    # 获取概率（如果支持）
    y_proba = None
    if hasattr(model, 'predict_proba'):
        try:
            y_proba = model.predict_proba(X)
        except:
            pass

    # 如果是XGBoost且使用了标签编码器，需要解码
    if model_type == 'xgboost' and label_encoder is not None:
        try:
            y_pred = label_encoder.inverse_transform(y_pred)
        except:
            pass

    return y_pred, y_proba


def validate_deep_learning_model(model, X, y, model_type, params, label_encoder=None):
    """验证深度学习模型"""
    import numpy as np

    # 准备数据
    X_processed = X.values.astype(np.float32)

    if model_type == 'bp_neural_network':
        # BP神经网络直接使用原始数据
        X_final = X_processed

    elif model_type in ['rnn', 'lstm', 'gru']:
        # 序列模型需要重塑数据
        sequence_length = params.get('sequence_length', 10)
        X_final = prepare_sequence_data_for_validation(X_processed, sequence_length)

    elif model_type == 'cnn':
        # CNN需要添加通道维度
        X_final = prepare_cnn_data_for_validation(X_processed)

    else:
        X_final = X_processed

    # 预测
    y_pred_proba = model.predict(X_final, verbose=0)

    # 获取类别数
    if len(y_pred_proba.shape) == 1 or y_pred_proba.shape[1] == 1:
        # 二分类
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        # 构建概率矩阵
        y_proba = np.column_stack([1 - y_pred_proba.flatten(), y_pred_proba.flatten()])
    else:
        # 多分类
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_proba = y_pred_proba

    # 如果有标签编码器，解码预测结果
    if label_encoder is not None:
        try:
            y_pred = label_encoder.inverse_transform(y_pred)
        except:
            pass

    return y_pred, y_proba


def get_feature_importance(model, feature_names, model_type):
    """获取特征重要性（根据模型类型）"""
    feature_importance = {}

    try:
        if hasattr(model, 'feature_importances_'):
            # 树模型（随机森林、GBDT等）
            importance = model.feature_importances_
            if len(feature_names) == len(importance):
                feature_importance = dict(zip(feature_names, importance))

        elif model_type == 'logistic_regression' and hasattr(model, 'coef_'):
            # 逻辑回归
            importance = np.abs(model.coef_).mean(axis=0)
            if len(feature_names) == len(importance):
                feature_importance = dict(zip(feature_names, importance))

        elif model_type == 'svm' and hasattr(model, 'coef_'):
            # 线性SVM
            importance = np.abs(model.coef_).mean(axis=0)
            if len(feature_names) == len(importance):
                feature_importance = dict(zip(feature_names, importance))

    except Exception as e:
        print(f"获取特征重要性时出错: {e}")

    return feature_importance


def prepare_sequence_data_for_validation(X, sequence_length):
    """为序列模型准备验证数据"""
    n_samples, n_features = X.shape

    if n_features >= sequence_length:
        # 如果特征数足够，使用前sequence_length个特征
        X_reshaped = X[:, :sequence_length].reshape(n_samples, sequence_length, 1)
    else:
        # 如果特征数不足，进行填充
        X_padded = np.pad(X, ((0, 0), (0, sequence_length - n_features)), mode='constant')
        X_reshaped = X_padded.reshape(n_samples, sequence_length, 1)

    return X_reshaped.astype(np.float32)


def prepare_cnn_data_for_validation(X):
    """为CNN准备验证数据"""
    # 添加通道维度
    return np.expand_dims(X, axis=2).astype(np.float32)



# 验证多个模型
def validate_multiple_models(X, y, model_paths, use_sliding_window=False, window_size=3,
                             use_column_features=True, use_raw_columns=False, progress_callback=None):
    """验证多个分类模型并返回比较结果"""
    try:
        # 初始化结果
        all_results = {
            'model_infos': [],
            'predictions': [],
            'probabilities': [],
            'metrics': [],
            'feature_importances': [],
            'model_names': [],
            'model_paths': model_paths,
            'y': y,
            'class_names': None  # 将从第一个模型设置
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
            probabilities = single_results['probabilities']
            metrics = single_results['metrics']
            feature_importance = single_results['feature_importance']

            # 设置类别名称（如果尚未设置）
            if all_results['class_names'] is None:
                all_results['class_names'] = model_info['class_names']

            # 添加到总结果
            all_results['model_infos'].append(model_info)
            all_results['predictions'].append(predictions)
            all_results['probabilities'].append(probabilities)
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
def export_single_model_results(results, filename="classification_validation_results.csv"):
    """导出单模型验证结果到CSV"""
    try:
        y = results['y']
        pred = results['predictions']

        # 创建数据框
        df = pd.DataFrame({
            '样本索引': np.arange(len(y)),
            '真实类别': y,
            '预测类别': pred
        })

        # 准备总体指标
        metrics = results['metrics']
        metrics_df = pd.DataFrame({
            '指标': ['准确率', '精确率', '召回率', 'F1分数'],
            '值': [
                metrics.get('accuracy', 'N/A'),
                metrics.get('precision', 'N/A'),
                metrics.get('recall', 'N/A'),
                metrics.get('f1', 'N/A')
            ]
        })

        # 保存到内存中的CSV
        buffer = BytesIO()

        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            # 写入元数据
            info_df = pd.DataFrame({
                '信息': [
                    '分类模型验证结果',
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
def export_multi_model_results(results, filename="multi_model_validation_results.csv"):
    """导出多模型验证结果到CSV"""
    try:
        y = results['y']
        all_predictions = results['predictions']
        model_names = results['model_names']
        all_metrics = results['metrics']

        # 创建样本预测数据框
        data = {'样本索引': np.arange(len(y)), '真实类别': y}

        # 添加每个模型的预测
        for i, (pred, name) in enumerate(zip(all_predictions, model_names)):
            data[f'{name}预测类别'] = pred

        df = pd.DataFrame(data)

        # 创建总体指标数据框
        metrics_data = {'指标': ['准确率', '精确率', '召回率', 'F1分数']}

        for i, (metrics, name) in enumerate(zip(all_metrics, model_names)):
            metrics_data[name] = [
                metrics.get('accuracy', 'N/A'),
                metrics.get('precision', 'N/A'),
                metrics.get('recall', 'N/A'),
                metrics.get('f1', 'N/A')
            ]

        metrics_df = pd.DataFrame(metrics_data)

        # 保存到内存中的Excel
        buffer = BytesIO()

        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            # 写入元数据
            info_data = [['多模型分类验证结果'], [f'验证时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}']]
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
def show_classification_validation_page():
    """显示分类验证页面"""
    st.title("分类模型验证")

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

    - **单模型验证**: 验证一个分类模型在给定数据集上的性能
    - **多模型比较**: 比较多个分类模型在同一数据集上的性能（最多3个模型）
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
            class_count = len(model_info.get('class_names', []))

            # 显示模型类型的中文名称
            from classification_training import CLASSIFIER_INFO
            model_name = CLASSIFIER_INFO.get(model_type, {}).get('name', model_type)

            # 显示模型信息
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("模型类型", model_name)
            with col2:
                st.metric("特征数量", feature_count)
            with col3:
                st.metric("类别数量", class_count)

            # 显示模型参数
            if 'params' in model_info and model_info['params']:
                with st.expander("查看模型参数"):
                    params_df = pd.DataFrame(
                        list(model_info['params'].items()),
                        columns=['参数', '值']
                    )
                    st.dataframe(params_df, hide_index=True)

            # 显示特征名称
            if 'feature_names' in model_info and model_info['feature_names']:
                with st.expander("查看特征名称"):
                    st.write(", ".join(model_info['feature_names']))

            # 显示类别名称
            if 'class_names' in model_info and model_info['class_names']:
                with st.expander("查看类别名称"):
                    st.write(", ".join(map(str, model_info['class_names'])))

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
        folder_path = st.text_input("输入文件夹路径 (子文件夹作为分类标签)", key="folder_path")

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
                            st.session_state.column_names = list(results['X'].columns)
                            st.session_state.data_source_type = "folder"
                            st.session_state.data_loaded = True

                            # 成功消息
                            st.success(
                                f"已成功加载文件夹数据: {len(results['X'])} 行, {len(results['class_mapping'])} 个类别")

                            # 显示类别映射
                            st.info("类别映射:")
                            mapping_df = pd.DataFrame([
                                {"类别ID": k, "类别名称": v}
                                for k, v in results['class_mapping'].items()
                            ])
                            st.dataframe(mapping_df)

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

        if st.session_state.data_source_type == "file":
            st.dataframe(st.session_state.data.head())
        else:
            st.dataframe(st.session_state.data['X'].head())

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

    # Inicializar variables temporales si no existen
    if 'temp_selected_input_columns' not in st.session_state:
        st.session_state.temp_selected_input_columns = st.session_state.selected_input_columns.copy()

    if 'temp_selected_output_column' not in st.session_state:
        st.session_state.temp_selected_output_column = st.session_state.selected_output_column

    if 'columns_selected_flag' not in st.session_state:
        st.session_state.columns_selected_flag = False

    # 创建三列布局
    col1, col2, col3 = st.columns([3, 3, 2])

    # 根据数据源类型显示不同的界面
    if st.session_state.data_source_type == "file":
        # 文件数据 - 需要选择输入和输出列
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
                    # 不调用 st.rerun()

                # 设为输出标签的按钮
                if len(selected_available) == 1 and st.button("设为分类标签", key="set_output"):
                    st.session_state.temp_selected_output_column = selected_available[0]
                    # 不调用 st.rerun()
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
                    # 不调用 st.rerun()
            else:
                st.info("未选择输入特征")

        with col3:
            st.subheader("分类标签")
            if st.session_state.temp_selected_output_column:
                st.info(f"已选择: {st.session_state.temp_selected_output_column}")

                # 清除输出列按钮
                if st.button("清除分类标签", key="clear_output"):
                    st.session_state.temp_selected_output_column = None
                    # 不调用 st.rerun()
            else:
                st.info("未选择分类标签")

    else:
        # 文件夹数据 - 只需要选择输入列，输出列是固定的
        with col1:
            st.subheader("可用特征")
            if st.session_state.column_names:
                # 过滤掉已在临时选择中的列
                available_columns = [col for col in st.session_state.column_names
                                     if col not in st.session_state.temp_selected_input_columns]

                selected_available = st.multiselect("选择列", available_columns, key="folder_available_cols")

                # 添加到输入特征的按钮
                if selected_available and st.button("添加为输入特征", key="folder_add_input"):
                    new_inputs = st.session_state.temp_selected_input_columns.copy()
                    new_inputs.extend(selected_available)
                    st.session_state.temp_selected_input_columns = new_inputs
                    # 不调用 st.rerun()
            else:
                st.info("没有可用特征")

        with col2:
            st.subheader("输入特征")
            if st.session_state.temp_selected_input_columns:
                selected_inputs = st.multiselect("已选特征", st.session_state.temp_selected_input_columns,
                                                 default=st.session_state.temp_selected_input_columns,
                                                 key="folder_selected_inputs")

                # 如果有变化，更新临时选择
                if selected_inputs != st.session_state.temp_selected_input_columns:
                    st.session_state.temp_selected_input_columns = selected_inputs

                # 移除选中的输入特征
                if st.button("移除选中的输入特征", key="folder_remove_input"):
                    st.session_state.temp_selected_input_columns = []
                    # 不调用 st.rerun()
            else:
                st.info("未选择输入特征")

        with col3:
            st.subheader("分类标签")
            st.info("类别已由文件夹结构自动确定")
            # 自动设置输出列
            st.session_state.temp_selected_output_column = "label"  # 在文件夹处理中创建的标签列

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
                    (len(st.session_state.selected_input_columns) > 0 or
                     (st.session_state.get("use_raw_columns", False) and
                      st.session_state.data_source_type == "folder")) and
                    st.session_state.selected_output_column is not None)

    validate_col1, validate_col2 = st.columns([3, 1])

    with validate_col1:
        if st.button("验证模型", type="primary", disabled=not can_validate, key="validate_button"):
            if st.session_state.validation_mode == "single":
                # 设置状态以在结果选项卡中显示进度
                st.session_state.is_validating = True
                # 转到结果选项卡
                st.session_state.validation_tab = 4
                st.rerun()
            else:
                # 多模型验证
                st.session_state.is_validating = True
                st.session_state.validation_tab = 4
                st.rerun()

    with validate_col2:
        # 显示验证进度
        st.progress(st.session_state.validation_progress / 100)


def create_results_section():
    """创建结果展示部分"""
    st.header("验证结果")



    # 添加调试信息（可选）
    # st.write(f"验证状态: {st.session_state.get('is_validating', False)}")
    # st.write(f"单模型结果: {'存在' if st.session_state.validation_results is not None else '不存在'}")
    # st.write(f"多模型结果: {'存在' if st.session_state.multi_validation_results is not None else '不存在'}")

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
                if st.session_state.data_source_type == "file":
                    X = st.session_state.data[st.session_state.selected_input_columns].copy()
                    y = st.session_state.data[st.session_state.selected_output_column].copy()
                else:
                    X = st.session_state.data['X'][st.session_state.selected_input_columns].copy()
                    y = st.session_state.data['y'].copy()

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
                # 不要在这里调用 st.rerun()

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
                    st.experimental_rerun()


def display_single_validation_results():
    """显示单模型验证结果"""
    if st.session_state.validation_results is None:
        st.info("请先进行验证或导入验证结果")
        return

    results = st.session_state.validation_results

    # 创建选项卡显示不同类型的结果
    result_tabs = st.tabs(["性能指标", "类别分布", "混淆矩阵", "模型指标"])

    with result_tabs[0]:
        st.subheader("验证性能指标")

        metrics = results['metrics']

        # 使用列布局显示指标
        metric_cols = st.columns(4)

        with metric_cols[0]:
            st.metric("准确率", f"{metrics['accuracy']:.4f}")

        with metric_cols[1]:
            st.metric("精确率", f"{metrics['precision']:.4f}")

        with metric_cols[2]:
            st.metric("召回率", f"{metrics['recall']:.4f}")

        with metric_cols[3]:
            st.metric("F1分数", f"{metrics['f1']:.4f}")

        # 显示模型信息
        st.subheader("模型信息")

        # 提取模型信息
        model_info = results['model_info']
        model_type = model_info.get('model_type', '未知')
        feature_names = model_info.get('feature_names', [])
        class_names = model_info.get('class_names', [])

        # 显示信息表格
        info_data = {
            "参数": ["模型类型", "特征数量", "类别数量", "使用滑动窗口", "窗口大小", "使用列特征", "使用原始列"],
            "值": [
                model_type,
                len(feature_names),
                len(class_names),
                "是" if model_info.get('use_sliding_window', False) else "否",
                model_info.get('window_size', 'N/A'),
                "是" if model_info.get('use_column_features', True) else "否",
                "是" if model_info.get('use_raw_columns', False) else "否"
            ]
        }

        st.dataframe(pd.DataFrame(info_data), use_container_width=True)

        # 导出结果按钮
        export_link = export_single_model_results(results, "classification_validation_results.xlsx")
        if export_link:
            st.markdown(export_link, unsafe_allow_html=True)

    with result_tabs[1]:
        st.subheader("类别分布")

        # 绘制类别分布图
        y_true = results['y']
        y_pred = results['predictions']
        class_names = results['model_info'].get('class_names', [])

        fig = plot_class_distribution(y_true, y_pred, class_names)
        st.pyplot(fig)

    with result_tabs[2]:
        st.subheader("混淆矩阵")

        # 绘制混淆矩阵
        conf_matrix = results['metrics'].get('confusion_matrix')
        class_names = results['model_info'].get('class_names', [])

        fig = plot_confusion_matrix(y_true, y_pred, class_names)
        st.pyplot(fig)

    with result_tabs[3]:
        st.subheader("模型指标图")

        # 绘制模型指标条形图
        metrics = results['metrics']
        model_name = os.path.basename(results['model_info'].get('model_path', '')).replace('.joblib', '')

        fig = plot_model_metrics(metrics, model_name)
        st.pyplot(fig)


def display_multi_validation_results():
    """显示多模型验证结果"""
    if st.session_state.multi_validation_results is None:
        st.info("请先进行多模型验证或导入验证结果")
        return

    results = st.session_state.multi_validation_results

    # 创建选项卡显示不同类型的结果
    result_tabs = st.tabs(["比较结果", "类别分布", "混淆矩阵", "模型指标"])

    with result_tabs[0]:
        st.subheader("模型比较结果")

        # 提取模型名称和指标
        model_names = results['model_names']
        all_metrics = results['metrics']

        # 创建比较表格
        comparison_data = {
            "指标": ["准确率", "精确率", "召回率", "F1分数"]
        }

        # 为每个模型添加列
        for i, (name, metrics) in enumerate(zip(model_names, all_metrics)):
            comparison_data[name] = [
                f"{metrics.get('accuracy', 0):.4f}",
                f"{metrics.get('precision', 0):.4f}",
                f"{metrics.get('recall', 0):.4f}",
                f"{metrics.get('f1', 0):.4f}"
            ]

        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

        # 导出结果按钮
        export_link = export_multi_model_results(results, "multi_model_validation_results.xlsx")
        if export_link:
            st.markdown(export_link, unsafe_allow_html=True)

    with result_tabs[1]:
        st.subheader("类别分布比较")

        # 绘制多模型类别分布图
        y_true = results['y']
        all_predictions = results['predictions']
        class_names = results['class_names']
        model_names = results['model_names']

        fig = plot_multi_model_class_distribution(y_true, all_predictions, class_names, model_names)
        st.pyplot(fig)

    with result_tabs[2]:
        st.subheader("混淆矩阵比较")

        # 绘制多个混淆矩阵
        confusion_matrices = [metrics.get('confusion_matrix') for metrics in results['metrics']]
        class_names = results['class_names']
        model_names = results['model_names']

        fig = plot_multi_confusion_matrices(confusion_matrices, class_names, model_names)
        st.pyplot(fig)

    with result_tabs[3]:
        st.subheader("模型指标比较")

        # 绘制多模型指标比较图
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