# plot_helper.py - 智能中英文切换助手
import matplotlib.pyplot as plt
import streamlit as st

# 标签字典
LABELS = {
    # 通用标签
    '数据分析': 'Data Analysis',
    '机器学习': 'Machine Learning',
    '深度学习': 'Deep Learning',
    '特征工程': 'Feature Engineering',
    '模型训练': 'Model Training',
    '模型评估': 'Model Evaluation',
    '数据预处理': 'Data Preprocessing',
    '数据可视化': 'Data Visualization',

    # 轴标签
    '时间': 'Time',
    '数值': 'Value',
    '准确率': 'Accuracy',
    '损失': 'Loss',
    '类别': 'Category',
    '数量': 'Count',
    '频率': 'Frequency',
    '概率': 'Probability',

    # 图表标题
    '训练结果': 'Training Results',
    '模型性能': 'Model Performance',
    '数据分布': 'Data Distribution',
    '预测结果': 'Prediction Results',

    # 图例
    '训练集': 'Train Set',
    '测试集': 'Test Set',
    '验证集': 'Val Set',
    '预测值': 'Predicted',
    '实际值': 'Actual',
}


def smart_label(text, force_english=False):
    """
    智能转换标签：如果系统不支持中文或强制英文，则返回英文

    Args:
        text: 中文文本
        force_english: 是否强制使用英文

    Returns:
        合适的标签文本
    """
    # 检查是否应该使用英文
    use_english = force_english or st.session_state.get('use_english_labels', False)

    if use_english and text in LABELS:
        return LABELS[text]
    return text


def set_chinese_labels(ax, title=None, xlabel=None, ylabel=None, force_english=False):
    """
    为图表设置标签，自动处理中英文

    Args:
        ax: matplotlib axes 对象
        title: 标题
        xlabel: X轴标签
        ylabel: Y轴标签
        force_english: 是否强制使用英文
    """
    if title:
        ax.set_title(smart_label(title, force_english))
    if xlabel:
        ax.set_xlabel(smart_label(xlabel, force_english))
    if ylabel:
        ax.set_ylabel(smart_label(ylabel, force_english))

    return ax


def check_chinese_support():
    """
    检查系统是否支持中文显示

    Returns:
        bool: True if Chinese is supported
    """
    try:
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.text(0.5, 0.5, '测试')
        plt.close(fig)

        # 检查是否已经设置了中文字体
        fonts = plt.rcParams['font.sans-serif']
        has_chinese = any('cjk' in f.lower() or 'noto' in f.lower() for f in fonts)

        return has_chinese
    except:
        return False


def auto_detect_language():
    """
    自动检测并设置语言偏好
    """
    if not check_chinese_support():
        st.session_state['use_english_labels'] = True
        st.info("System doesn't support Chinese fonts. Using English labels.")
    else:
        st.session_state['use_english_labels'] = False


# 使用示例
def example_usage():
    """
    示例：如何使用这个助手
    """
    import numpy as np

    # 自动检测语言支持
    auto_detect_language()

    # 创建图表
    fig, ax = plt.subplots()

    # 数据
    x = np.arange(5)
    y = [23, 45, 67, 89, 90]

    # 绘制
    ax.plot(x, y)

    # 使用助手设置标签
    set_chinese_labels(ax,
                       title='数据分析',  # 会自动转换为 'Data Analysis' 如果需要
                       xlabel='时间',  # 会自动转换为 'Time' 如果需要
                       ylabel='数值')  # 会自动转换为 'Value' 如果需要

    return fig