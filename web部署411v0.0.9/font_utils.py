# font_utils.py - 优化解决方案
import os
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO)


@st.cache_resource
def setup_chinese_font():
    """
    从项目内部加载字体，并将其注册到 matplotlib 的字体管理器中。
    这是最彻底的解决方案，可以处理代码中各种字体名称的请求。
    """
    logging.info("开始从项目路径设置中文字体...")

    try:
        # 构建字体文件的完整路径
        project_root = os.getcwd()
        # 路径调整为相对于项目根目录
        font_file_name = "fonts/NotoSansSC-VariableFont_wght.ttf"
        local_font_path = os.path.join(project_root, font_file_name)

        # 兼容旧的路径结构
        if not os.path.exists(local_font_path):
            app_subdirectory = "web部署411v0.0.9"
            local_font_path = os.path.join(project_root, app_subdirectory, "fonts", "NotoSansSC-VariableFont_wght.ttf")

        if os.path.exists(local_font_path):
            logging.info(f"在项目路径中找到字体文件: {local_font_path}")

            # 将字体文件添加到 matplotlib 的字体管理器
            try:
                fm.fontManager.addfont(local_font_path)
                logging.info("字体已成功添加到 matplotlib 字体管理器")
            except Exception as e:
                logging.warning(f"无法添加字体到字体管理器: {e}")

            # 获取字体的实际名称
            font_prop = FontProperties(fname=local_font_path)
            actual_font_name = font_prop.get_name()
            logging.info(f"字体实际名称: {actual_font_name}")

            # --- 优化核心 ---
            # 将找到的字体名插入到 sans-serif 列表的最前面，而不是替换整个列表
            # 这样可以避免因硬编码的字体名不存在而产生的警告
            plt.rcParams['font.sans-serif'].insert(0, actual_font_name)

            # 确保字体家族设置为sans-serif
            plt.rcParams['font.family'] = 'sans-serif'

            # 解决负号显示问题
            plt.rcParams['axes.unicode_minus'] = False

            # 保存字体信息到 session_state
            st.session_state['font_prop'] = font_prop
            st.session_state['chinese_font_path'] = local_font_path
            st.session_state['font_name'] = actual_font_name

            logging.info(f"字体设置完成！当前 font.sans-serif 列表最优先字体: {plt.rcParams['font.sans-serif'][0]}")
            return True

        else:
            logging.error(f"字体文件不存在: {local_font_path}")
            st.error(f"部署环境中未找到指定的字体文件。请确保 'fonts/NotoSansSC-VariableFont_wght.ttf' 文件存在。")
            return False

    except Exception as e:
        logging.error(f"设置字体时发生严重错误: {e}", exc_info=True)
        st.error(f"设置中文字体时出错: {e}")
        return False


def get_chinese_font_prop():
    """
    获取中文字体属性对象，确保字体已经设置
    """
    if 'font_prop' not in st.session_state:
        setup_chinese_font()

    return st.session_state.get('font_prop', FontProperties())


def apply_chinese_to_axes(ax):
    """
    将中文字体应用到特定的 axes 对象
    """
    font_prop = get_chinese_font_prop()
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        if item:
            item.set_fontproperties(font_prop)
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_fontproperties(font_prop)
    return ax


# --- 其他辅助函数保持不变 ---
def create_chinese_figure(figsize=(10, 6)):
    setup_chinese_font()
    return plt.figure(figsize=figsize)


def chinese_text(ax, x, y, text, **kwargs):
    return ax.text(x, y, text, fontproperties=get_chinese_font_prop(), **kwargs)


def chinese_title(ax, title, **kwargs):
    return ax.set_title(title, fontproperties=get_chinese_font_prop(), **kwargs)


def chinese_xlabel(ax, label, **kwargs):
    return ax.set_xlabel(label, fontproperties=get_chinese_font_prop(), **kwargs)


def chinese_ylabel(ax, label, **kwargs):
    return ax.set_ylabel(label, fontproperties=get_chinese_font_prop(), **kwargs)


def chinese_legend(ax, labels=None, **kwargs):
    font_prop = get_chinese_font_prop()
    if labels:
        return ax.legend(labels, prop=font_prop, **kwargs)
    return ax.legend(prop=font_prop, **kwargs)


def set_chinese_labels(ax, title=None, xlabel=None, ylabel=None):
    if title: chinese_title(ax, title)
    if xlabel: chinese_xlabel(ax, xlabel)
    if ylabel: chinese_ylabel(ax, ylabel)
    return ax


# 兼容旧代码的别名
def setup_better_chinese_font():
    return setup_chinese_font()


# 在模块加载时自动初始化
if __name__ != "__main__":
    setup_chinese_font()