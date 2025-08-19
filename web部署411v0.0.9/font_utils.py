# font_utils.py - 终极解决方案
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
    logging.info("开始从项目路径设置中文字体 (Ultimate Solution)...")

    try:
        # 构建字体文件的完整路径
        project_root = os.getcwd()
        app_subdirectory = "web部署411v0.0.9"
        font_file_name = "NotoSansSC-VariableFont_wght.ttf"
        local_font_path = os.path.join(project_root, app_subdirectory, "fonts", font_file_name)

        if os.path.exists(local_font_path):
            logging.info(f"在项目路径中找到字体文件: {local_font_path}")

            # 方案1：创建字体属性对象并设置为全局默认
            font_prop = FontProperties(fname=local_font_path)

            # 方案2：将字体文件添加到 matplotlib 的字体管理器
            try:
                # 注册字体到字体管理器
                fm.fontManager.addfont(local_font_path)
                logging.info("字体已添加到 matplotlib 字体管理器")
            except Exception as e:
                logging.warning(f"无法添加字体到字体管理器: {e}")

            # 方案3：设置多个可能的字体别名
            # 获取字体的实际名称
            actual_font_name = font_prop.get_name()
            logging.info(f"字体实际名称: {actual_font_name}")

            # 设置 matplotlib 的全局参数，使用多个后备选项
            plt.rcParams['font.sans-serif'] = [
                actual_font_name,  # 实际字体名
                'Noto Sans SC',  # 可能的别名1
                'Noto Sans CJK SC',  # 可能的别名2
                'DejaVu Sans'  # 最终后备
            ]
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

            # 保存字体信息到 session_state
            st.session_state['font_prop'] = font_prop
            st.session_state['chinese_font_path'] = local_font_path
            st.session_state['font_name'] = actual_font_name

            logging.info(f"字体设置完成！")
            logging.info(f"当前 font.sans-serif: {plt.rcParams['font.sans-serif']}")

            return True

        else:
            logging.error(f"字体文件不存在: {local_font_path}")

            # 尝试使用系统中已安装的字体作为后备
            logging.info("尝试使用系统字体作为后备...")
            system_fonts = [
                '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
                '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc'
            ]

            for sys_font_path in system_fonts:
                if os.path.exists(sys_font_path):
                    logging.info(f"找到系统字体: {sys_font_path}")
                    try:
                        fm.fontManager.addfont(sys_font_path)
                        font_prop = FontProperties(fname=sys_font_path)

                        plt.rcParams['font.sans-serif'] = [
                            font_prop.get_name(),
                            'Noto Sans CJK SC',
                            'DejaVu Sans'
                        ]
                        plt.rcParams['font.family'] = 'sans-serif'
                        plt.rcParams['axes.unicode_minus'] = False

                        st.session_state['font_prop'] = font_prop
                        st.session_state['chinese_font_path'] = sys_font_path

                        logging.info("成功使用系统字体作为后备")
                        return True
                    except Exception as e:
                        logging.warning(f"无法使用系统字体 {sys_font_path}: {e}")

            # 如果所有尝试都失败，设置一个基本的后备
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            logging.warning("未找到中文字体，使用默认字体")
            return False

    except Exception as e:
        logging.error(f"设置字体时发生错误: {e}", exc_info=True)
        st.error(f"设置中文字体时出错: {e}")
        return False


def get_chinese_font_prop():
    """
    获取中文字体属性对象，确保字体已经设置
    """
    if 'font_prop' not in st.session_state:
        setup_chinese_font()

    if 'font_prop' in st.session_state:
        return st.session_state['font_prop']
    else:
        # 返回一个默认的字体属性
        return FontProperties()


def apply_chinese_to_axes(ax):
    """
    将中文字体应用到特定的 axes 对象
    这个函数可以在绘图时直接调用，确保该图使用正确的字体
    """
    font_prop = get_chinese_font_prop()

    # 设置标题、标签的字体
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        if item:
            item.set_fontproperties(font_prop)

    # 如果有图例，也设置字体
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_fontproperties(font_prop)

    return ax


def create_chinese_figure(figsize=(10, 6)):
    """
    创建一个已经配置好中文字体的 figure
    """
    # 确保字体已设置
    setup_chinese_font()

    # 创建 figure
    fig = plt.figure(figsize=figsize)

    return fig


def chinese_text(ax, x, y, text, **kwargs):
    """在图表上添加中文文本"""
    font_prop = get_chinese_font_prop()
    return ax.text(x, y, text, fontproperties=font_prop, **kwargs)


def chinese_title(ax, title, **kwargs):
    """设置中文标题"""
    font_prop = get_chinese_font_prop()
    return ax.set_title(title, fontproperties=font_prop, **kwargs)


def chinese_xlabel(ax, label, **kwargs):
    """设置中文 X 轴标签"""
    font_prop = get_chinese_font_prop()
    return ax.set_xlabel(label, fontproperties=font_prop, **kwargs)


def chinese_ylabel(ax, label, **kwargs):
    """设置中文 Y 轴标签"""
    font_prop = get_chinese_font_prop()
    return ax.set_ylabel(label, fontproperties=font_prop, **kwargs)


def chinese_legend(ax, labels=None, **kwargs):
    """设置中文图例"""
    font_prop = get_chinese_font_prop()
    if labels:
        return ax.legend(labels, prop=font_prop, **kwargs)
    else:
        return ax.legend(prop=font_prop, **kwargs)


def set_chinese_labels(ax, title=None, xlabel=None, ylabel=None):
    """一次性设置所有中文标签"""
    if title:
        chinese_title(ax, title)
    if xlabel:
        chinese_xlabel(ax, xlabel)
    if ylabel:
        chinese_ylabel(ax, ylabel)
    return ax


def chinese_tick_labels(ax, axis='both'):
    """设置坐标轴刻度标签为中文"""
    font_prop = get_chinese_font_prop()

    if axis in ['x', 'both']:
        for tick in ax.get_xticklabels():
            tick.set_fontproperties(font_prop)

    if axis in ['y', 'both']:
        for tick in ax.get_yticklabels():
            tick.set_fontproperties(font_prop)

    return ax


# 兼容旧代码的别名
def setup_better_chinese_font():
    return setup_chinese_font()


def get_font_prop():
    return get_chinese_font_prop()


# 导出常用变量
FONT_PROP = None  # 将在第一次调用时初始化


def apply_plot_style(ax):
    """应用绘图样式"""
    if ax is not None:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)
    return ax


def debug_font_info():
    """调试信息：显示当前字体配置"""
    info = []
    info.append("=== 字体调试信息 ===")

    # 显示字体文件路径
    font_path = st.session_state.get('chinese_font_path', '未设置')
    info.append(f"字体文件路径: {font_path}")

    if font_path != '未设置' and os.path.exists(font_path):
        info.append(f"✓ 字体文件存在")
        info.append(f"  文件大小: {os.path.getsize(font_path) / 1024 / 1024:.1f} MB")
    else:
        info.append(f"✗ 字体文件不存在!")

    # 显示 matplotlib 配置
    info.append(f"\nMatplotlib 配置:")
    info.append(f"  版本: {plt.matplotlib.__version__}")
    info.append(f"  font.family: {plt.rcParams.get('font.family', 'N/A')}")
    info.append(f"  font.sans-serif: {plt.rcParams.get('font.sans-serif', [])[:3]}...")  # 只显示前3个
    info.append(f"  axes.unicode_minus: {plt.rcParams.get('axes.unicode_minus', 'N/A')}")

    # 显示已注册的中文字体
    info.append(f"\n已注册的字体名称:")
    if 'font_name' in st.session_state:
        info.append(f"  {st.session_state['font_name']}")

    # 检查字体管理器中的字体
    try:
        available_fonts = [f.name for f in fm.fontManager.ttflist if 'Noto' in f.name or 'CJK' in f.name]
        if available_fonts:
            info.append(f"\n字体管理器中的相关字体:")
            for font in available_fonts[:5]:  # 只显示前5个
                info.append(f"  - {font}")
    except:
        pass

    return "\n".join(info)


# 在模块加载时自动初始化
if __name__ != "__main__":
    # 当模块被导入时自动设置字体
    setup_chinese_font()