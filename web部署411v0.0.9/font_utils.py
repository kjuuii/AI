# font_utils.py - 全局字体补丁最终版
import os
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO)

# --- 核心代码：定义一个全局变量来存储字体属性 ---
CHINESE_FONT_PROP = None


@st.cache_resource
def setup_chinese_font():
    """
    在程序启动时运行，为 Matplotlib 设置一个全局的、可用的中文字体。
    这是解决所有后续绘图模块字体问题的关键。
    """
    global CHINESE_FONT_PROP
    # 如果已经设置过，就直接返回
    if CHINESE_FONT_PROP is not None:
        return True

    logging.info("--- 开始执行全局中文字体设置 ---")

    try:
        # 寻找服务器上可用的 Noto Sans CJK 字体
        # Streamlit Cloud 通过 packages.txt 安装后，字体通常在这个路径
        font_path = ""
        possible_paths = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"
        ]

        for path in possible_paths:
            if os.path.exists(path):
                font_path = path
                break

        # 如果在标准路径找不到，则尝试从项目内部寻找
        if not font_path:
            project_font_path = "fonts/NotoSansSC-VariableFont_wght.ttf"
            if os.path.exists(project_font_path):
                font_path = project_font_path

        if font_path:
            logging.info(f"成功找到可用的中文字体文件: {font_path}")

            # --- 最关键的步骤：设置全局字体 ---
            # 1. 将字体添加到 Matplotlib 的管理器中
            fm.fontManager.addfont(font_path)

            # 2. 创建字体属性对象
            CHINESE_FONT_PROP = FontProperties(fname=font_path)

            # 3. 强力设置 Matplotlib 的全局默认字体
            # 这会让所有不特别指定字体的绘图操作都使用这款字体
            plt.rcParams['font.family'] = CHINESE_FONT_PROP.get_name()
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

            logging.info(f"Matplotlib 全局字体已设置为: {CHINESE_FONT_PROP.get_name()}")
            logging.info("--- 全局中文字体设置完成 ---")
            return True
        else:
            logging.error("在服务器上未能找到任何可用的中文字体文件！")
            st.error("部署错误：服务器缺少中文字体文件，请检查`packages.txt`和字体文件路径。")
            return False

    except Exception as e:
        logging.error(f"设置全局字体时发生严重错误: {e}", exc_info=True)
        st.error(f"初始化中文字体环境时出错: {e}")
        return False


def get_font_prop():
    """
    提供一个统一的接口，让其他模块可以获取到已经配置好的字体属性。
    """
    if CHINESE_FONT_PROP is None:
        setup_chinese_font()
    return CHINESE_FONT_PROP


# --- 在模块被导入时，就立即执行字体设置 ---
setup_chinese_font()