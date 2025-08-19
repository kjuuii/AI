# update_log.py
import streamlit as st
import os

# Define the path to the log file relative to this script
LOG_FILE_PATH = "NEU Synthesis Core AI版本更新日志.txt"


def display_update_log():
    """
    Reads the update log file and displays its content in an expander.
    """
    if os.path.exists(LOG_FILE_PATH):
        try:
            with open(LOG_FILE_PATH, 'r', encoding='utf-8') as f:
                log_content = f.read()

            # Use an expander at the bottom of the page or sidebar
            with st.expander("📜 查看更新日志", expanded=False):
                st.text(log_content)
                # Note: Positioning exactly in the bottom-right corner might require
                # more complex HTML/CSS injection with st.markdown(unsafe_allow_html=True)
                # or custom components. This expander is a simpler approach.

        except FileNotFoundError:
            st.warning(f"更新日志文件未找到: {LOG_FILE_PATH}")
        except Exception as e:
            st.error(f"读取更新日志时出错: {e}")
    else:
        st.warning(f"更新日志文件不存在于预期路径: {LOG_FILE_PATH}")

# You can add more functions here if needed, e.g., for formatting the log