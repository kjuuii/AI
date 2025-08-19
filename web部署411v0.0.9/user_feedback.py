# user_feedback.py
import streamlit as st
import pandas as pd
import os
from datetime import datetime

FEEDBACK_FILE = "user_feedback.csv"


def display_feedback_section():
    """
    Displays a section for users to submit feedback and view previous feedback.
    """
    st.title("✉️ 用户反馈与讨论")
    st.markdown("我们重视您的意见！请在这里留下您的反馈、建议或遇到的问题。")

    # --- Feedback Submission Form ---
    st.subheader("提交新反馈")
    with st.form("feedback_form", clear_on_submit=True):
        feedback_type = st.selectbox("反馈类型", ["建议", "Bug报告", "问题咨询", "讨论", "其他"])
        feedback_text = st.text_area("反馈内容:", height=150, placeholder="请详细描述...")
        submitted = st.form_submit_button("提交反馈")

        if submitted:
            if feedback_text:
                save_feedback(feedback_type, feedback_text)
                st.success("感谢您的反馈！我们已收到。")
            else:
                st.warning("反馈内容不能为空。")

    st.markdown("---")

    # --- Display Previous Feedback (Optional Discussion Area) ---
    st.subheader("历史反馈 / 讨论区")
    feedback_df = load_feedback()
    if not feedback_df.empty:
        # Display in reverse chronological order (newest first)
        st.dataframe(feedback_df.sort_index(ascending=False), use_container_width=True)
    else:
        st.info("还没有用户反馈。")


def save_feedback(feedback_type, feedback_text):
    """Saves the submitted feedback to a CSV file."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Try to get username, default to 'Anonymous'
    username = st.session_state.get('username', 'Anonymous')

    new_feedback = pd.DataFrame({
        'Timestamp': [now],
        'User': [username],
        'Type': [feedback_type],
        'Feedback': [feedback_text]
    })

    try:
        if os.path.exists(FEEDBACK_FILE):
            existing_feedback = pd.read_csv(FEEDBACK_FILE)
            updated_feedback = pd.concat([existing_feedback, new_feedback], ignore_index=True)
        else:
            updated_feedback = new_feedback

        updated_feedback.to_csv(FEEDBACK_FILE, index=False, encoding='utf-8-sig')
    except Exception as e:
        st.error(f"保存反馈时出错: {e}")


def load_feedback():
    """Loads feedback from the CSV file."""
    if os.path.exists(FEEDBACK_FILE):
        try:
            return pd.read_csv(FEEDBACK_FILE)
        except Exception as e:
            st.error(f"加载历史反馈时出错: {e}")
            return pd.DataFrame()  # Return empty dataframe on error
    else:
        return pd.DataFrame()  # Return empty dataframe if file doesn't exist