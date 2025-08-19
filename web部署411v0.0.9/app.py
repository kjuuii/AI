# -*- coding: utf-8 -*-
# 第一个 Streamlit 命令必须是 st.set_page_config
import streamlit as st
st.set_page_config(
    page_title="Synthesis Core AI | NEU",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
import json
import hashlib
import sqlite3
import atexit # 用于确保数据库连接关闭
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt # Matplotlib might be used within modules, keep for now
from datetime import datetime # 确保导入 datetime
import base64
# from PIL import Image # PIL might be used within modules, keep for now
# import io # io might be used within modules, keep for now
# import streamlit.components.v1 as components # Removed as animation is gone
import time # Keep for spinners
import traceback # For detailed error logging
import matplotlib.pyplot as plt
from font_utils import setup_chinese_font

# --- 新增：从 font_utils 导入字体设置 ---
try:
    from font_utils import FONT_PROP, apply_plot_style # 导入 apply_plot_style (如果也移过去了)
    print("Font utils loaded successfully from font_utils.py")
except ImportError as e:
    print(f"错误：无法加载 font_utils.py: {e}")
    # 定义一个备用的 FONT_PROP 和 apply_plot_style
    FONT_PROP = None
    def apply_plot_style(ax): return ax # 占位函数
# --- 结束新增 ---

setup_chinese_font()

# --- 移除 app.py 中的字体设置函数和调用 ---
# def setup_better_chinese_font(): ...
# FONT_PROP = setup_better_chinese_font() # 移除这一行

# --- 导入模块 ---
# (保持这部分不变, 但要确保 data_evaluator 和 data_processing 在下面)
# 确保所有需要的模块都存在或已创建占位符
try:
    import clustering
except ImportError as e:
    print(f"模块 clustering.py 导入失败: {e}")
    clustering = None
try:
    import clustering_tutorial
    TUTORIAL_MODULE_LOADED = True
except ImportError as e:
    print(f"模块 clustering_tutorial.py 导入失败: {e}")
    clustering_tutorial = None
    TUTORIAL_MODULE_LOADED = False
try:
    import classification_training
    CLASSIFICATION_TRAINING_LOADED = True
except ImportError as e:
    print(f"模块 classification_training.py 导入失败: {e}")
    classification_training = None
    CLASSIFICATION_TRAINING_LOADED = False
try:
    import classification_tutorial
    CLASSIFICATION_TUTORIAL_MODULE_LOADED = True
except ImportError as e:
    print(f"模块 classification_tutorial.py 导入失败: {e}")
    classification_tutorial = None
    CLASSIFICATION_TUTORIAL_MODULE_LOADED = False
try:
    import classification_validation
except ImportError as e:
    print(f"模块 classification_validation.py 导入失败: {e}")
    classification_validation = None
try:
    import regression_training
    REGRESSION_TRAINING_LOADED = True
    print("Successfully imported regression_training module.")
except ImportError as e:
    print(f"模块 regression_training.py 导入失败 (ImportError): {e}")
    print(f"详细错误信息: {traceback.format_exc()}")
    regression_training = None
    REGRESSION_TRAINING_LOADED = False
except SyntaxError as e:
    print(f"模块 regression_training.py 存在语法错误: {e}")
    print(f"错误位置: 文件 {e.filename}, 第 {e.lineno} 行")
    regression_training = None
    REGRESSION_TRAINING_LOADED = False
except Exception as e:
    print(f"模块 regression_training.py 导入时发生未知错误: {e}")
    print(f"错误类型: {type(e).__name__}")
    print(f"详细错误: {traceback.format_exc()}")
    regression_training = None
    REGRESSION_TRAINING_LOADED = False
try:
    import regression_tutorial
    REGRESSION_TUTORIAL_MODULE_LOADED = True
except ImportError as e:
    print(f"模块 regression_tutorial.py 导入失败: {e}")
    regression_tutorial = None
    REGRESSION_TUTORIAL_MODULE_LOADED = False
try:
    import regression_validation
except ImportError as e:
    print(f"模块 regression_validation.py 导入失败: {e}")
    regression_validation = None
try:
    import outlier_detection
except ImportError as e:
    print(f"模块 outlier_detection.py 导入失败: {e}")
    outlier_detection = None
try:
    import missing_value_handler
except ImportError as e:
    print(f"模块 missing_value_handler.py 导入失败: {e}")
    missing_value_handler = None
try:
    import data_balancer
except ImportError as e:
    print(f"模块 data_balancer.py 导入失败: {e}")
    data_balancer = None
try:
    import data_evaluator
    DATA_EVALUATOR_LOADED = True
    print("Successfully imported data_evaluator module.") # Add print statement for confirmation
except ImportError as e:
    print(f"错误：无法导入 'data_evaluator' 模块。详细错误: {e}")
    data_evaluator = None # Set to None if import fails
    DATA_EVALUATOR_LOADED = False
except Exception as e: # Catch other potential errors during import
    print(f"错误：导入 'data_evaluator' 模块时发生意外错误: {e}")
    import traceback
    print(traceback.format_exc()) # Print full traceback for debugging
    data_evaluator = None
    DATA_EVALUATOR_LOADED = False
try:
    import data_reduction
    DATA_REDUCTION_LOADED = True
    print("Successfully imported data_reduction module.")
except ImportError as e:
    print(f"错误：无法导入 'data_reduction' 模块。详细错误: {e}")
    data_reduction = None
    DATA_REDUCTION_LOADED = False
except Exception as e:
    print(f"错误：导入 'data_reduction' 模块时发生意外错误: {e}")
    import traceback
    print(traceback.format_exc())
    data_reduction = None
    DATA_REDUCTION_LOADED = False
try:
    from update_log import display_update_log
    UPDATE_LOG_LOADED = True
except ImportError as e:
    print(f"模块 update_log.py 导入失败: {e}")
    display_update_log = lambda: None # Placeholder
    UPDATE_LOG_LOADED = False

try:
    from user_feedback import display_feedback_section
    FEEDBACK_MODULE_LOADED = True
except ImportError as e:
    print(f"模块 user_feedback.py 导入失败: {e}")
    # Placeholder function if import fails
    def display_feedback_section():
        st.title("✉️ 用户反馈与讨论")
        st.error("用户反馈模块加载失败。")
    FEEDBACK_MODULE_LOADED = False
# --- 新增：导入数据处理模块 ---
try:
    import data_processing
    DATA_PROCESSING_LOADED = True
    print("Successfully imported data_processing module.")
except ImportError as e:
    print(f"错误：无法导入 'data_processing' 模块。详细错误: {e}")
    data_processing = None
    DATA_PROCESSING_LOADED = False
except Exception as e:
    print(f"错误：导入 'data_processing' 模块时发生意外错误: {e}")
    import traceback
    print(traceback.format_exc())
    data_processing = None
    DATA_PROCESSING_LOADED = False

# --- 新增：导入特征提取模块 ---
try:
    import feature_extraction
    FEATURE_EXTRACTION_LOADED = True
    print("Successfully imported feature_extraction module.")
except ImportError as e:
    print(f"错误：无法导入 'feature_extraction' 模块。详细错误: {e}")
    feature_extraction = None
    FEATURE_EXTRACTION_LOADED = False
except Exception as e:
    print(f"错误：导入 'feature_extraction' 模块时发生意外错误: {e}")
    import traceback
    print(traceback.format_exc())
    feature_extraction = None
    FEATURE_EXTRACTION_LOADED = False
# --- 结束新增 ---


DATABASE_FILE = 'users.db'

def get_db_connection():
    """创建并返回数据库连接"""
    conn = sqlite3.connect(DATABASE_FILE, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row # 让查询结果可以通过列名访问
    return conn


def save_session_state(user_id):
    """保存会话状态到数据库和文件系统"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 创建用户数据目录
    user_data_dir = f"user_data/{user_id}"
    os.makedirs(user_data_dir, exist_ok=True)

    # 处理复杂对象：保存到文件系统
    complex_objects = {}

    # 处理classification_data (DataFrame或字典)
    if 'classification_data' in st.session_state and st.session_state.classification_data is not None:
        data_path = f"{user_data_dir}/classification_data.pkl"
        try:
            import joblib
            joblib.dump(st.session_state.classification_data, data_path)
            complex_objects['classification_data_path'] = data_path
        except Exception as e:
            print(f"保存classification_data时出错: {e}")

    # 将相关会话状态转换为JSON (排除复杂对象)
    state_to_save = {
        'current_page': st.session_state.current_page,
        'selected_input_columns': st.session_state.get('selected_input_columns', []),
        'selected_output_column': st.session_state.get('selected_output_column'),
        'complex_objects': complex_objects,
        # 添加其他需要保存的状态...
        'data_source_type': st.session_state.get('data_source_type', 'file'),
        'normalize_features': st.session_state.get('normalize_features', True),
        'test_size': st.session_state.get('test_size', 0.2),
        'current_model_type': st.session_state.get('current_model_type', 'catboost'),
    }

    # 序列化为JSON
    state_json = json.dumps(state_to_save)

    # 保存或更新会话状态
    cursor.execute('''
        INSERT OR REPLACE INTO user_sessions (user_id, session_data, updated_at)
        VALUES (?, ?, ?)
    ''', (user_id, state_json, get_current_time()))

    conn.commit()
    conn.close()


def load_session_state(user_id):
    """从数据库和文件系统加载会话状态"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT session_data FROM user_sessions WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        state_json = row[0]
        state_data = json.loads(state_json)

        # 恢复基本状态到session_state
        for key, value in state_data.items():
            if key != 'complex_objects':  # 跳过复杂对象标记
                st.session_state[key] = value

        # 恢复复杂对象
        if 'complex_objects' in state_data:
            complex_objects = state_data['complex_objects']

            # 恢复classification_data
            if 'classification_data_path' in complex_objects:
                try:
                    import joblib
                    data_path = complex_objects['classification_data_path']
                    if os.path.exists(data_path):
                        st.session_state.classification_data = joblib.load(data_path)
                except Exception as e:
                    print(f"加载classification_data时出错: {e}")

        return True

    return False


def init_db():
    """初始化数据库，创建用户表和会话表"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # 用户表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        ''')

        # 会话表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                user_id INTEGER PRIMARY KEY,
                session_data TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')

        cursor.execute('CREATE INDEX IF NOT EXISTS idx_username ON users (username)')
        conn.commit()
        conn.close()
        print("Database initialized.")
    except Exception as e:
        print(f"Error initializing database: {e}")

# 在应用启动时初始化数据库 (确保只执行一次)
if 'db_initialized' not in st.session_state:
    init_db()
    st.session_state.db_initialized = True

# ============= 帮助函数 (保持不变) =============
def get_current_time():
    """返回当前时间字符串"""
    return datetime.now().isoformat()

def hash_password(password, salt):
    """密码哈希处理"""
    password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return password_hash

def save_credentials(username, remember):
    """保存用户凭证 (只保存用户名和记住状态)"""
    settings = {}
    if os.path.exists("settings.json"):
        try:
            with open("settings.json", 'r', encoding='utf-8') as f:
                settings = json.load(f)
        except Exception as e:
            print(f"加载 settings.json 错误: {e}")
            settings = {}
    if 'saved_credentials' not in settings:
        settings['saved_credentials'] = {}
    if remember:
        settings['saved_credentials']['username'] = username
        settings['saved_credentials']['remember'] = True
    else:
        if 'saved_credentials' in settings:
            settings['saved_credentials'] = {}
    try:
        with open("settings.json", 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"写入 settings.json 错误: {e}")

def load_saved_credentials():
    """加载保存的凭证 (只加载用户名和记住状态)"""
    username = ""; remember = False
    if os.path.exists("settings.json"):
        try:
            with open("settings.json", 'r', encoding='utf-8') as f:
                settings = json.load(f)
            if 'saved_credentials' in settings:
                if 'username' in settings['saved_credentials']:
                    username = settings['saved_credentials']['username']
                if 'remember' in settings['saved_credentials']:
                    remember = settings['saved_credentials']['remember']
                if not remember:
                    username = ""
        except Exception as e:
            print(f"加载 settings.json 错误: {e}")
            username = ""; remember = False
    return username, remember

def authenticate_user(username, password):
    """验证用户登录 (使用SQLite)"""
    if not username or not password:
        return False, "请输入用户名和密码", None
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, password_hash, salt FROM users WHERE username = ?", (username,))
        user_row = cursor.fetchone()
        if user_row:
            user_id, stored_hash, salt = user_row['id'], user_row['password_hash'], user_row['salt']
            hashed_password = hash_password(password, salt)
            if hashed_password == stored_hash:
                return True, f"欢迎回来，{username}！", user_id
            else:
                return False, "无效的用户名或密码", None
        else:
            return False, "无效的用户名或密码", None
    except sqlite3.Error as e:
        print(f"数据库认证错误: {e}")
        return False, f"认证时发生数据库错误: {e}", None
    except Exception as e:
        print(f"认证时发生未知错误: {e}")
        return False, f"认证时发生未知错误: {e}", None
    finally:
        if conn:
            conn.close()

def register_user(username, password, confirm_password):
    """注册新用户 (使用SQLite) - 修正了 except 块的语法"""
    if not username or not password:
        return False, "请填写所有字段"
    if password != confirm_password:
        return False, "密码不匹配"
    if len(password) < 6:
        return False, "密码至少应包含6个字符"

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            return False, "用户名已存在"
        salt = os.urandom(16).hex()
        hashed_password = hash_password(password, salt)
        created_time = get_current_time()
        cursor.execute('''
            INSERT INTO users (username, password_hash, salt, created_at)
            VALUES (?, ?, ?, ?)
        ''', (username, hashed_password, salt, created_time))
        conn.commit()
        return True, "注册成功！您现在可以登录了。"
    except sqlite3.IntegrityError:
        # 这个错误理论上在上面的 SELECT 检查后不应发生，但作为保险
        return False, "用户名已存在 (数据库约束)"
    except sqlite3.Error as e:
        # --- 修改开始 ---
        print(f"数据库注册错误: {e}") # 打印详细错误
        if conn: # 检查 conn 是否已成功创建
            conn.rollback() # 如果出错则回滚
        return False, f"注册时发生数据库错误: {e}"
        # --- 修改结束 ---
    except Exception as e:
        # --- 修改开始 ---
        print(f"注册时发生未知错误: {e}") # 打印详细错误
        if conn:
           conn.rollback() # 回滚
        return False, f"注册时发生未知错误: {e}"
        # --- 修改结束 ---
    finally:
        if conn:
            conn.close()

# ============= 样式和资源 (保持不变) =============
def load_css():
    """加载简化的CSS样式"""
    # --- CSS 代码保持不变 ---
    # (CSS code from the original file - too long to repeat here)
    return """
    <style>
    /* 全局字体和样式 */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
        box-sizing: border-box;
    }

    /* 主要色彩变量 */
    :root {
        --primary-color: #4F46E5; /* Indigo */
        --primary-light: rgba(79, 70, 229, 0.1);
        --primary-dark: #3730A3;
        --secondary-color: #06B6D4; /* Cyan */
        --secondary-light: rgba(6, 182, 212, 0.1);
        --accent-color: #A855F7; /* Purple */
        --success-color: #10B981; /* Emerald */
        --error-color: #EF4444; /* Red */
        --warning-color: #F59E0B; /* Amber */
        --info-color: #3B82F6; /* Blue */
        /* --- 新增: 为评估消息添加浅色背景 --- */
        --info-color-light: rgba(59, 130, 246, 0.08);
        --warning-color-light: rgba(245, 158, 11, 0.08);
        --error-color-light: rgba(239, 68, 68, 0.08);
        --success-color-light: rgba(16, 185, 129, 0.08);
        /* --- 新增: 为评估消息添加深色文本 --- */
        --info-color-dark: #1E40AF;
        --warning-color-dark: #92400E;
        --error-color-dark: #991B1B;
        --success-color-dark: #065F46;
        /* --- 结束新增 --- */
        --text-primary: #1E293B; /* Slate 800 */
        --text-secondary: #475569; /* Slate 600 */
        --text-muted: #94A3B8; /* Slate 400 */
        --border-color: #E2E8F0; /* Slate 200 */
        --bg-color: #F8FAFC; /* Slate 50 */
        --card-bg: #FFFFFF;
        --sidebar-bg-start: #1F2937; /* Slate 800 */
        --sidebar-bg-end: #111827; /* Slate 900 */
        --sidebar-text: #D1D5DB; /* Gray 300 */
        --sidebar-text-hover: #FFFFFF;
        --sidebar-icon: #9CA3AF; /* Gray 400 */
        --sidebar-border: #374151; /* Slate 700 */
        --sidebar-hover-bg: rgba(255, 255, 255, 0.05);
        --sidebar-active-bg: rgba(79, 70, 229, 0.15); /* Primary light */
        --sidebar-active-text: var(--primary-color);
        --sidebar-expander-line: var(--primary-color);

        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    }

    /* 重置 Streamlit 默认样式 */
    .stApp { background-color: var(--bg-color); }
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none !important; }
    .stApp > header { background-color: transparent; }

    /* 主内容区域边距调整 */
    .main .block-container {
        padding-top: 1.5rem !important; /* 稍微增加顶部内边距 */
        padding-bottom: 3rem !important;
        padding-left: 2rem !important; /* 稍微增加左右内边距 */
        padding-right: 2rem !important;
        max-width: 100% !important;
    }

    /* 登录页面特定样式 (保持不变) */
    .app-header { text-align: center; padding: 2.5rem 0 1.5rem 0; }
    .app-title { font-size: 3.5rem; font-weight: 800; background: linear-gradient(135deg, var(--primary-color), var(--accent-color)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; color: transparent; line-height: 1.1; margin-bottom: 0.75rem; letter-spacing: -1px; animation: fadeInUp 0.8s ease-out; }
    .app-subtitle { font-size: 1.25rem; color: var(--text-secondary); font-weight: 500; margin-bottom: 1.5rem; animation: fadeInUp 0.8s ease-out 0.2s both; }
    .auth-card { max-width: 420px; margin: 0 auto; background-color: var(--card-bg); border-radius: 16px; overflow: hidden; box-shadow: var(--shadow-xl); transition: all 0.3s ease; animation: fadeInUp 0.8s ease-out 0.4s both; position: relative; border: 1px solid rgba(226, 232, 240, 0.6); }
    .auth-card-header { position: relative; padding: 2rem 2rem 1.5rem 2rem; text-align: center; }
    .auth-card-header::after { content: ''; position: absolute; bottom: 0; left: 10%; right: 10%; height: 1px; background: linear-gradient(90deg, transparent, rgba(226, 232, 240, 0.6), transparent); }
    .auth-card-body { padding: 1.5rem 2rem 2rem 2rem; }
    .auth-title { font-size: 1.75rem; font-weight: 700; color: var(--text-primary); margin-bottom: 0.5rem; }
    .auth-subtitle { font-size: 1rem; color: var(--text-secondary); margin-bottom: 0.5rem; }
    .auth-separator { display: flex; align-items: center; margin: 1.5rem 0; }
    .auth-separator::before, .auth-separator::after { content: ''; flex: 1; height: 1px; background: linear-gradient(90deg, transparent, var(--border-color), transparent); }
    .auth-separator-text { padding: 0 1rem; color: var(--text-muted); font-size: 0.875rem; font-weight: 500; }
    .features-container { display: flex; justify-content: center; gap: 1rem; margin-top: 0.5rem; margin-bottom: 1rem; animation: fadeInUp 0.8s ease-out 0.6s both; }
    .feature-card { background-color: white; padding: 1.5rem; border-radius: 12px; width: 100px; display: flex; flex-direction: column; align-items: center; box-shadow: var(--shadow-md); transition: all 0.3s ease; }
    .feature-card:hover { transform: translateY(-5px); box-shadow: var(--shadow-lg); }
    .feature-icon { width: 48px; height: 48px; display: flex; align-items: center; justify-content: center; background: linear-gradient(135deg, var(--primary-light), var(--secondary-light)); border-radius: 12px; margin-bottom: 0.75rem; font-size: 1.5rem; }
    .feature-title { font-size: 0.875rem; font-weight: 600; color: var(--text-secondary); text-align: center; }
    .app-footer { width: 100%; text-align: center; padding: 2rem 0 1rem 0; font-size: 0.8rem; color: var(--text-muted); animation: fadeIn 1s ease-out; }
    .forgot-password { text-align: right; margin-bottom: 1rem; margin-top: 0.5rem; }
    .forgot-password a { color: var(--primary-color); font-size: 0.875rem; text-decoration: none; transition: color 0.2s ease; }
    .forgot-password a:hover { color: var(--primary-dark); text-decoration: underline; }

    /* 登录/注册按钮样式 */
    div[data-testid="stButton"] > button { width: 100%; background: linear-gradient(135deg, var(--primary-color), var(--accent-color)) !important; color: white !important; font-weight: 600 !important; padding: 0.75rem 1.5rem !important; border-radius: 8px !important; border: none !important; cursor: pointer !important; transition: all 0.3s ease !important; box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25) !important; font-size: 1rem !important; height: auto !important; margin-top: 0.5rem !important; }
    div[data-testid="stButton"] > button:hover { transform: translateY(-2px) !important; box-shadow: 0 6px 16px rgba(79, 70, 229, 0.35) !important; }
    div[data-testid="stButton"] > button:active { transform: translateY(0) !important; box-shadow: 0 2px 8px rgba(79, 70, 229, 0.2) !important; }
    div.auth-card-body div[data-testid="stButton"]:nth-of-type(2) > button, div.auth-card-body div[data-testid="stButton"]:nth-of-type(3) > button { background: transparent !important; color: var(--primary-color) !important; box-shadow: none !important; border: 1px solid var(--primary-color) !important; margin-top: 0.75rem !important; }
    div.auth-card-body div[data-testid="stButton"]:nth-of-type(2) > button:hover, div.auth-card-body div[data-testid="stButton"]:nth-of-type(3) > button:hover { background-color: var(--primary-light) !important; transform: translateY(-1px) !important; }

    /* 输入框样式 */
    div[data-testid="stTextInput"] > div[data-testid="stWidgetLabel"] ~ div { border: none !important; box-shadow: none !important; padding: 0 !important; }
    div[data-testid="stTextInput"] div[data-baseweb="base-input"] { background-color: #F9FAFB !important; border: 1px solid #E2E8F0 !important; border-radius: 8px !important; padding: 0.5rem 1rem !important; transition: all 0.3s ease !important; }
    div[data-testid="stTextInput"] div[data-baseweb="base-input"]:focus-within { background-color: white !important; border-color: var(--primary-color) !important; box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15) !important; }
    div[data-testid="stTextInput"] input { font-size: 0.95rem !important; color: var(--text-primary) !important; padding-left: 0 !important; }
    div[data-testid="stVerticalBlock"] > div:not(:last-child) { margin-bottom: 0.75rem !important; }
    .stCheckbox > div > div > div { display: flex !important; align-items: center !important; }
    .stCheckbox > div > div > div > label { font-size: 0.9rem !important; color: var(--text-secondary) !important; margin-left: 0.5rem; }

    /* 动画效果 */
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    @keyframes fadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }

    /* 响应式调整 */
    @media (max-width: 768px) {
        .app-title { font-size: 2.5rem; } .app-subtitle { font-size: 1rem; }
        .auth-card { max-width: 90%; margin: 1rem auto; }
        .auth-card-header, .auth-card-body, .auth-card-footer { padding-left: 1.5rem; padding-right: 1.5rem; }
        .features-container { flex-wrap: wrap; } .feature-card { width: 80px; padding: 1rem; }
        .feature-icon { width: 40px; height: 40px; font-size: 1.25rem; }
        .main .block-container { padding-left: 1rem !important; padding-right: 1rem !important; } /* 调整主内容区左右边距 */
    }

    /* ================================= */
    /* === 主应用侧边栏样式 (优化) === */
    /* ================================= */
     [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, var(--sidebar-bg-start) 0%, var(--sidebar-bg-end) 100%);
        border-right: 1px solid var(--sidebar-border);
        padding-top: 1rem; /* 增加顶部内边距 */
        display: flex; /* 使用 Flexbox 布局 */
        flex-direction: column; /* 垂直排列 */
        height: 100%; /* 占满整个侧边栏高度 */
    }
    /* 侧边栏 Logo/标题区域 */
    .sidebar-title-area {
        text-align: center;
        padding-bottom: 1.5rem; /* 增加底部间距 */
        margin-bottom: 1rem; /* 与下方元素间距 */
        border-bottom: 1px solid var(--sidebar-border);
        flex-shrink: 0; /* 防止标题区域被压缩 */
    }
    .sidebar-title {
        color: #E5E7EB; /* Light Gray */
        font-weight: 600;
        font-size: 1.3rem; /* 稍微增大 */
        letter-spacing: 1px;
        display: flex; /* 使用 flex 布局对齐图标和文字 */
        align-items: center;
        justify-content: center; /* 居中 */
        gap: 0.5rem; /* 图标和文字间距 */
    }
    .sidebar-title-icon {
        font-size: 1.5rem; /* 图标大小 */
        line-height: 1; /* 修正对齐 */
    }

    /* 侧边栏导航主体区域 */
    .sidebar-nav-container {
        flex-grow: 1; /* 占据剩余空间 */
        overflow-y: auto; /* 内容过多时允许滚动 */
        padding-bottom: 1rem; /* 底部留白 */
    }


    /* 侧边栏导航标题 */
    .sidebar-nav-header {
        color: var(--text-muted);
        font-size: 0.8rem; /* 稍小 */
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 0.75rem 1rem 0.5rem 1rem; /* 调整内边距 */
        margin-bottom: 0.25rem;
    }

    /* 侧边栏 Expander 样式 */
    [data-testid="stExpander"] summary {
        background-color: transparent !important;
        border: none !important;
        padding: 0.6rem 1rem !important;
        font-size: 0.95rem !important; /* 稍小 */
        font-weight: 500 !important;
        color: var(--sidebar-text) !important;
        transition: background-color 0.2s ease, color 0.2s ease;
        border-radius: 6px;
        margin-bottom: 2px;
        display: flex; /* 使用 Flex 对齐图标和文字 */
        align-items: center;
        gap: 0.6rem; /* 图标和文字间距 */
    }
    [data-testid="stExpander"] summary svg { /* Expander 箭头图标 */
        fill: var(--sidebar-icon) !important;
        transition: fill 0.2s ease;
        order: -1; /* 将箭头图标移到前面 */
        margin-right: 0.3rem; /* 调整箭头和图标间距 */
    }
    [data-testid="stExpander"] summary:hover {
        background-color: var(--sidebar-hover-bg) !important;
        color: var(--sidebar-text-hover) !important;
    }
    [data-testid="stExpander"] summary:hover svg {
        fill: var(--sidebar-text-hover) !important;
    }
    /* Expander 内容区域 */
    [data-testid="stExpander"] [data-testid="stVerticalBlock"] {
         border-left: 2px solid var(--sidebar-expander-line);
         margin-left: 1.5rem; /* 增加缩进 */
         padding-left: 0.8rem;
         padding-top: 0.3rem;
         padding-bottom: 0.3rem;
    }

    /* 侧边栏按钮通用样式 (包括 Expander 内和顶级按钮) */
    /* 修正：移除按钮内的图标 span，直接使用 emoji */
    [data-testid="stSidebar"] .stButton button {
        background-color: transparent !important;
        color: var(--sidebar-text) !important;
        font-weight: 400 !important;
        padding: 0.5rem 1rem !important; /* 统一内边距 */
        border-radius: 4px !important;
        text-align: left !important;
        font-size: 0.9rem !important;
        width: 100% !important;
        border: none !important;
        box-shadow: none !important;
        margin-top: 1px !important;
        margin-bottom: 1px !important;
        transition: background-color 0.2s ease, color 0.2s ease;
        display: flex; /* Flex 对齐 */
        align-items: center;
        gap: 0.6rem; /* 图标和文字间距 */
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background-color: var(--sidebar-hover-bg) !important;
        color: var(--sidebar-text-hover) !important;
        transform: none !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] .stButton button p { /* 按钮内文字 */
         text-align: left !important;
         width: 100%;
         margin: 0; /* 移除默认边距 */
         line-height: 1.4; /* 调整行高 */
    }
    /* 移除按钮图标 span 的样式 */
    /* [data-testid="stSidebar"] .stButton button .button-icon { ... } */


    /* 顶级按钮 (Home) 的特殊样式 */
    [data-testid="stSidebar"] .stButton[key^="nav_home"] button { /* 使用 key 前缀匹配 */
        font-weight: 500 !important;
        padding: 0.6rem 1rem !important; /* 稍微大一点的内边距 */
        margin-bottom: 4px !important; /* 与下方 Expander 间距 */
    }

    /* 登出按钮区域 */
    .sidebar-logout-area {
        margin-top: auto; /* 将此区域推到底部 */
        padding: 1rem; /* 内边距 */
        border-top: 1px solid var(--sidebar-border); /* 分隔线 */
        flex-shrink: 0; /* 防止被压缩 */
    }

    /* 登出按钮 */
     [data-testid="stSidebar"] .stButton[key="nav_logout"] button {
        background-color: rgba(239, 68, 68, 0.1) !important;
        color: #FCA5A5 !important;
        font-weight: 500 !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
        border-radius: 6px !important;
        /* margin-top: auto !important; */ /* 从这里移除 */
        text-align: center !important;
        padding: 0.6rem 1rem !important;
        justify-content: center; /* 居中内容 */
        margin-top: 0 !important; /* 移除顶部外边距 */
        margin-bottom: 0.5rem !important; /* 与版本信息间距 */
     }
      [data-testid="stSidebar"] .stButton[key="nav_logout"] button:hover {
        background-color: rgba(239, 68, 68, 0.2) !important;
        color: #FECACA !important;
     }
    [data-testid="stSidebar"] .stButton[key="nav_logout"] button p {
         text-align: center !important;
         width: auto; /* 恢复自动宽度 */
    }
     /* 移除登出按钮图标 span 的样式 */
     /* [data-testid="stSidebar"] .stButton[key="nav_logout"] button .button-icon { ... } */
     /* [data-testid="stSidebar"] .stButton[key="nav_logout"] button:hover .button-icon { ... } */

    /* 侧边栏底部版本信息 */
    .sidebar-footer {
        /* position: absolute; */ /* 不再需要绝对定位 */
        /* bottom: 10px; */
        /* left: 0; */
        /* right: 0; */
        text-align: center;
        font-size: 0.75rem;
        color: var(--text-muted);
        /* padding: 0 1rem; */ /* 从这里移除 */
    }

    /* ===================================== */
    /* === 主应用通用样式 (保持不变) === */
    /* ===================================== */
    .home-container { margin-top: 2rem; text-align: center; padding-top: 5rem; padding-bottom: 5rem; }
    .home-title { font-size: 2.8rem; font-weight: 700; color: var(--text-primary); background: linear-gradient(120deg, var(--primary-dark), var(--secondary-color)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; color: transparent; margin-bottom: 0.5rem; }
    .home-subtitle { font-size: 1.3rem; color: var(--text-secondary); margin-bottom: 1.5rem; font-weight: 500; }
    .home-instruction { font-size: 1.1rem; color: var(--text-muted); margin-bottom: 2.5rem; }
    .home-icon-area { font-size: 2.5rem; opacity: 0.8; }
    h1, h2, h3, h4 { color: var(--text-primary); }
    /* Streamlit Alert 样式 */
    .stAlert[data-baseweb="alert"] { border-radius: 8px !important; border-width: 1px !important; border-left-width: 4px !important; border-style: solid !important; padding-top: 0.8rem !important; padding-bottom: 0.8rem !important; }
    .stAlert[data-baseweb="alert"] > div:first-child { padding-right: 0.8rem !important; padding-left: 0.5rem !important; }
    .stAlert[data-baseweb="alert"] div[data-testid="stNotificationContent"] { font-size: 0.9rem !important; line-height: 1.5 !important; }
    .stAlert[data-baseweb="alert"][kind="info"] { background-color: rgba(59, 130, 246, 0.05) !important; border-color: rgba(59, 130, 246, 0.3) !important; border-left-color: #3B82F6 !important; }
    .stAlert[data-baseweb="alert"][kind="info"] > div:first-child svg { color: #3B82F6 !important; fill: #3B82F6 !important; }
    .stAlert[data-baseweb="alert"][kind="info"] div[data-testid="stNotificationContent"] { color: #1E40AF !important; }
    .stAlert[data-baseweb="alert"][kind="error"] { background-color: rgba(239, 68, 68, 0.05) !important; border-color: rgba(239, 68, 68, 0.3) !important; border-left-color: var(--error-color) !important; }
    .stAlert[data-baseweb="alert"][kind="error"] > div:first-child svg { color: var(--error-color) !important; fill: var(--error-color) !important; }
    .stAlert[data-baseweb="alert"][kind="error"] div[data-testid="stNotificationContent"] { color: #991B1B !important; }
    .stAlert[data-baseweb="alert"][kind="success"] { background-color: rgba(16, 185, 129, 0.05) !important; border-color: rgba(16, 185, 129, 0.3) !important; border-left-color: var(--success-color) !important; }
    .stAlert[data-baseweb="alert"][kind="success"] > div:first-child svg { color: var(--success-color) !important; fill: var(--success-color) !important; }
    .stAlert[data-baseweb="alert"][kind="success"] div[data-testid="stNotificationContent"] { color: #065F46 !important; }
    .stAlert[data-baseweb="alert"][kind="warning"] { background-color: rgba(245, 158, 11, 0.05) !important; border-color: rgba(245, 158, 11, 0.3) !important; border-left-color: var(--warning-color) !important; }
    .stAlert[data-baseweb="alert"][kind="warning"] > div:first-child svg { color: var(--warning-color) !important; fill: var(--warning-color) !important; }
    .stAlert[data-baseweb="alert"][kind="warning"] div[data-testid="stNotificationContent"] { color: #92400E !important; } /* Dark Amber */

    /* 主页面按钮样式 */
    .main .block-container .stButton:not([key^='nav_']) button { /* Target buttons in main area, excluding sidebar nav buttons */ background: transparent !important; color: var(--primary-color) !important; box-shadow: none !important; border: 1px solid var(--primary-color) !important; font-weight: 500 !important; width: auto !important; padding: 0.5rem 1rem !important; margin-top: 0.5rem !important; margin-bottom: 0.5rem !important; }
    .main .block-container .stButton:not([key^='nav_']) button:hover { background-color: var(--primary-light) !important; transform: translateY(-1px) !important; box-shadow: none !important; border: 1px solid var(--primary-color) !important; color: var(--primary-dark) !important; }
    .main .block-container .stButton:not([key^='nav_']) button p { text-align: center !important; }

    /* --- 新增: 数据评估页面特定样式 --- */
    .eval-finding {
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
        border-left-width: 4px;
        border-left-style: solid;
        font-size: 0.9rem;
    }
    .eval-finding.info {
        background-color: var(--info-color-light, rgba(59, 130, 246, 0.08)); /* Use fallback */
        border-left-color: var(--info-color);
        color: var(--info-color-dark, #1E40AF);
    }
    .eval-finding.warning {
        background-color: var(--warning-color-light, rgba(245, 158, 11, 0.08));
        border-left-color: var(--warning-color);
        color: var(--warning-color-dark, #92400E);
    }
    .eval-finding.error {
        background-color: var(--error-color-light, rgba(239, 68, 68, 0.08));
        border-left-color: var(--error-color);
        color: var(--error-color-dark, #991B1B);
    }
     .eval-finding.recommendation {
        background-color: var(--success-color-light, rgba(16, 185, 129, 0.08));
        border-left-color: var(--success-color);
        color: var(--success-color-dark, #065F46);
    }
    .eval-finding.header {
        font-weight: 600;
        margin-top: 1rem;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid var(--border-color);
        color: var(--text-secondary);
    }
    /* --- 结束新增 --- */

    </style>
    """

# --- Login Page Functions (unchanged) ---
def create_app_header(): return """<div class="app-header"><h1 class="app-title">NEU Synthesis Core AI</h1><p class="app-subtitle">智能数据分析和预测平台</p></div>""" # (Keep unchanged)
def render_features(): return """<div class="features-container"><div class="feature-card"> <div class="feature-icon">🧠</div> <div class="feature-title">先进算法</div> </div><div class="feature-card"> <div class="feature-icon">📈</div> <div class="feature-title">精准预测</div> </div><div class="feature-card"> <div class="feature-icon">📊</div> <div class="feature-title">数据可视化</div> </div></div>""" # (Keep unchanged)
def render_footer(): return """<div class="app-footer">© 2025 NEU Synthesis Core AI | 版本 1.0.1</div>""" # (Keep unchanged)
def login_page():
    """显示简化后的登录页面 (保持不变)"""
    # (Keep login page code unchanged)
    st.markdown(load_css(), unsafe_allow_html=True)
    st.markdown(create_app_header(), unsafe_allow_html=True)
    st.markdown(render_features(), unsafe_allow_html=True)
    st.markdown('<div class="auth-card">', unsafe_allow_html=True)

    st.markdown('<div class="auth-card-header">', unsafe_allow_html=True)
    if not st.session_state.get("register_mode", False):
        st.markdown('<h3 class="auth-title">欢迎回来</h3>', unsafe_allow_html=True)
        st.markdown('<p class="auth-subtitle">登录您的账户访问AI分析平台</p>', unsafe_allow_html=True)
        st.markdown('</div><div class="auth-card-body">', unsafe_allow_html=True)
        saved_username, saved_remember = load_saved_credentials()
        username = st.text_input("用户名", value=saved_username, placeholder="输入您的用户名", key="login_username", label_visibility="collapsed")
        password = st.text_input("密码", type="password", value="", placeholder="输入您的密码", key="login_password", label_visibility="collapsed")
        col1, col2 = st.columns([3, 2])
        with col1: remember = st.checkbox("记住我", value=saved_remember, key="login_remember")
        with col2: st.markdown('<div class="forgot-password"><a href="#">忘记密码？</a></div>', unsafe_allow_html=True)
        if st.button("登 录", key="login_btn", use_container_width=True):
            with st.spinner("登录中..."):
                time.sleep(0.5)
                success, message, user_id = authenticate_user(username, password)
                if success:
                    save_credentials(username, remember)
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.user_id = user_id
                    st.session_state.current_page = "home"

                    # 加载上一次的会话状态
                    load_session_state(user_id)

                    # 设置URL参数，用于识别刷新
                    st.query_params['refresh'] = 'true'
                    st.query_params['page'] = 'home'
                    st.rerun()
                else: st.error(message)
        st.markdown('<div class="auth-separator"><span class="auth-separator-text">或</span></div>', unsafe_allow_html=True)
        if st.button("创建新账户", key="register_btn", use_container_width=True):
            st.session_state.register_mode = True; st.rerun()
    else: # 注册表单
        st.markdown('<h3 class="auth-title">创建账户</h3>', unsafe_allow_html=True)
        st.markdown('<p class="auth-subtitle">加入NEU Synthesis Core AI开始您的AI之旅</p>', unsafe_allow_html=True)
        st.markdown('</div><div class="auth-card-body">', unsafe_allow_html=True)
        new_username = st.text_input("用户名", placeholder="选择一个用户名", key="reg_username", label_visibility="collapsed")
        new_password = st.text_input("密码", type="password", placeholder="设置一个强密码（至少6个字符）", key="reg_password", label_visibility="collapsed")
        confirm_password = st.text_input("确认密码", type="password", placeholder="再次输入您的密码", key="reg_confirm", label_visibility="collapsed")
        terms = st.checkbox("我已阅读并同意服务条款和隐私政策", key="reg_terms")
        if st.button("注 册", key="signup_btn", disabled=not terms, use_container_width=True):
             with st.spinner("创建账户中..."):
                time.sleep(0.5)
                success, message = register_user(new_username, new_password, confirm_password)
                if success:
                    st.success(message); time.sleep(1.5); st.session_state.register_mode = False; st.rerun()
                else: st.error(message)
        st.markdown('<div class="auth-separator"><span class="auth-separator-text">或</span></div>', unsafe_allow_html=True)
        if st.button("返回登录", key="return_login_btn", use_container_width=True):
            st.session_state.register_mode = False; st.rerun()

    st.markdown('</div></div>', unsafe_allow_html=True) # 关闭 auth-card-body 和 auth-card
    st.markdown(render_footer(), unsafe_allow_html=True)


# ==================== 主应用界面部分 ====================

def setup_main_sidebar():
    """设置主应用的侧边栏 (包含数据处理)"""
    with st.sidebar:
        # --- Logo/Title Area ---
        st.markdown(
            """
            <div class="sidebar-title-area">
                <div class="sidebar-title">
                    <span class="sidebar-title-icon">🧠</span> NEU Synthesis AI
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # --- Navigation Tree ---
        with st.container():
            st.markdown("<h3 class='sidebar-nav-header'>导航菜单</h3>", unsafe_allow_html=True)

            # --- 主页 ---
            home_icon = "🏠"
            if st.button(f"{home_icon} 主页", use_container_width=True, key="nav_home"):
                st.session_state.current_page = "home"; st.rerun()

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True) # Spacer

            # --- 1. 数据分析 (保持不变，或考虑重命名/重组) ---
            analysis_icon = "🔍"
            is_analysis_page = st.session_state.current_page in ["outlier_detection", "missing_value", "data_balancing", "data_evaluation"]
            with st.expander(f"{analysis_icon} 数据分析", expanded=is_analysis_page):
                outlier_icon = "⚠️"; missing_icon = "🧩"; balance_icon = "⚖️"; eval_icon = "📝"
                if st.button(f"{outlier_icon} 异常点发现", use_container_width=True, key="nav_outlier_detection"):
                    st.session_state.current_page = "outlier_detection"; st.rerun()
                if st.button(f"{missing_icon} 缺失值处理", use_container_width=True, key="nav_missing_value"):
                    st.session_state.current_page = "missing_value"; st.rerun()
                if st.button(f"{balance_icon} 数据平衡处理", use_container_width=True, key="nav_data_balancing"):
                    st.session_state.current_page = "data_balancing"; st.rerun()
                if st.button(f"{eval_icon} 数据评估", use_container_width=True, key="nav_data_evaluation"):
                    st.session_state.current_page = "data_evaluation"; st.rerun()

            # --- 新增：数据处理 ---
            processing_icon = "🔧" # New icon for processing
            is_processing_page = st.session_state.current_page in ["data_processing", "feature_extraction"]
            with st.expander(f"{processing_icon} 数据处理", expanded=is_processing_page):
                 vis_icon = "📊" # Visualization icon
                 feature_icon = "🔍" # Feature extraction icon
                 dim_red_icon = "📉"
                 # Add button for the main data processing page
                 if st.button(f"{vis_icon} 数据可视化与分割", use_container_width=True, key="nav_data_processing"):
                      st.session_state.current_page = "data_processing"; st.rerun()
                 # Add button for feature extraction page
                 if st.button(f"{feature_icon} 特征提取", use_container_width=True, key="nav_feature_extraction"):
                     st.session_state.current_page = "feature_extraction"; st.rerun()
                 # 新增数据降维按钮
                 if st.button(f"{dim_red_icon} 数据降维", use_container_width=True, key="nav_data_reduction"):
                     st.session_state.current_page = "data_reduction";
                     st.rerun()


            # --- 2. 聚类分析 (保持不变) ---
            cluster_icon = "🔬"
            is_clustering_page = st.session_state.current_page == "clustering"
            with st.expander(f"{cluster_icon} 聚类分析", expanded=is_clustering_page):
                if st.button(f"🎯 数据聚类", use_container_width=True, key="nav_clustering"):
                    st.session_state.current_page = "clustering"; st.rerun()

            # --- 3. 分类模型 (保持不变) ---
            classify_icon = "📊"
            is_classification_page = st.session_state.current_page in ["classification_training", "classification_validation", "classification_migration"]
            with st.expander(f"{classify_icon} 分类模型", expanded=is_classification_page):
                train_icon = "💡"; eval_icon_cls = "📈"; migrate_icon = "✈️" # Use specific eval icon name
                if st.button(f"{train_icon} 分类训练", use_container_width=True, key="nav_classification_training"):
                    st.session_state.current_page = "classification_training"; st.rerun()
                if st.button(f"{eval_icon_cls} 分类评估", use_container_width=True, key="nav_classification_validation"):
                    st.session_state.current_page = "classification_validation"; st.rerun()
                if st.button(f"{migrate_icon} 模型迁移 (分类)", use_container_width=True, key="nav_classification_migration"):
                    st.session_state.current_page = "classification_migration"; st.rerun()

            # --- 4. 回归模型 (保持不变) ---
            regress_icon = "📈"
            is_regression_page = st.session_state.current_page in ["regression_training", "regression_validation", "regression_migration"]
            with st.expander(f"{regress_icon} 回归模型", expanded=is_regression_page):
                eval_icon_reg = "📉" # Use specific eval icon name
                if st.button(f"{train_icon} 回归训练", use_container_width=True, key="nav_regression_training"):
                    st.session_state.current_page = "regression_training"; st.rerun()
                if st.button(f"{eval_icon_reg} 回归验证", use_container_width=True, key="nav_regression_validation"):
                    st.session_state.current_page = "regression_validation"; st.rerun()
                if st.button(f"{migrate_icon} 模型迁移 (回归)", use_container_width=True, key="nav_regression_migration"):
                    st.session_state.current_page = "regression_migration"; st.rerun()

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)  # Spacer
            feedback_icon = "✉️"
            if st.button(f"{feedback_icon} 用户反馈", use_container_width=True, key="nav_feedback"):
                st.session_state.current_page = "user_feedback";
                st.rerun()

        # --- Footer Area (Logout and Version - 保持不变) ---
        with st.container():
            st.markdown("<div class='sidebar-logout-area'>", unsafe_allow_html=True)  # 包裹登出和版本
            logout_icon = "🚪"
            if st.button(f"{logout_icon} 登出", use_container_width=True, key="nav_logout"):

                # 1. 定义需要清除的 Session State Keys
                keys_to_clear = ['logged_in', 'username', 'user_id', 'current_page', 'register_mode']

                # 添加所有模块特定的前缀（确保覆盖所有模块的状态）
                module_prefixes = [
                    'cls_tut_', 'reg_tut_', 'tut_',  # 教程模块
                    'clf_', 'reg_', 'cluster_',  # 训练/验证/主模块 (假设有 cluster_ 前缀)
                    'mv_', 'db_', 'outlier_', 'de_', 'dp_'  # 数据处理/分析模块
                    # 根据你的实际情况添加或修改前缀
                ]
                for prefix in module_prefixes:
                    keys_to_clear.extend([k for k in st.session_state if k.startswith(prefix)])

                # 添加其他可能需要重置的关键状态
                keys_to_clear.extend([
                    'classification_data', 'regression_data', 'clustering_data',
                    'training_results', 'cv_results', 'validation_results',
                    'multi_validation_results', 'elbow_analysis_results',
                    'model_trained_flag', 'scaler', 'label_encoder',
                    'data_loaded', 'column_names', 'selected_input_columns',
                    'selected_output_column', 'data_source_type', 'file_names',
                    'has_group_column', 'selected_group_column'
                ])

                # 确保列表中的 key 唯一
                keys_to_clear = list(set(keys_to_clear))

                print("Logging out, clearing session state keys:", keys_to_clear)

                # 2. 清除 Session State
                for key in keys_to_clear:
                    if key in st.session_state:
                        try:
                            del st.session_state[key]
                        except Exception as e:
                            print(f"Error deleting session state key '{key}': {e}")

                # --- !! 重要：删除登录缓存文件 !! ---
                # 3. 删除登录缓存文件
                login_cache_file = "temp/.login_cache"
                if os.path.exists(login_cache_file):
                    try:
                        os.remove(login_cache_file)
                        print(f"Deleted login cache file: {login_cache_file}")
                    except OSError as e:
                        # 如果删除失败，最好告知用户，但仍然尝试继续登出流程
                        print(f"Error deleting login cache file {login_cache_file}: {e}")
                        st.warning(f"无法删除登录缓存文件: {e}。您可能需要手动清除浏览器缓存或稍后再试。")
                # --- 结束删除缓存文件部分 ---

                # 4. 清除 URL 参数 (可选，但推荐)
                st.query_params.clear()

                # 5. 显示登出消息并 Rerun
                st.success("您已成功退出登录。正在返回登录页面...")
                time.sleep(0.5)
                st.rerun()  # 重新运行应用，此时 logged_in 应为 False

        st.markdown(
                """
                <div class='sidebar-footer'>
                    v0.0.5 | © 2025 NEU
                </div>
                """,
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True) # 关闭 sidebar-logout-area


def render_home_page():
    """渲染主页 (移除快速开始部分)"""
    st.markdown(
        """
        <div class="home-container" style="text-align: center; padding-top: 5rem; padding-bottom: 5rem;">
            <h1 class="home-title">欢迎使用 NEU Synthesis Core AI</h1>
            <p class="home-subtitle">您的智能数据分析与预测伙伴</p>
            <p class="home-instruction">请从左侧导航栏选择一个功能模块开始。</p>
            <div class="home-icon-area">
                 🧠 &nbsp;&nbsp; 📈 &nbsp;&nbsp; 📊 &nbsp;&nbsp; 🔬 &nbsp;&nbsp; 🔍 &nbsp;&nbsp; 🧩 &nbsp;&nbsp; ⚖️ &nbsp;&nbsp; 📝 &nbsp;&nbsp; 🔧 &nbsp;&nbsp; ✂️
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )



# --- 页面渲染函数 ---
# (Keep existing render functions for other pages unchanged)
def render_clustering_page():
    """渲染聚类分析页面 (现在包含教学模块的调用)"""
    st.title("🔬 聚类分析") # 在这里设置页面标题

    if clustering_tutorial and TUTORIAL_MODULE_LOADED:
        with st.expander("🎓 打开/关闭聚类教学演示模块", expanded=False):
             # 调用教学模块的主函数来显示其内容
             clustering_tutorial.show_tutorial_page()
    else:
        st.info("聚类教学模块当前不可用。")
    st.markdown("---")

    # 调用主聚类模块的 UI 函数
    if clustering:
        # 注意：这里调用的是 clustering 模块的主 UI 函数
        # 确保 clustering.py 中有类似 create_clustering_ui 的函数被 show_clustering_page 调用
        clustering.show_clustering_page()
    else:
        st.error("聚类分析主模块未能成功加载。")

def render_classification_training_page():
    """渲染分类训练页面 (现在包含教学模块的调用)"""
    st.title("💡 分类训练") # 可以修改标题

    if CLASSIFICATION_TUTORIAL_MODULE_LOADED and classification_tutorial: # 检查模块是否成功加载
        with st.expander("🎓 打开/关闭分类教学演示模块", expanded=False):
             # 调用教学模块的主函数来显示其内容
             classification_tutorial.show_tutorial_page()
    else:
        st.info("分类教学模块当前不可用。")
    st.markdown("---")

    if classification_training:
        classification_training.show_classification_page() # 确保调用的是 classification_training 中的主函数
    else:
        st.error("分类训练主模块未能成功加载。")

def render_classification_validation_page():
    """渲染分类评估页面"""
    if classification_validation:
        classification_validation.show_classification_validation_page()
    else:
        st.error("分类验证模块未能成功加载。")

def render_classification_migration_page():
    """渲染分类模型迁移页面"""
    st.title("📊 分类模型迁移")
    st.write("在这里可以进行分类模型的迁移学习。")
    st.info("分类模型迁移模块开发中，敬请期待...")

def render_regression_training_page():
    """渲染回归训练页面 (现在包含教学模块的调用)"""
    st.title("📈 回归训练") # 可以修改标题

    if REGRESSION_TUTORIAL_MODULE_LOADED and regression_tutorial: # 检查模块是否成功加载
        with st.expander("🎓 打开/关闭回归教学演示模块", expanded=False):
             # 调用教学模块的主函数来显示其内容
             regression_tutorial.show_regression_tutorial_page()
    else:
        st.info("回归教学模块当前不可用。")
    st.markdown("---")

    # --- 原有的回归训练模块调用逻辑 ---
    if regression_training:
        try:
            regression_training.show_regression_training_page()
        except Exception as e:
            st.error(f"回归训练模块执行出错：{str(e)}")
            st.code(traceback.format_exc())
    else:
        st.error("回归训练模块未能成功加载。")
        st.info("请检查 regression_training.py 文件是否存在，以及是否有语法错误或缺失的依赖库。")

        # 显示诊断信息
        with st.expander("查看详细诊断信息"):
            st.code("""
               可能的原因：
               1. 缺少必要的依赖库（catboost, xgboost, lightgbm等）
               2. Python语法错误
               3. 模块中的全局代码执行失败

               建议操作：
               1. 运行: pip install catboost xgboost lightgbm
               2. 检查 regression_training.py 的语法
               3. 查看控制台错误信息
               """)

def render_regression_validation_page():
    """渲染回归验证页面"""
    if regression_validation:
        regression_validation.show_regression_validation_page()
    else:
        st.error("回归验证模块未能成功加载。")

def render_regression_migration_page():
    """渲染回归模型迁移页面"""
    st.title("📈 回归模型迁移")
    st.write("在这里可以进行回归模型的迁移学习。")
    st.info("回归模型迁移模块开发中，敬请期待...")

def render_outlier_detection_page():
    """渲染异常点检测页面"""
    if outlier_detection:
        outlier_detection.show_outlier_detection_page()
    else:
        st.error("异常点检测模块未能成功加载。")

def render_data_reduction_page():
    """渲染数据降维页面"""
    if DATA_REDUCTION_LOADED and data_reduction and hasattr(data_reduction, 'show_data_reduction_page'):
        try:
            data_reduction.show_data_reduction_page()
        except Exception as e:
            st.error(f"渲染数据降维页面时发生错误: {e}")
            st.code(traceback.format_exc())
    elif data_reduction is None:
        st.error("数据降维模块未能成功加载（可能在导入时出错），无法使用此功能。请检查应用启动时的错误信息或控制台输出。")
    elif not hasattr(data_reduction, 'show_data_reduction_page'):
        st.error("错误：数据降维模块已加载，但在模块中未找到 'show_data_reduction_page' 函数。请检查 data_reduction.py 文件。")
    else:
        st.error("数据降维模块未能成功加载，无法使用此功能。请检查文件 data_reduction.py 是否存在且无导入错误。")

def render_missing_value_page():
    """渲染缺失值处理页面"""
    if missing_value_handler:
        missing_value_handler.show_missing_value_page()
    else:
        st.error("缺失值处理模块未能成功加载。")

def render_balancing_page():
    """渲染数据平衡处理页面"""
    if data_balancer:
        data_balancer.show_balancing_page()
    else:
        st.error("数据平衡处理模块未能成功加载。")

def render_data_evaluation_page():
    """渲染数据评估页面 - 调用模块的主函数"""
    # Check if the module was loaded and the function exists
    if DATA_EVALUATOR_LOADED and data_evaluator and hasattr(data_evaluator, 'show_data_evaluator_page'):
         try:
              # Now safely call the function
              data_evaluator.show_data_evaluator_page()
         except Exception as e:
              st.error(f"渲染数据评估页面时发生错误: {e}")
              st.code(traceback.format_exc()) # Show error details
    elif data_evaluator is None:
         st.error("数据评估模块未能成功加载（可能在导入时出错），无法使用此功能。请检查应用启动时的错误信息或控制台输出。")
    elif not hasattr(data_evaluator, 'show_data_evaluator_page'):
         st.error("错误：数据评估模块已加载，但在模块中未找到 'show_data_evaluator_page' 函数。请检查 data_evaluator.py 文件。")
    else: # Fallback if DATA_EVALUATOR_LOADED is False but data_evaluator isn't None (shouldn't happen with current logic, but safe)
        st.error("数据评估模块未能成功加载，无法使用此功能。请检查文件 data_evaluator.py 是否存在且无导入错误。")

# --- 新增：渲染数据处理页面 ---
def render_data_processing_page():
    """渲染数据处理页面 - 调用模块的主函数"""
    if DATA_PROCESSING_LOADED and data_processing and hasattr(data_processing, 'show_data_processing_page'):
        try:
            data_processing.show_data_processing_page()
        except Exception as e:
            st.error(f"渲染数据处理页面时发生错误: {e}")
            st.code(traceback.format_exc())
    elif data_processing is None:
        st.error("数据处理模块未能成功加载（可能在导入时出错），无法使用此功能。请检查应用启动时的错误信息或控制台输出。")
    elif not hasattr(data_processing, 'show_data_processing_page'):
        st.error("错误：数据处理模块已加载，但在模块中未找到 'show_data_processing_page' 函数。请检查 data_processing.py 文件。")
    else:
        st.error("数据处理模块未能成功加载，无法使用此功能。请检查文件 data_processing.py 是否存在且无导入错误。")

# --- 新增：渲染特征提取页面 ---
def render_feature_extraction_page():
    """渲染特征提取页面 - 调用模块的主函数"""
    if FEATURE_EXTRACTION_LOADED and feature_extraction and hasattr(feature_extraction, 'show_feature_extraction_page'):
        try:
            feature_extraction.show_feature_extraction_page()
        except Exception as e:
            st.error(f"渲染特征提取页面时发生错误: {e}")
            st.code(traceback.format_exc())
    elif feature_extraction is None:
        st.error("特征提取模块未能成功加载（可能在导入时出错），无法使用此功能。请检查应用启动时的错误信息或控制台输出。")
    elif not hasattr(feature_extraction, 'show_feature_extraction_page'):
        st.error("错误：特征提取模块已加载，但在模块中未找到 'show_feature_extraction_page' 函数。请检查 feature_extraction.py 文件。")
    else:
        st.error("特征提取模块未能成功加载，无法使用此功能。请检查文件 feature_extraction.py 是否存在且无导入错误。")
# --- 结束新增 ---


def main_app():
    """主应用界面"""
    st.markdown(load_css(), unsafe_allow_html=True)
    if "current_page" not in st.session_state:
        st.session_state.current_page = "home"

    setup_main_sidebar() # 渲染侧边栏

    # --- 主内容区域 ---
    st.markdown('<div class="main-app-container">', unsafe_allow_html=True)

    page = st.session_state.current_page
    print(f"Current page: {page}") # 调试信息

    # --- 修改：添加数据处理页面到路由 ---
    page_render_functions = {
        "home": render_home_page,
        "classification_training": render_classification_training_page,
        "classification_validation": render_classification_validation_page,
        "classification_migration": render_classification_migration_page,
        "regression_training": render_regression_training_page,
        "regression_validation": render_regression_validation_page,
        "regression_migration": render_regression_migration_page,
        "clustering": render_clustering_page,
        "outlier_detection": render_outlier_detection_page,
        "missing_value": render_missing_value_page,
        "data_balancing": render_balancing_page,
        "data_evaluation": render_data_evaluation_page,
        "data_processing": render_data_processing_page,
        "feature_extraction": render_feature_extraction_page,
        "data_reduction": render_data_reduction_page,
        "user_feedback": display_feedback_section,
    }
    # --- 结束修改 ---

    # 获取对应的渲染函数，如果找不到则默认渲染主页
    render_func = page_render_functions.get(page, render_home_page)
    try:
        render_func() # 调用渲染函数
    except Exception as e:
         st.error(f"渲染页面 '{page}' 时发生错误。")
         st.code(traceback.format_exc()) # 显示详细错误供调试

    # Display update log at the bottom of the home page content
    if page == "home" and UPDATE_LOG_LOADED:
        st.markdown("---") # Add a separator
        display_update_log()

    st.markdown('</div>', unsafe_allow_html=True) # Close main-app-container


def main():
    """主程序入口"""

    # 检查URL参数，判断是否为刷新操作
    is_refresh = 'refresh' in st.query_params and st.query_params['refresh'] == 'true'

    # 初始化 session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        # 检查cookie或本地存储是否有登录信息
        if os.path.exists("temp/.login_cache"):
            try:
                with open("temp/.login_cache", "r") as f:
                    login_info = json.load(f)
                    if 'user_id' in login_info and 'timestamp' in login_info:
                        # 验证时间戳是否在有效期内（例如24小时）
                        timestamp = datetime.fromisoformat(login_info['timestamp'])
                        if (datetime.now() - timestamp).total_seconds() < 86400:  # 24小时
                            st.session_state.logged_in = True
                            st.session_state.user_id = login_info['user_id']
                            st.session_state.username = login_info.get('username', '')
                            # 恢复之前的会话状态
                            load_session_state(login_info['user_id'])
                            # 设置恢复标志
                            st.session_state.is_restored = True
            except Exception as e:
                print(f"恢复登录状态出错: {e}")

    if 'register_mode' not in st.session_state:
        st.session_state.register_mode = False

    # 如果是登录状态并且是刷新操作，设置页面保持状态
    if st.session_state.logged_in and is_refresh:
        # 设置URL参数，确保刷新后仍保持当前页面
        current_page = st.session_state.get('current_page', 'home')
        st.query_params['refresh'] = 'true'
        st.query_params['page'] = current_page

    # 检查登录状态
    if not st.session_state.logged_in:
        login_page()
    else:
        # 每次页面加载时保存当前状态
        if hasattr(st.session_state, 'user_id') and st.session_state.user_id:
            # 保存登录信息到临时文件
            os.makedirs("temp", exist_ok=True)
            with open("temp/.login_cache", "w") as f:
                json.dump({
                    'user_id': st.session_state.user_id,
                    'username': st.session_state.get('username', ''),
                    'timestamp': datetime.now().isoformat()
                }, f)

            # 保存会话状态
            save_session_state(st.session_state.user_id)

        main_app()

if __name__ == "__main__":
    required_modules = [
        "clustering.py",
        "clustering_tutorial.py",
        "classification_training.py",
        "classification_tutorial.py",
        "classification_validation.py",
        "regression_training.py",
        "regression_tutorial.py",
        "regression_validation.py",
        "outlier_detection.py",
        "missing_value_handler.py",
        "data_balancer.py",
        "data_evaluator.py",
        "data_processing.py",
        "data_reduction.py",
        "font_utils.py"
    ]

    # 检查文件是否存在，如果不存在，尝试创建空文件或提示用户
    all_modules_found = True
    for mod_file in required_modules:
        if not os.path.exists(mod_file) and mod_file not in ["font_utils.py", "data_processing.py"]: # These will be created
            # 尝试创建空文件，避免启动错误，但功能会缺失
            try:
                with open(mod_file, 'w', encoding='utf-8') as f: # 指定编码
                    f.write(f"# Placeholder for module: {mod_file}\n")
                    f.write("import streamlit as st\n\n")
                    # 添加一个简单的函数，避免导入时直接出错
                    # --- 修改：确保占位函数名唯一 ---
                    func_name = f"show_{mod_file.replace('.py', '').replace('.', '_').replace('-', '_')}_page" # Replace hyphens too
                    # --- 结束修改 ---
                    f.write(f"def {func_name}():\n")
                    f.write(f"    st.warning('模块 {mod_file} 功能尚未实现或文件不存在。')\n") # 修改提示
                print(f"警告: 模块文件 {mod_file} 不存在，已创建占位文件。请确保实现其功能。")
            except Exception as e:
                # 使用 st.error 在应用启动时显示错误
                st.error(f"错误：无法创建缺失的模块文件 {mod_file}: {e}")
                all_modules_found = False # 标记为失败
                # st.stop() # 暂时不停止，允许应用启动，但功能会缺失

    # --- 新增：确保 data_processing.py 存在 ---
    if not os.path.exists("data_processing.py"):
        try:
            # 创建 data_processing.py 的基本结构
            with open("data_processing.py", "w", encoding="utf-8") as f:
                f.write("# -*- coding: utf-8 -*-\n")
                f.write("import streamlit as st\n")
                f.write("import pandas as pd\n")
                f.write("import matplotlib.pyplot as plt\n\n")
                f.write("def show_data_processing_page():\n")
                f.write("    st.title('🔧 数据处理')\n")
                f.write("    st.info('数据处理模块正在建设中...')\n")
                f.write("    # Add placeholder UI elements here if needed\n")
            print("已创建占位 data_processing.py 文件。")
        except Exception as e:
            st.error(f"错误：无法创建 data_processing.py 文件: {e}")
            all_modules_found = False
    # --- 结束新增 ---

    if all_modules_found:
         # 确保 font_utils.py 已创建 (虽然理论上上面的代码会处理)
         if not os.path.exists("font_utils.py"):
              st.error("错误：font_utils.py 未能创建。")
         else:
              main()
    else:
         st.error("部分必需的模块文件缺失或无法创建，应用可能无法完整运行。请检查控制台输出获取详细信息。")

