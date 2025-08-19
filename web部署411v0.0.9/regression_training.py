# -*- coding: utf-8 -*-
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
import joblib
import platform
import matplotlib.font_manager as fm
import random
import warnings

# --- 模型库导入 ---
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# 回归模型
from catboost import CatBoostRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              ExtraTreesRegressor, AdaBoostRegressor)
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

# 深度学习
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam, SGD, RMSprop
    from tensorflow.keras.callbacks import EarlyStopping

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.warning("TensorFlow未安装，深度学习模型将不可用")

# --- 全局设置 ---
warnings.filterwarnings("ignore", category=UserWarning, message="Glyph.*? missing from current font")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")
warnings.filterwarnings("ignore", category=UserWarning, message="`use_label_encoder` is deprecated")

# --- 回归模型信息配置 ---
REGRESSOR_INFO = {
    "decision_tree": {
        "name": "决策树回归 (Decision Tree)",
        "description": "通过递归地将数据分割成子集来构建预测模型的树形算法",
        "advantages": [
            "易于理解和解释，可以可视化",
            "不需要数据预处理（如标准化）",
            "可以处理数值型和类别型特征",
            "能够捕捉非线性关系"
        ],
        "disadvantages": [
            "容易过拟合，特别是深层树",
            "对噪声数据敏感",
            "决策边界是轴平行的",
            "对连续值预测可能不够平滑"
        ],
        "suitable_for": "中小型数据集，需要模型可解释性的回归任务"
    },
    "random_forest": {
        "name": "随机森林回归 (Random Forest)",
        "description": "集成多个决策树，通过平均预测值来提高准确性和稳定性",
        "advantages": [
            "准确率高，不容易过拟合",
            "可以评估特征重要性",
            "能够处理高维数据",
            "对缺失数据和异常值不敏感"
        ],
        "disadvantages": [
            "模型较大，预测速度相对较慢",
            "在某些线性关系强的数据上可能表现不如线性模型",
            "难以解释单个预测"
        ],
        "suitable_for": "大多数回归任务，特别适合需要特征重要性分析的场景"
    },
    "gradient_boosting": {
        "name": "梯度提升回归 (GBDT)",
        "description": "通过逐步改进的方式构建集成模型，每个新模型都尝试纠正前一个模型的错误",
        "advantages": [
            "预测准确率高",
            "能够处理非线性关系",
            "对异常值的鲁棒性较好",
            "可以处理不同类型的特征"
        ],
        "disadvantages": [
            "训练时间长",
            "需要仔细调参以避免过拟合",
            "难以并行化训练过程"
        ],
        "suitable_for": "竞赛和需要高精度的回归场景"
    },
    "catboost": {
        "name": "CatBoost回归",
        "description": "专门处理类别特征的梯度提升算法，自动处理类别特征编码",
        "advantages": [
            "自动处理类别特征，无需预处理",
            "减少过拟合，内置正则化",
            "训练速度快，支持GPU加速",
            "默认参数表现良好"
        ],
        "disadvantages": [
            "模型文件较大",
            "对于纯数值特征可能不如其他专门算法",
            "相对较新，社区支持较少"
        ],
        "suitable_for": "包含大量类别特征的回归数据集"
    },
    "xgboost": {
        "name": "极限梯度提升回归 (XGBoost)",
        "description": "高效的梯度提升实现，在许多机器学习竞赛中表现优异",
        "advantages": [
            "速度快，性能高",
            "内置正则化减少过拟合",
            "可以处理缺失值",
            "支持并行计算和GPU加速"
        ],
        "disadvantages": [
            "参数较多，调参复杂",
            "对类别特征需要预处理",
            "内存消耗较大"
        ],
        "suitable_for": "大规模数据集，竞赛常用，对性能要求高的场景"
    },
    "lightgbm": {
        "name": "轻量级梯度提升回归 (LightGBM)",
        "description": "基于直方图的高效梯度提升算法，微软开发",
        "advantages": [
            "训练速度极快",
            "内存使用少",
            "准确率高",
            "支持分布式训练"
        ],
        "disadvantages": [
            "可能在小数据集上过拟合",
            "对噪声敏感",
            "需要较新版本的依赖库"
        ],
        "suitable_for": "大规模数据集，需要快速训练的回归场景"
    },
    "extra_trees": {
        "name": "极端随机树回归 (Extra Trees)",
        "description": "比随机森林更随机的集成方法，在选择分割点时增加更多随机性",
        "advantages": [
            "训练速度比随机森林快",
            "减少过拟合风险",
            "在某些数据集上表现优于随机森林",
            "方差更小"
        ],
        "disadvantages": [
            "可能需要更多的树来达到相同精度",
            "模型解释性较差",
            "对参数设置敏感"
        ],
        "suitable_for": "高维数据，需要快速训练的回归场景"
    },
    "adaboost": {
        "name": "自适应增强回归 (AdaBoost)",
        "description": "通过组合多个弱学习器构建强学习器的集成方法",
        "advantages": [
            "能够提高弱学习器的性能",
            "不容易过拟合",
            "可以使用各种类型的回归器作为基学习器",
            "理论基础扎实"
        ],
        "disadvantages": [
            "对噪声和异常值敏感",
            "训练时间可能较长",
            "在复杂数据上表现可能不如其他集成方法"
        ],
        "suitable_for": "数据质量较高的回归问题，基学习器相对简单的场景"
    },
    "linear": {
        "name": "线性回归 (Linear Regression)",
        "description": "最基本的线性回归模型，假设特征和目标之间存在线性关系",
        "advantages": [
            "简单快速，易于理解",
            "可解释性强",
            "不需要调整超参数",
            "适合线性关系明显的数据"
        ],
        "disadvantages": [
            "只能处理线性关系",
            "对异常值敏感",
            "当特征数量接近样本数量时可能不稳定"
        ],
        "suitable_for": "线性关系明显的回归问题，需要快速建模和强可解释性的场景"
    },
    "ridge": {
        "name": "岭回归 (Ridge Regression)",
        "description": "带L2正则化的线性回归，可以处理多重共线性问题",
        "advantages": [
            "解决多重共线性问题",
            "防止过拟合",
            "系数收缩使模型更稳定",
            "有闭式解"
        ],
        "disadvantages": [
            "仍然假设线性关系",
            "不能进行特征选择",
            "需要选择正则化参数"
        ],
        "suitable_for": "特征间存在多重共线性的线性回归问题"
    },
    "lasso": {
        "name": "套索回归 (Lasso Regression)",
        "description": "带L1正则化的线性回归，可以进行自动特征选择",
        "advantages": [
            "自动进行特征选择",
            "产生稀疏模型",
            "防止过拟合",
            "可解释性强"
        ],
        "disadvantages": [
            "仍然假设线性关系",
            "当特征数大于样本数时选择有限",
            "对相关特征可能随机选择一个"
        ],
        "suitable_for": "高维稀疏数据，需要特征选择的线性回归问题"
    },
    "elastic_net": {
        "name": "弹性网络回归 (ElasticNet)",
        "description": "结合L1和L2正则化的线性回归，兼具Ridge和Lasso的优点",
        "advantages": [
            "兼具Ridge和Lasso的优点",
            "可以选择相关特征组",
            "在特征数大于样本数时表现良好",
            "参数调节灵活"
        ],
        "disadvantages": [
            "需要调节两个正则化参数",
            "仍然假设线性关系",
            "计算复杂度相对较高"
        ],
        "suitable_for": "高维数据，特征间存在相关性且需要特征选择的场景"
    },
    "svm": {
        "name": "支持向量回归 (SVR)",
        "description": "通过寻找最优超平面来进行回归预测，可以处理非线性关系",
        "advantages": [
            "在高维空间表现良好",
            "使用核技巧可处理非线性问题",
            "对异常值相对鲁棒",
            "泛化能力强"
        ],
        "disadvantages": [
            "大规模数据训练慢",
            "对参数和核函数选择敏感",
            "难以处理大规模数据",
            "模型解释性差"
        ],
        "suitable_for": "中小型数据集，特别是高维数据或非线性关系明显的场景"
    },
    "knn": {
        "name": "K近邻回归 (KNN Regression)",
        "description": "基于实例的懒惰学习算法，预测值是k个最近邻的平均值",
        "advantages": [
            "简单直观，易于理解",
            "对非线性数据效果好",
            "不需要训练过程",
            "可以处理多输出回归"
        ],
        "disadvantages": [
            "计算成本高（预测慢）",
            "对高维数据效果差（维度灾难）",
            "对局部噪声敏感",
            "需要大量存储空间"
        ],
        "suitable_for": "小型数据集，特征维度不高，局部模式明显的回归场景"
    },
    "neural_network": {
        "name": "神经网络回归 (MLP)",
        "description": "多层感知器，可以学习复杂的非线性关系",
        "advantages": [
            "可以学习复杂的非线性关系",
            "适应性强",
            "可以处理大规模特征",
            "理论上可以逼近任意函数"
        ],
        "disadvantages": [
            "需要大量数据",
            "训练时间长",
            "难以解释",
            "需要调整多个超参数",
            "容易过拟合"
        ],
        "suitable_for": "大型复杂数据集，非线性关系强的回归场景"
    },
    "bp_neural_network": {
        "name": "BP神经网络回归",
        "description": "使用反向传播算法的深度神经网络，适合复杂回归任务",
        "advantages": [
            "可以学习复杂的非线性关系",
            "适用于各种回归问题",
            "可以自动学习特征表示",
            "支持多层结构和不同激活函数"
        ],
        "disadvantages": [
            "容易过拟合",
            "需要大量数据",
            "训练时间较长",
            "需要调整很多超参数",
            "需要GPU加速"
        ],
        "suitable_for": "中大型数据集，复杂的非线性回归问题"
    },
    "rnn": {
        "name": "循环神经网络回归 (RNN)",
        "description": "专门处理序列数据的神经网络，适合时序回归任务",
        "advantages": [
            "可以处理变长序列",
            "具有记忆功能",
            "适合时序数据",
            "参数共享效率高"
        ],
        "disadvantages": [
            "梯度消失问题",
            "训练速度慢",
            "难以处理长序列",
            "对数据格式要求高"
        ],
        "suitable_for": "时序数据回归、序列预测问题"
    },
    "lstm": {
        "name": "长短期记忆网络回归 (LSTM)",
        "description": "解决RNN梯度消失问题的改进版循环神经网络",
        "advantages": [
            "解决长期依赖问题",
            "避免梯度消失",
            "适合长序列",
            "记忆能力强"
        ],
        "disadvantages": [
            "计算复杂度高",
            "参数较多",
            "训练时间长",
            "对数据预处理要求高"
        ],
        "suitable_for": "长时序数据回归、需要长期记忆的序列预测问题"
    },
    "gru": {
        "name": "门控循环单元回归 (GRU)",
        "description": "LSTM的简化版本，计算效率更高的循环神经网络",
        "advantages": [
            "比LSTM参数少",
            "训练速度较快",
            "性能接近LSTM",
            "避免梯度消失"
        ],
        "disadvantages": [
            "仍需要序列数据",
            "超参数调节复杂",
            "对短序列效果一般",
            "可解释性差"
        ],
        "suitable_for": "中长序列数据回归、计算资源有限的序列预测"
    },
    "cnn": {
        "name": "卷积神经网络回归 (CNN)",
        "description": "使用卷积操作提取特征的神经网络，适合有空间结构的回归数据",
        "advantages": [
            "平移不变性",
            "局部特征提取能力强",
            "参数共享减少过拟合",
            "计算效率高"
        ],
        "disadvantages": [
            "需要较多数据",
            "对数据维度有要求",
            "超参数敏感",
            "可解释性差"
        ],
        "suitable_for": "图像数据回归、空间结构数据、特征具有局部相关性的回归问题"
    }
}


# [保留原有的字体设置和深度学习模型构建函数...]
def setup_chinese_font():
    """设置中文字体支持"""
    system = platform.system()
    font_candidates = []
    if system == 'Windows':
        font_candidates = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']
    elif system == 'Darwin':  # macOS
        font_candidates = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Apple LiGothic Medium']
    else:  # Linux and other systems
        font_candidates = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'Droid Sans Fallback']

    font_candidates.extend(['DejaVu Sans', 'Arial', 'Helvetica'])

    font_prop = None
    for font_name in font_candidates:
        try:
            font_path = fm.findfont(fm.FontProperties(family=font_name))
            if os.path.exists(font_path) and 'DejaVuSans' not in font_path:
                print(f"字体日志: 使用字体 '{font_name}' 在路径: {font_path}")
                plt.rcParams['font.family'] = 'sans-serif'
                plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
                font_prop = fm.FontProperties(family=font_name)
                break
        except Exception as e:
            print(f"字体日志: 尝试字体 {font_name} 失败: {e}")

    if not font_prop:
        print("字体日志: 未找到合适的中文字体，绘图中的中文可能无法正常显示。")

    plt.rcParams['axes.unicode_minus'] = False
    return font_prop


FONT_PROP = setup_chinese_font()


# [这里保留原有的深度学习模型构建函数，因为它们已经很完善了]
def build_bp_neural_network_regressor(input_shape, params):
    """构建BP神经网络回归模型"""
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow不可用，无法构建深度学习模型")

    model = Sequential()

    # 解析隐藏层结构
    try:
        hidden_units = [int(x.strip()) for x in params['hidden_layers'].split(',') if x.strip()]
    except:
        hidden_units = [128, 64, 32]

    # 输入层
    model.add(layers.Dense(hidden_units[0], activation=params['activation'], input_shape=(input_shape,)))
    if params['dropout_rate'] > 0:
        model.add(layers.Dropout(params['dropout_rate']))

    # 隐藏层
    for units in hidden_units[1:]:
        model.add(layers.Dense(units, activation=params['activation']))
        if params['dropout_rate'] > 0:
            model.add(layers.Dropout(params['dropout_rate']))

    # 输出层
    model.add(layers.Dense(1))

    # 编译模型
    optimizer_map = {
        'adam': Adam(learning_rate=params['learning_rate']),
        'sgd': SGD(learning_rate=params['learning_rate']),
        'rmsprop': RMSprop(learning_rate=params['learning_rate'])
    }

    model.compile(optimizer=optimizer_map[params['optimizer']], loss='mse', metrics=['mae'])
    return model


# [保留其他深度学习模型构建函数...]

# --- 新增：回归模型训练管理器 ---
class MultiRegressionTrainer:
    """多模型回归训练管理器"""

    def __init__(self):
        self.trainers = {}
        self.results = {}
        self.scaler = None

    def get_default_params(self, model_type):
        """获取模型的默认参数"""
        defaults = {
            'decision_tree': {'max_depth': 5, 'min_samples_split': 2, 'min_samples_leaf': 1,
                              'criterion': 'squared_error'},
            'random_forest': {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2},
            'gradient_boosting': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'subsample': 1.0},
            'catboost': {'iterations': 100, 'learning_rate': 0.1, 'depth': 6},
            'xgboost': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'subsample': 1.0,
                        'colsample_bytree': 1.0},
            'lightgbm': {'n_estimators': 100, 'learning_rate': 0.1, 'num_leaves': 31, 'max_depth': -1},
            'extra_trees': {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2},
            'adaboost': {'n_estimators': 50, 'learning_rate': 1.0, 'loss': 'linear'},
            'linear': {},
            'ridge': {'alpha': 1.0},
            'lasso': {'alpha': 1.0},
            'elastic_net': {'alpha': 1.0, 'l1_ratio': 0.5},
            'svm': {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale', 'epsilon': 0.1},
            'knn': {'n_neighbors': 5, 'weights': 'uniform', 'algorithm': 'auto', 'p': 2},
            'neural_network': {'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam', 'alpha': 0.0001,
                               'learning_rate_init': 0.001, 'max_iter': 200},
            'bp_neural_network': {
                'hidden_layers': '128,64,32', 'activation': 'relu', 'dropout_rate': 0.2,
                'learning_rate': 0.001, 'batch_size': 32, 'epochs': 100, 'optimizer': 'adam',
                'early_stopping': True, 'patience': 10
            },
            'rnn': {
                'rnn_units': 64, 'num_layers': 2, 'dropout_rate': 0.2, 'learning_rate': 0.001,
                'batch_size': 32, 'epochs': 100, 'optimizer': 'adam', 'sequence_length': 10,
                'early_stopping': True, 'patience': 10
            },
            'lstm': {
                'lstm_units': 64, 'num_layers': 2, 'dropout_rate': 0.2, 'learning_rate': 0.001,
                'batch_size': 32, 'epochs': 100, 'optimizer': 'adam', 'sequence_length': 10,
                'early_stopping': True, 'patience': 10
            },
            'gru': {
                'gru_units': 64, 'num_layers': 2, 'dropout_rate': 0.2, 'learning_rate': 0.001,
                'batch_size': 32, 'epochs': 100, 'optimizer': 'adam', 'sequence_length': 10,
                'early_stopping': True, 'patience': 10
            },
            'cnn': {
                'conv_layers': '32,64,128', 'kernel_size': 3, 'pool_size': 2, 'dense_units': 128,
                'dropout_rate': 0.2, 'learning_rate': 0.001, 'batch_size': 32, 'epochs': 100,
                'optimizer': 'adam', 'early_stopping': True, 'patience': 10
            }
        }
        return defaults.get(model_type, {})

    def train_models(self, X, y, selected_models, model_params, groups=None,
                     normalize_features=True, test_size=0.2, progress_callback=None):
        """训练多个模型"""
        # 数据预处理 - 更严格的数据验证
        print(f"原始数据形状: X={X.shape}, y={y.shape}")

        # 转换为数值类型，但保留原始数据结构
        X_clean = X.copy()
        y_clean = y.copy()

        # 更安全的数值转换
        for col in X_clean.columns:
            if X_clean[col].dtype == 'object':
                # 尝试转换为数值，失败的保留为NaN
                X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce')

        if y_clean.dtype == 'object':
            y_clean = pd.to_numeric(y_clean, errors='coerce')

        # 检查转换后的数据
        print(f"转换后数据形状: X_clean={X_clean.shape}, y_clean={y_clean.shape}")
        print(f"X_clean数据类型: {X_clean.dtypes}")
        print(f"y_clean数据类型: {y_clean.dtype}")
        print(f"y_clean统计: min={y_clean.min()}, max={y_clean.max()}, mean={y_clean.mean()}")

        # 合并数据以便一起删除NaN行
        combined = pd.concat([X_clean, y_clean], axis=1)
        if groups is not None:
            combined = pd.concat([combined, groups], axis=1)

        # 删除包含NaN的行
        initial_rows = len(combined)
        combined.dropna(inplace=True)
        final_rows = len(combined)

        print(f"数据清洗: {initial_rows} -> {final_rows} 行 (删除了 {initial_rows - final_rows} 行)")

        if combined.empty:
            raise ValueError("清洗后数据为空，请检查数据质量")

        # 分离清洗后的数据
        X_processed = combined[X.columns]
        y_processed = combined[y.name]
        groups_processed = combined[groups.name] if groups is not None else None

        # 再次验证数据
        if len(X_processed) == 0 or len(y_processed) == 0:
            raise ValueError("处理后的数据为空")

        if y_processed.std() == 0:
            raise ValueError("目标变量无变化（标准差为0），无法进行有效的回归分析")

        # 标准化
        if normalize_features:
            self.scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_processed),
                columns=X_processed.columns,
                index=X_processed.index
            )
            print("已应用特征标准化")
        else:
            X_scaled = X_processed
            self.scaler = None
            print("未使用特征标准化")

        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_processed, test_size=test_size, random_state=42
        )

        print(f"数据分割: 训练集{X_train.shape}, 测试集{X_test.shape}")

        # 训练模型
        total_models = len(selected_models)
        for i, model_type in enumerate(selected_models):
            if progress_callback:
                progress_callback(int((i / total_models) * 100))

            try:
                # 获取参数
                params = model_params.get(model_type, self.get_default_params(model_type))

                # 训练模型
                result = self._train_single_model(
                    X_train, X_test, y_train, y_test,
                    model_type, params, X_processed, y_processed, groups_processed
                )

                self.results[model_type] = result
                print(f"模型 {model_type} 训练完成，测试R²: {result.get('test_r2', 0):.4f}")

            except Exception as e:
                st.error(f"训练模型 {REGRESSOR_INFO[model_type]['name']} 时出错: {str(e)}")
                print(f"模型 {model_type} 训练失败: {str(e)}")
                self.results[model_type] = {'error': str(e)}

        if progress_callback:
            progress_callback(100)

        return self.results

    def _train_single_model(self, X_train, X_test, y_train, y_test,
                            model_type, params, X_full, y_full, groups):
        """训练单个模型"""
        # 创建模型
        if model_type == 'decision_tree':
            model = DecisionTreeRegressor(**params, random_state=42)
        elif model_type == 'random_forest':
            model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        elif model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(**params, random_state=42)
        elif model_type == 'catboost':
            model = CatBoostRegressor(**params, verbose=0, random_state=42)
        elif model_type == 'xgboost':
            model = XGBRegressor(**params, random_state=42)
        elif model_type == 'lightgbm':
            model = LGBMRegressor(**params, random_state=42)
        elif model_type == 'extra_trees':
            model = ExtraTreesRegressor(**params, random_state=42, n_jobs=-1)
        elif model_type == 'adaboost':
            model = AdaBoostRegressor(**params, random_state=42)
        elif model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'ridge':
            model = Ridge(**params, random_state=42)
        elif model_type == 'lasso':
            model = Lasso(**params, random_state=42)
        elif model_type == 'elastic_net':
            model = ElasticNet(**params, random_state=42)
        elif model_type == 'svm':
            model = SVR(**params)
        elif model_type == 'knn':
            model = KNeighborsRegressor(**params)
        elif model_type == 'neural_network':
            model = MLPRegressor(**params, random_state=42)
        elif model_type in ['bp_neural_network', 'rnn', 'lstm', 'gru', 'cnn']:
            return self._train_deep_learning_model(
                X_train, X_test, y_train, y_test, model_type, params, X_full, y_full
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        # 训练模型
        model.fit(X_train, y_train)

        # 预测
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        # 修复：正确处理全数据集预测时的标准化
        if self.scaler is not None:
            # 如果使用了标准化，需要对X_full进行标准化后再预测
            X_full_scaled = pd.DataFrame(
                self.scaler.transform(X_full),
                columns=X_full.columns,
                index=X_full.index
            )
            full_pred = model.predict(X_full_scaled)
        else:
            # 如果没有使用标准化，直接预测
            full_pred = model.predict(X_full)

        # 计算指标
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)

        # 特征重要性
        feature_importance = {}
        try:
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(X_train.columns, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                feature_importance = dict(zip(X_train.columns, np.abs(model.coef_)))
        except:
            pass

        return {
            'model': model,
            'model_type': model_type,
            'params': params,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'feature_importance': feature_importance,
            'y_full': y_full,
            'pred_full': full_pred,
            'index_full': X_full.index
        }

    def _train_deep_learning_model(self, X_train, X_test, y_train, y_test,
                                   model_type, params, X_full, y_full):
        """训练深度学习模型"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow不可用")

        # 数据预处理（根据模型类型）
        if model_type == 'bp_neural_network':
            X_train_processed = X_train.values.astype(np.float32)
            X_test_processed = X_test.values.astype(np.float32)
            input_shape = X_train_processed.shape[1]
            model = build_bp_neural_network_regressor(input_shape, params)
        # 这里可以添加其他深度学习模型的处理逻辑

        y_train_processed = y_train.values.astype(np.float32)
        y_test_processed = y_test.values.astype(np.float32)

        # 设置回调
        callbacks = []
        if params.get('early_stopping', True):
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=params.get('patience', 10),
                restore_best_weights=True
            )
            callbacks.append(early_stopping)

        # 训练
        model.fit(
            X_train_processed, y_train_processed,
            validation_data=(X_test_processed, y_test_processed),
            epochs=params.get('epochs', 100),
            batch_size=params.get('batch_size', 32),
            callbacks=callbacks,
            verbose=0
        )

        # 预测
        train_pred = model.predict(X_train_processed, verbose=0).flatten()
        test_pred = model.predict(X_test_processed, verbose=0).flatten()

        X_full_processed = X_full.values.astype(np.float32)
        if self.scaler is not None:
            X_full_processed = self.scaler.transform(X_full).astype(np.float32)
        full_pred = model.predict(X_full_processed, verbose=0).flatten()

        # 计算指标
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)

        return {
            'model': model,
            'model_type': model_type,
            'params': params,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'feature_importance': {},  # 深度学习模型暂不提供特征重要性
            'y_full': y_full,
            'pred_full': full_pred,
            'index_full': X_full.index
        }


# --- 新增：初始化会话状态 ---
def initialize_regression_session_state():
    """初始化回归页面的会话状态变量"""
    defaults = {
        'regression_data': None,
        'column_names': [],
        'selected_input_columns': [],
        'selected_output_column': None,
        'data_source_type': 'file',
        'file_names': None,
        'training_results_dict': {},
        'model_trained_flag': False,
        'normalize_features': True,
        'test_size': 0.2,
        'scaler': None,
        'selected_models': [],
        'model_params': {},
        'has_group_column': False,
        'selected_group_column': None,
        'multi_trainer': MultiRegressionTrainer(),
        'use_cv': False,
        'cv_folds': 5,
    }

    # 添加所有模型的默认参数
    for model_type in REGRESSOR_INFO.keys():
        defaults[f'{model_type}_params'] = MultiRegressionTrainer().get_default_params(model_type)

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# --- 新增：主页面函数 ---
def show_regression_training_page():
    """显示回归训练页面"""
    initialize_regression_session_state()

    st.title("回归模型训练")
    st.markdown("---")

    # 创建选项卡
    tab1, tab2, tab3, tab4 = st.tabs([
        "📁 1. 数据导入",
        "📊 2. 特征选择",
        "⚙️ 3. 模型训练",
        "📈 4. 结果展示"
    ])

    with tab1:
        create_data_import_section()

    with tab2:
        create_column_selection_section()

    with tab3:
        create_model_training_section()

    with tab4:
        create_results_section()


# --- 新增：UI创建函数 ---
def create_data_import_section():
    """创建数据导入部分UI"""
    st.header("数据源选择")
    st.info("您可以上传单个包含特征和目标列的文件，或者上传一个文件夹，其中每个文件代表一个样本组。")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("上传文件")
        uploaded_file = st.file_uploader("选择 CSV 或 Excel 文件", type=["csv", "xlsx", "xls"], key="reg_file_uploader")

        if uploaded_file:
            if st.button("加载文件数据", key="load_file_btn_reg"):
                with st.spinner(f"正在加载 {uploaded_file.name}..."):
                    try:
                        data = pd.read_csv(uploaded_file) if uploaded_file.name.lower().endswith(
                            '.csv') else pd.read_excel(uploaded_file)

                        # 基本清理
                        data.dropna(axis=1, how='all', inplace=True)

                        if data.empty:
                            st.error("上传的文件为空或不包含有效数据。")
                        else:
                            st.session_state.regression_data = data
                            st.session_state.column_names = list(data.columns)
                            st.session_state.data_source_type = "file"
                            st.session_state.file_names = None
                            # 清除旧结果
                            st.session_state.training_results_dict = {}
                            st.session_state.model_trained_flag = False
                            st.session_state.selected_input_columns = []
                            st.session_state.selected_output_column = None

                            st.success(
                                f"已成功加载: {uploaded_file.name} (包含 {len(data)} 行, {len(data.columns)} 列)")
                            st.rerun()

                    except Exception as e:
                        st.error(f"加载数据时出错: {str(e)}")
                        st.session_state.regression_data = None

    with col2:
        st.subheader("上传文件夹")
        folder_path = st.text_input("输入包含数据文件的文件夹路径", key="reg_folder_path")

        if folder_path and os.path.isdir(folder_path):
            if st.button("加载文件夹数据", key="load_folder_btn_reg"):
                folder_progress = st.progress(0)
                status_text = st.empty()

                def update_folder_progress(p):
                    folder_progress.progress(p / 100)
                    status_text.text(f"正在处理文件夹... {p}%")

                with st.spinner("正在处理文件夹中的文件..."):
                    results, error_msg = process_folder_data(folder_path, progress_callback=update_folder_progress)
                    folder_progress.progress(100)
                    status_text.text("文件夹处理完成。")

                    if results:
                        st.session_state.regression_data = results
                        st.session_state.column_names = list(results['X'].columns)
                        st.session_state.data_source_type = "folder"
                        st.session_state.file_names = results['groups']
                        # 清除旧结果
                        st.session_state.training_results_dict = {}
                        st.session_state.model_trained_flag = False
                        st.session_state.selected_input_columns = list(results['X'].columns)
                        st.session_state.selected_output_column = None

                        st.success(
                            f"成功加载文件夹数据: {len(results['X'])} 行, {len(results['groups'].unique())} 个文件组。")
                        st.rerun()
                    else:
                        st.error(f"处理文件夹时出错: {error_msg}")

        elif folder_path:
            st.warning("输入的路径不是有效的文件夹。")

    # 显示数据预览
    if st.session_state.regression_data is not None:
        st.markdown("---")
        st.subheader("数据预览 (前5行)")
        try:
            if st.session_state.data_source_type == "file":
                st.dataframe(st.session_state.regression_data.head())
            else:
                st.dataframe(st.session_state.regression_data['X'].head())
        except Exception as e:
            st.error(f"预览数据时出错: {e}")


def create_column_selection_section():
    """创建列选择部分UI"""
    st.header("特征和目标列选择")

    if st.session_state.regression_data is None:
        st.info("请先在数据导入选项卡中加载数据。")
        return

    all_columns = st.session_state.column_names
    if not all_columns:
        st.warning("未能从加载的数据中获取列名。")
        return

    st.info("请选择用于模型训练的输入特征列和要预测的目标列。")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("输入特征 (X)")
        default_inputs = [col for col in st.session_state.selected_input_columns if col in all_columns]

        selected_inputs = st.multiselect(
            "选择一个或多个输入特征列",
            all_columns,
            default=default_inputs,
            key="input_col_multi_reg"
        )

        if selected_inputs:
            st.write(f"已选择 {len(selected_inputs)} 个输入特征。")
        else:
            st.warning("请至少选择一个输入特征。")

    with col2:
        st.subheader("目标列 (Y)")
        output_options = [col for col in all_columns if col not in selected_inputs]
        current_output_index = 0
        if st.session_state.selected_output_column in output_options:
            current_output_index = output_options.index(st.session_state.selected_output_column) + 1

        selected_output = st.selectbox(
            "选择一个目标（预测）列",
            [None] + output_options,
            index=current_output_index,
            key="output_col_select_reg"
        )

        if selected_output:
            st.write(f"已选择 '{selected_output}' 作为目标列。")
        else:
            st.warning("请选择一个目标列。")

    # 确认按钮
    if st.button("✅ 确认特征选择", key="confirm_columns_reg_btn", use_container_width=True):
        st.session_state.selected_input_columns = selected_inputs
        st.session_state.selected_output_column = selected_output
        st.success("特征和目标列已确认！")
        time.sleep(0.5)

    st.markdown("---")
    st.subheader("数据预处理选项")
    col_prep1, col_prep2 = st.columns(2)

    with col_prep1:
        normalize_features = st.checkbox(
            "标准化输入特征 (StandardScaler, 推荐)",
            value=st.session_state.normalize_features,
            key="normalize_cb_reg"
        )

    with col_prep2:
        test_size = st.slider(
            "测试集比例",
            min_value=0.1,
            max_value=0.5,
            value=st.session_state.test_size,
            step=0.05,
            help="用于模型评估的数据比例。",
            key="test_size_slider_reg"
        )

    # 确认预处理设置
    if st.button("✅ 确认预处理设置", key="confirm_preproc_reg_btn", use_container_width=True):
        st.session_state.normalize_features = normalize_features
        st.session_state.test_size = test_size
        st.success("预处理设置已确认！")
        time.sleep(0.5)


def create_model_training_section():
    """创建模型训练选项部分UI"""
    st.header("模型训练配置")

    # 前置检查
    data_loaded = st.session_state.regression_data is not None
    features_selected = bool(st.session_state.selected_input_columns)
    target_selected = bool(st.session_state.selected_output_column)

    if not data_loaded:
        st.info("请先在数据导入选项卡中加载数据。")
        return
    if not features_selected or not target_selected:
        st.warning("请先在特征选择选项卡中选择输入特征和目标列。")
        return

    # 初始化session state
    if 'selected_models' not in st.session_state:
        st.session_state.selected_models = []
    if 'model_params' not in st.session_state:
        st.session_state.model_params = {}
    if 'training_results_dict' not in st.session_state:
        st.session_state.training_results_dict = {}

    # 模型选择和配置部分
    st.subheader("选择回归算法")

    # 使用columns布局
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 可用算法列表")
        # 显示所有可用的回归器
        for model_key, model_info in REGRESSOR_INFO.items():
            if st.checkbox(model_info['name'], key=f"check_{model_key}"):
                if model_key not in st.session_state.selected_models:
                    st.session_state.selected_models.append(model_key)
                    # 初始化该模型的参数
                    if model_key not in st.session_state.model_params:
                        st.session_state.model_params[model_key] = st.session_state.multi_trainer.get_default_params(
                            model_key)
            else:
                if model_key in st.session_state.selected_models:
                    st.session_state.selected_models.remove(model_key)
                    # 清除该模型的参数和结果
                    if model_key in st.session_state.model_params:
                        del st.session_state.model_params[model_key]
                    if model_key in st.session_state.training_results_dict:
                        del st.session_state.training_results_dict[model_key]

    with col2:
        st.markdown("### 模型配置和说明")

        if st.session_state.selected_models:
            # 使用tabs显示每个选中模型的配置
            model_tabs = st.tabs([REGRESSOR_INFO[m]['name'] for m in st.session_state.selected_models])

            for i, (model_key, tab) in enumerate(zip(st.session_state.selected_models, model_tabs)):
                with tab:
                    # 显示模型说明
                    with st.expander("算法说明", expanded=True):
                        info = REGRESSOR_INFO[model_key]
                        st.markdown(f"**描述：** {info['description']}")

                        col_adv, col_dis = st.columns(2)
                        with col_adv:
                            st.markdown("**优点：**")
                            for adv in info['advantages']:
                                st.markdown(f"• {adv}")

                        with col_dis:
                            st.markdown("**缺点：**")
                            for dis in info['disadvantages']:
                                st.markdown(f"• {dis}")

                        st.markdown(f"**适用场景：** {info['suitable_for']}")

                    # 显示参数配置
                    st.markdown("#### 参数设置")
                    params = create_param_widgets(model_key, f"{model_key}_params")
                    st.session_state.model_params[model_key] = params
        else:
            st.info("请从左侧选择至少一个回归算法")

    # 训练选项
    st.markdown("---")
    st.subheader("训练选项")

    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        use_cv = st.checkbox("使用交叉验证评估性能", value=st.session_state.get('use_cv', False))
        st.session_state.use_cv = use_cv

    with col_opt2:
        if use_cv:
            cv_folds = st.slider("交叉验证折数", 2, 10, st.session_state.get('cv_folds', 5))
            st.session_state.cv_folds = cv_folds

    # 训练按钮
    st.markdown("---")
    if st.button("🚀 开始训练所选模型", type="primary", use_container_width=True):
        if not st.session_state.selected_models:
            st.error("请至少选择一个回归算法")
            return

        # 训练所有选中的模型
        train_selected_models()

    # 显示训练进度和结果摘要
    if st.session_state.training_results_dict:
        st.markdown("---")
        st.subheader("训练结果摘要")
        display_results_summary()


def create_param_widgets(model_type, key_prefix):
    """为不同模型创建参数输入控件"""
    params = {}

    if model_type == "decision_tree":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            params['max_depth'] = st.slider("最大深度 (0=无限制)", 0, 50, 5, 1, key=f"{key_prefix}_max_depth")
            if params['max_depth'] == 0:
                params['max_depth'] = None
        with col2:
            params['min_samples_split'] = st.slider("最小分裂样本数", 2, 20, 2, 1, key=f"{key_prefix}_min_split")
        with col3:
            params['min_samples_leaf'] = st.slider("叶节点最小样本数", 1, 20, 1, 1, key=f"{key_prefix}_min_leaf")
        with col4:
            params['criterion'] = st.selectbox("分裂标准", ["squared_error", "friedman_mse", "absolute_error"],
                                               key=f"{key_prefix}_criterion")

    elif model_type == "random_forest":
        col1, col2, col3 = st.columns(3)
        with col1:
            params['n_estimators'] = st.slider("树的数量", 10, 1000, 100, 10, key=f"{key_prefix}_n_est")
        with col2:
            max_depth = st.slider("最大深度 (0=无限制)", 0, 50, 0, 1, key=f"{key_prefix}_max_depth")
            params['max_depth'] = None if max_depth == 0 else max_depth
        with col3:
            params['min_samples_split'] = st.slider("最小分裂样本数", 2, 20, 2, 1, key=f"{key_prefix}_min_split")

    elif model_type == "gradient_boosting":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            params['n_estimators'] = st.slider("树的数量", 50, 1000, 100, 10, key=f"{key_prefix}_n_est")
        with col2:
            params['learning_rate'] = st.slider("学习率", 0.001, 0.5, 0.1, 0.01, format="%.3f", key=f"{key_prefix}_lr")
        with col3:
            params['max_depth'] = st.slider("最大深度", 1, 20, 3, 1, key=f"{key_prefix}_max_depth")
        with col4:
            params['subsample'] = st.slider("子样本比例", 0.1, 1.0, 1.0, 0.1, format="%.1f",
                                            key=f"{key_prefix}_subsample")

    elif model_type == "catboost":
        col1, col2, col3 = st.columns(3)
        with col1:
            params['iterations'] = st.slider("迭代次数", 50, 2000, 100, 50, key=f"{key_prefix}_iter")
        with col2:
            params['learning_rate'] = st.slider("学习率", 0.001, 0.5, 0.1, 0.01, format="%.3f", key=f"{key_prefix}_lr")
        with col3:
            params['depth'] = st.slider("树深度", 1, 16, 6, 1, key=f"{key_prefix}_depth")

    elif model_type == "xgboost":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            params['n_estimators'] = st.slider("树的数量", 50, 1000, 100, 10, key=f"{key_prefix}_n_est")
        with col2:
            params['learning_rate'] = st.slider("学习率", 0.001, 0.5, 0.1, 0.01, format="%.3f", key=f"{key_prefix}_lr")
        with col3:
            params['max_depth'] = st.slider("最大深度", 1, 20, 6, 1, key=f"{key_prefix}_max_depth")
        with col4:
            params['subsample'] = st.slider("子样本比例", 0.1, 1.0, 1.0, 0.1, format="%.1f",
                                            key=f"{key_prefix}_subsample")

    elif model_type == "lightgbm":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            params['n_estimators'] = st.slider("树的数量", 50, 1000, 100, 10, key=f"{key_prefix}_n_est")
        with col2:
            params['learning_rate'] = st.slider("学习率", 0.001, 0.5, 0.1, 0.01, format="%.3f", key=f"{key_prefix}_lr")
        with col3:
            params['num_leaves'] = st.slider("叶子数量", 10, 300, 31, 5, key=f"{key_prefix}_num_leaves")
        with col4:
            max_depth = st.slider("最大深度 (-1=无限制)", -1, 20, -1, 1, key=f"{key_prefix}_max_depth")
            params['max_depth'] = max_depth

    elif model_type == "extra_trees":
        col1, col2, col3 = st.columns(3)
        with col1:
            params['n_estimators'] = st.slider("树的数量", 10, 1000, 100, 10, key=f"{key_prefix}_n_est")
        with col2:
            max_depth = st.slider("最大深度 (0=无限制)", 0, 50, 0, 1, key=f"{key_prefix}_max_depth")
            params['max_depth'] = None if max_depth == 0 else max_depth
        with col3:
            params['min_samples_split'] = st.slider("最小分裂样本数", 2, 20, 2, 1, key=f"{key_prefix}_min_split")

    elif model_type == "adaboost":
        col1, col2, col3 = st.columns(3)
        with col1:
            params['n_estimators'] = st.slider("弱学习器数量", 10, 500, 50, 10, key=f"{key_prefix}_n_est")
        with col2:
            params['learning_rate'] = st.slider("学习率", 0.1, 2.0, 1.0, 0.1, format="%.1f", key=f"{key_prefix}_lr")
        with col3:
            params['loss'] = st.selectbox("损失函数", ["linear", "square", "exponential"], key=f"{key_prefix}_loss")

    elif model_type == "linear":
        st.info("线性回归没有需要调整的超参数")
        params = {}

    elif model_type == "ridge":
        col1, col2 = st.columns(2)
        with col1:
            params['alpha'] = st.slider("正则化参数 Alpha", 0.1, 100.0, 1.0, 0.1, format="%.1f",
                                        key=f"{key_prefix}_alpha")
        with col2:
            params['solver'] = st.selectbox("求解器", ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
                                            key=f"{key_prefix}_solver")

    elif model_type == "lasso":
        col1, col2 = st.columns(2)
        with col1:
            params['alpha'] = st.slider("正则化参数 Alpha", 0.1, 100.0, 1.0, 0.1, format="%.1f",
                                        key=f"{key_prefix}_alpha")
        with col2:
            params['max_iter'] = st.slider("最大迭代次数", 100, 5000, 1000, 100, key=f"{key_prefix}_max_iter")

    elif model_type == "elastic_net":
        col1, col2, col3 = st.columns(3)
        with col1:
            params['alpha'] = st.slider("正则化参数 Alpha", 0.1, 100.0, 1.0, 0.1, format="%.1f",
                                        key=f"{key_prefix}_alpha")
        with col2:
            params['l1_ratio'] = st.slider("L1比例", 0.0, 1.0, 0.5, 0.1, format="%.1f", key=f"{key_prefix}_l1_ratio")
        with col3:
            params['max_iter'] = st.slider("最大迭代次数", 100, 5000, 1000, 100, key=f"{key_prefix}_max_iter")

    elif model_type == "svm":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            params['C'] = st.slider("正则化参数 C", 0.1, 100.0, 1.0, 0.1, format="%.1f", key=f"{key_prefix}_c")
        with col2:
            params['kernel'] = st.selectbox("核函数", ["rbf", "linear", "poly", "sigmoid"], key=f"{key_prefix}_kernel")
        with col3:
            gamma_options = ["scale", "auto", 0.001, 0.01, 0.1, 1.0]
            params['gamma'] = st.selectbox("Gamma", gamma_options, key=f"{key_prefix}_gamma")
        with col4:
            params['epsilon'] = st.slider("Epsilon", 0.01, 1.0, 0.1, 0.01, format="%.2f", key=f"{key_prefix}_epsilon")

    elif model_type == "knn":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            params['n_neighbors'] = st.slider("邻居数量 K", 1, 50, 5, 1, key=f"{key_prefix}_k")
        with col2:
            params['weights'] = st.selectbox("权重", ["uniform", "distance"], key=f"{key_prefix}_weights")
        with col3:
            params['algorithm'] = st.selectbox("算法", ["auto", "ball_tree", "kd_tree", "brute"],
                                               key=f"{key_prefix}_algorithm")
        with col4:
            params['p'] = st.slider("距离度量参数", 1, 5, 2, 1, key=f"{key_prefix}_p")

    elif model_type == "neural_network":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            hidden_size = st.slider("隐藏层大小", 10, 500, 100, 10, key=f"{key_prefix}_hidden_size")
            params['hidden_layer_sizes'] = (hidden_size,)
        with col2:
            params['activation'] = st.selectbox("激活函数", ["relu", "tanh", "logistic"],
                                                key=f"{key_prefix}_activation")
        with col3:
            params['solver'] = st.selectbox("求解器", ["adam", "lbfgs", "sgd"], key=f"{key_prefix}_solver")
        with col4:
            params['alpha'] = st.slider("正则化参数", 0.0001, 0.1, 0.0001, 0.0001, format="%.4f",
                                        key=f"{key_prefix}_alpha")

    elif model_type == "bp_neural_network":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            params['hidden_layers'] = st.text_input("隐藏层结构 (逗号分隔)", "128,64,32",
                                                    key=f"{key_prefix}_hidden_layers")
        with col2:
            params['activation'] = st.selectbox("激活函数", ["relu", "tanh", "sigmoid"], key=f"{key_prefix}_activation")
        with col3:
            params['dropout_rate'] = st.slider("Dropout率", 0.0, 0.8, 0.2, 0.1, format="%.1f",
                                               key=f"{key_prefix}_dropout")
        with col4:
            params['learning_rate'] = st.slider("学习率", 0.0001, 0.1, 0.001, 0.0001, format="%.4f",
                                                key=f"{key_prefix}_lr")

        col5, col6, col7, col8 = st.columns(4)
        with col5:
            params['batch_size'] = st.slider("批次大小", 8, 256, 32, 8, key=f"{key_prefix}_batch_size")
        with col6:
            params['epochs'] = st.slider("训练轮数", 10, 500, 100, 10, key=f"{key_prefix}_epochs")
        with col7:
            params['optimizer'] = st.selectbox("优化器", ["adam", "sgd", "rmsprop"], key=f"{key_prefix}_optimizer")
        with col8:
            params['early_stopping'] = st.checkbox("早停", True, key=f"{key_prefix}_early_stopping")

    # 对于其他深度学习模型，可以类似处理
    elif model_type in ["rnn", "lstm", "gru", "cnn"]:
        st.info(f"{REGRESSOR_INFO[model_type]['name']} 使用默认参数")
        params = {}

    else:
        st.warning(f"未定义参数控件的模型类型: {model_type}")
        params = {}

    return params


def train_selected_models():
    """训练所有选中的模型"""
    # 准备数据
    if st.session_state.data_source_type == "file":
        data_source = st.session_state.regression_data
        X = data_source[st.session_state.selected_input_columns].copy()
        y = data_source[st.session_state.selected_output_column].copy()
        groups = None
        if st.session_state.has_group_column and st.session_state.selected_group_column:
            groups = data_source[st.session_state.selected_group_column].copy()
    else:
        # 修复：文件夹数据处理时正确获取目标变量
        data_dict = st.session_state.regression_data
        X_data = data_dict['X']

        # 检查目标列是否存在
        if st.session_state.selected_output_column not in X_data.columns:
            st.error(f"目标列 '{st.session_state.selected_output_column}' 不存在于数据中")
            return

        X = X_data[st.session_state.selected_input_columns].copy()
        y = X_data[st.session_state.selected_output_column].copy()  # 修复：从X_data获取y
        groups = data_dict['groups'].copy()

    # 数据验证
    if X.empty or y.empty:
        st.error("输入数据或目标数据为空")
        return

    if len(X) != len(y):
        st.error(f"输入数据长度({len(X)})与目标数据长度({len(y)})不匹配")
        return

    # 创建进度条
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(p):
        progress_bar.progress(p / 100)
        status_text.text(f"正在训练模型... {p}%")

    try:
        # 训练所有选中的模型
        results = st.session_state.multi_trainer.train_models(
            X, y,
            st.session_state.selected_models,
            st.session_state.model_params,
            groups=groups,
            normalize_features=st.session_state.normalize_features,
            test_size=st.session_state.test_size,
            progress_callback=update_progress
        )

        st.session_state.training_results_dict = results
        st.session_state.model_trained_flag = True

        progress_bar.progress(100)
        status_text.text("所有模型训练完成！")
        st.success("模型训练完成！")

    except Exception as e:
        st.error(f"训练过程中出错: {str(e)}")
        progress_bar.progress(0)
        status_text.text("训练失败")


def display_results_summary():
    """显示所有模型的结果摘要"""
    # 创建结果比较表格
    results_data = []
    for model_type, results in st.session_state.training_results_dict.items():
        if 'error' not in results:
            row = {
                '模型': REGRESSOR_INFO[model_type]['name'],
                '训练R²': f"{results.get('train_r2', 0):.4f}",
                '测试R²': f"{results.get('test_r2', 0):.4f}",
                '训练MSE': f"{results.get('train_mse', 0):.4f}",
                '测试MSE': f"{results.get('test_mse', 0):.4f}",
                '测试MAE': f"{results.get('test_mae', 0):.4f}"
            }
            results_data.append(row)

    if results_data:
        df = pd.DataFrame(results_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # 找出最佳模型
        best_model_idx = df['测试R²'].apply(lambda x: float(x)).idxmax()
        best_model = df.iloc[best_model_idx]['模型']
        st.success(f"🏆 最佳模型（按测试R²）：{best_model}")


def create_results_section():
    """创建结果展示部分UI"""
    st.header("结果展示与分析")

    if not st.session_state.model_trained_flag or not st.session_state.training_results_dict:
        st.info("请先在模型训练选项卡中训练模型以查看结果。")
        return

    # 创建子选项卡
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 模型比较", "📈 详细结果", "🔍 特征分析", "📉 可视化", "💾 导出"
    ])

    with tab1:
        create_model_comparison_tab()

    with tab2:
        create_detailed_results_tab()

    with tab3:
        create_feature_analysis_tab()

    with tab4:
        create_visualization_tab()

    with tab5:
        create_export_tab()


def create_model_comparison_tab():
    """创建模型比较选项卡"""
    st.subheader("模型性能比较")

    # 收集所有模型的指标
    comparison_data = []
    for model_type, results in st.session_state.training_results_dict.items():
        if 'error' not in results:
            metrics = {
                '模型': REGRESSOR_INFO[model_type]['name'],
                '训练R²': results.get('train_r2', 0),
                '测试R²': results.get('test_r2', 0),
                '训练MSE': results.get('train_mse', 0),
                '测试MSE': results.get('test_mse', 0),
                '训练MAE': results.get('train_mae', 0),
                '测试MAE': results.get('test_mae', 0)
            }
            comparison_data.append(metrics)

    if comparison_data:
        df = pd.DataFrame(comparison_data)

        # 创建比较图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # R²比较
        ax = axes[0, 0]
        x = range(len(df))
        width = 0.35
        ax.bar([i - width / 2 for i in x], df['训练R²'], width, label='训练集', alpha=0.8, color='#3498db')
        ax.bar([i + width / 2 for i in x], df['测试R²'], width, label='测试集', alpha=0.8, color='#e74c3c')
        ax.set_xlabel('模型')
        ax.set_ylabel('R² 分数')
        ax.set_title('R² 分数比较')
        ax.set_xticks(x)
        ax.set_xticklabels(df['模型'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # MSE比较
        ax = axes[0, 1]
        ax.bar([i - width / 2 for i in x], df['训练MSE'], width, label='训练集', alpha=0.8, color='#3498db')
        ax.bar([i + width / 2 for i in x], df['测试MSE'], width, label='测试集', alpha=0.8, color='#e74c3c')
        ax.set_xlabel('模型')
        ax.set_ylabel('均方误差 (MSE)')
        ax.set_title('MSE比较')
        ax.set_xticks(x)
        ax.set_xticklabels(df['模型'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 雷达图 - 测试集指标
        ax = axes[1, 0]
        categories = ['R²', 'MSE', 'MAE']
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        ax = plt.subplot(2, 2, 3, projection='polar')
        for idx, row in df.iterrows():
            # 标准化指标用于雷达图
            r2_norm = row['测试R²']
            mse_norm = 1 / (1 + row['测试MSE'])  # MSE越小越好，转换为越大越好
            mae_norm = 1 / (1 + row['测试MAE'])  # MAE越小越好，转换为越大越好

            values = [r2_norm, mse_norm, mae_norm]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=row['模型'])
            ax.fill(angles, values, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_title('测试集性能雷达图')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)

        # 散点图 - 训练vs测试R²
        ax = axes[1, 1]
        colors = plt.cm.Set3(np.linspace(0, 1, len(df)))
        for idx, row in df.iterrows():
            ax.scatter(row['训练R²'], row['测试R²'],
                       s=200, c=[colors[idx]], alpha=0.6, edgecolors='black', linewidth=2)
            ax.annotate(row['模型'], (row['训练R²'], row['测试R²']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)

        # 添加对角线
        max_r2 = max(df['训练R²'].max(), df['测试R²'].max())
        min_r2 = min(df['训练R²'].min(), df['测试R²'].min())
        ax.plot([min_r2, max_r2], [min_r2, max_r2], 'k--', alpha=0.5)
        ax.set_xlabel('训练R²')
        ax.set_ylabel('测试R²')
        ax.set_title('训练vs测试R²')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        # 显示详细比较表格
        st.markdown("### 详细指标对比")
        st.dataframe(df.style.highlight_max(axis=0, subset=[col for col in df.columns if col != '模型']),
                     use_container_width=True)


def create_detailed_results_tab():
    """创建详细结果选项卡"""
    st.subheader("模型详细结果")

    # 选择要查看的模型
    available_models = [model_type for model_type, results in st.session_state.training_results_dict.items()
                        if 'error' not in results]

    if not available_models:
        st.warning("没有可用的训练结果")
        return

    selected_model = st.selectbox(
        "选择要查看详细结果的模型",
        available_models,
        format_func=lambda x: REGRESSOR_INFO[x]['name']
    )

    if selected_model:
        results = st.session_state.training_results_dict[selected_model]

        # 显示性能指标
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 训练集指标")
            metrics_train = {
                "R² 分数": f"{results.get('train_r2', 0):.4f}",
                "均方误差 (MSE)": f"{results.get('train_mse', 0):.4f}",
                "平均绝对误差 (MAE)": f"{results.get('train_mae', 0):.4f}"
            }
            for metric, value in metrics_train.items():
                st.metric(metric, value)

        with col2:
            st.markdown("### 测试集指标")
            metrics_test = {
                "R² 分数": f"{results.get('test_r2', 0):.4f}",
                "均方误差 (MSE)": f"{results.get('test_mse', 0):.4f}",
                "平均绝对误差 (MAE)": f"{results.get('test_mae', 0):.4f}"
            }
            for metric, value in metrics_test.items():
                st.metric(metric, value)

        # 显示模型参数 - 改进的处理
        st.markdown("### 模型参数")
        if 'params' in results and results['params']:
            # 将参数转换为更好的显示格式
            params_to_display = {}
            for key, value in results['params'].items():
                if value is None:
                    params_to_display[key] = "None"
                elif isinstance(value, float):
                    params_to_display[key] = f"{value:.4f}"
                else:
                    params_to_display[key] = str(value)

            params_df = pd.DataFrame(list(params_to_display.items()), columns=['参数', '值'])
            st.dataframe(params_df, hide_index=True, use_container_width=True)
        else:
            # 如果没有训练时的参数，显示默认参数
            default_params = st.session_state.multi_trainer.get_default_params(selected_model)
            if default_params:
                st.info("显示默认参数（训练时可能使用了自定义参数）")
                params_to_display = {}
                for key, value in default_params.items():
                    if value is None:
                        params_to_display[key] = "None"
                    elif isinstance(value, float):
                        params_to_display[key] = f"{value:.4f}"
                    else:
                        params_to_display[key] = str(value)

                params_df = pd.DataFrame(list(params_to_display.items()), columns=['参数', '值'])
                st.dataframe(params_df, hide_index=True, use_container_width=True)
            else:
                st.info("该模型没有可调参数或使用默认设置")

        # 显示模型信息
        model_info = REGRESSOR_INFO.get(selected_model, {})
        if model_info:
            with st.expander("模型说明", expanded=False):
                st.markdown(f"**描述：** {model_info.get('description', '无描述')}")
                st.markdown(f"**适用场景：** {model_info.get('suitable_for', '无说明')}")


def create_feature_analysis_tab():
    """创建特征分析选项卡"""
    st.subheader("特征重要性分析")

    # 收集有特征重要性的模型
    models_with_importance = []
    for model_type, results in st.session_state.training_results_dict.items():
        if 'error' not in results and results.get('feature_importance'):
            models_with_importance.append(model_type)

    if not models_with_importance:
        st.info("当前训练的模型中没有支持特征重要性分析的模型")
        st.warning("决策树、随机森林、梯度提升等模型支持特征重要性分析")
        return

    # 选择模型
    selected_model = st.selectbox(
        "选择模型查看特征重要性",
        models_with_importance,
        format_func=lambda x: REGRESSOR_INFO[x]['name']
    )

    if selected_model:
        results = st.session_state.training_results_dict[selected_model]
        feature_importance = results['feature_importance']

        # 绘制特征重要性图
        fig = plot_feature_importance(feature_importance, top_n=20)
        st.pyplot(fig)

        # 显示特征重要性表格
        st.markdown("### 特征重要性详细数据")
        importance_df = pd.DataFrame({
            '特征': list(feature_importance.keys()),
            '重要性': list(feature_importance.values())
        }).sort_values(by='重要性', ascending=False)

        st.dataframe(importance_df, use_container_width=True, hide_index=True)

        # 下载特征重要性数据
        csv = importance_df.to_csv(index=False)
        st.download_button(
            label="下载特征重要性数据 (CSV)",
            data=csv,
            file_name=f"feature_importance_{selected_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv'
        )


def create_visualization_tab():
    """创建可视化选项卡"""
    st.subheader("模型可视化分析")

    # 选择模型
    available_models = [model_type for model_type, results in st.session_state.training_results_dict.items()
                        if 'error' not in results]

    if not available_models:
        st.warning("没有可用的模型结果")
        return

    selected_model = st.selectbox(
        "选择要可视化的模型",
        available_models,
        format_func=lambda x: REGRESSOR_INFO[x]['name'],
        key="viz_model_select"
    )

    if selected_model:
        results = st.session_state.training_results_dict[selected_model]

        # 获取数据
        y_full = results.get('y_full')
        pred_full = results.get('pred_full')
        index_full = results.get('index_full')

        # 选择可视化类型
        viz_options = ["预测值vs真实值", "残差分析"]

        viz_type = st.selectbox("选择可视化类型", viz_options, key="viz_type_select")

        # 生成可视化
        try:
            if viz_type == "预测值vs真实值":
                fig = plot_training_results(y_full, pred_full, index_full)
            elif viz_type == "残差分析":
                fig = plot_residuals(y_full, pred_full, index_full)

            st.pyplot(fig)

            # 提供下载选项
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            st.download_button(
                label=f"下载{viz_type}图片",
                data=buf,
                file_name=f"{viz_type}_{selected_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime='image/png'
            )
            plt.close(fig)

        except Exception as e:
            st.error(f"生成可视化时出错: {str(e)}")


def create_export_tab():
    """创建模型导出选项卡"""
    st.subheader("模型导出与部署")

    # 选择要导出的模型
    available_models = [model_type for model_type, results in st.session_state.training_results_dict.items()
                        if 'error' not in results]

    if not available_models:
        st.warning("没有可导出的模型")
        return

    selected_model = st.selectbox(
        "选择要导出的模型",
        available_models,
        format_func=lambda x: REGRESSOR_INFO[x]['name'],
        key="export_model_select"
    )

    if selected_model:
        results = st.session_state.training_results_dict[selected_model]

        # 显示模型信息
        st.info(f"""
        **模型类型**: {REGRESSOR_INFO[selected_model]['name']}  
        **测试R²**: {results.get('test_r2', 0):.4f}  
        **测试MSE**: {results.get('test_mse', 0):.4f}
        """)

        # 导出选项
        col1, col2 = st.columns(2)

        with col1:
            # 导出模型文件
            if st.button("导出模型文件 (.joblib)", key="export_joblib"):
                try:
                    # 准备导出数据
                    export_data = {
                        'model': results['model'],
                        'model_type': selected_model,
                        'feature_names': st.session_state.selected_input_columns,
                        'scaler': st.session_state.multi_trainer.scaler,
                        'params': results.get('params'),
                        'metrics': {
                            'test_r2': results.get('test_r2'),
                            'test_mse': results.get('test_mse'),
                            'test_mae': results.get('test_mae')
                        },
                        'export_time': datetime.now().isoformat()
                    }

                    # 保存到临时文件
                    temp_file = BytesIO()
                    joblib.dump(export_data, temp_file)
                    temp_file.seek(0)

                    # 提供下载
                    st.download_button(
                        label="下载模型文件",
                        data=temp_file,
                        file_name=f"model_{selected_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib",
                        mime='application/octet-stream'
                    )
                    st.success("模型已准备好下载！")

                except Exception as e:
                    st.error(f"导出模型时出错: {str(e)}")

        with col2:
            # 导出预测结果
            if st.button("导出预测结果 (.csv)", key="export_predictions"):
                try:
                    # 创建预测结果DataFrame
                    pred_df = pd.DataFrame({
                        '真实值': results['y_full'],
                        '预测值': results['pred_full']
                    })

                    # 转换为CSV
                    csv = pred_df.to_csv(index=False)

                    st.download_button(
                        label="下载预测结果",
                        data=csv,
                        file_name=f"predictions_{selected_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime='text/csv'
                    )
                    st.success("预测结果已准备好下载！")

                except Exception as e:
                    st.error(f"导出预测结果时出错: {str(e)}")


# --- 绘图函数 ---
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
    return ax


def plot_training_results(y_true, predictions, indices=None):
    """绘制训练结果对比图"""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=80)
    apply_plot_style(ax)
    font_kwargs = {'fontproperties': FONT_PROP} if FONT_PROP else {}

    try:
        if indices is None:
            indices = np.arange(len(y_true))

        # 更安全的数据转换
        if isinstance(y_true, pd.Series):
            y_true_np = y_true.values
            y_true_name = y_true.name
        else:
            y_true_np = np.asarray(y_true)
            y_true_name = "真实值"

        if isinstance(predictions, pd.Series):
            predictions_np = predictions.values
        else:
            predictions_np = np.asarray(predictions)

        if isinstance(indices, pd.Series):
            indices_np = indices.values
        else:
            indices_np = np.asarray(indices)

        # 数据验证
        print(f"绘图数据统计:")
        print(f"y_true: min={y_true_np.min()}, max={y_true_np.max()}, mean={y_true_np.mean()}")
        print(f"predictions: min={predictions_np.min()}, max={predictions_np.max()}, mean={predictions_np.mean()}")

        if len(y_true_np) != len(predictions_np) or len(y_true_np) != len(indices_np):
            raise ValueError(
                f"绘图数据长度不匹配: y_true={len(y_true_np)}, pred={len(predictions_np)}, indices={len(indices_np)}")

        # 检查是否所有值都为0
        if np.all(y_true_np == 0) and np.all(predictions_np == 0):
            ax.text(0.5, 0.5, '警告：所有数据值都为0\n请检查数据处理过程',
                    ha='center', va='center', color='red', fontsize=12, **font_kwargs)
            return fig

        sort_order = np.argsort(indices_np)
        sorted_indices = indices_np[sort_order]
        sorted_true = y_true_np[sort_order]
        sorted_pred = predictions_np[sort_order]

        ax.plot(sorted_indices, sorted_true, color='#2ecc71', label='真实值', lw=1.5, marker='o', ms=3, alpha=0.8)
        ax.plot(sorted_indices, sorted_pred, color='#e74c3c', label='预测值', lw=1.5, ls='--', marker='x', ms=4,
                alpha=0.8)

        ax.set_xlabel('样本索引 (原始顺序)', **font_kwargs)
        ax.set_ylabel('值', **font_kwargs)
        ax.set_title('模型预测值 vs 真实值', **font_kwargs)
        legend = ax.legend(prop=FONT_PROP, fontsize=9)
        plt.tight_layout()

    except Exception as e:
        print(f"绘制训练结果图时出错: {e}")
        ax.text(0.5, 0.5, f'绘图错误: {e}', ha='center', va='center', color='red', **font_kwargs)

    return fig


def plot_residuals(y_true, predictions, indices=None):
    """绘制残差图"""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=80)
    apply_plot_style(ax)
    font_kwargs = {'fontproperties': FONT_PROP} if FONT_PROP else {}

    try:
        if indices is None:
            indices = np.arange(len(y_true))

        y_true_np = y_true.values if isinstance(y_true, pd.Series) else np.asarray(y_true)
        predictions_np = predictions.values if isinstance(predictions, pd.Series) else np.asarray(predictions)
        indices_np = indices.values if isinstance(indices, pd.Series) else np.asarray(indices)

        if len(y_true_np) != len(predictions_np) or len(y_true_np) != len(indices_np):
            raise ValueError("绘图数据长度不匹配")

        residuals = y_true_np - predictions_np
        sort_order = np.argsort(indices_np)
        sorted_indices = indices_np[sort_order]
        sorted_residuals = residuals[sort_order]

        ax.plot(sorted_indices, sorted_residuals, color='#3498db', label='残差', lw=1.5, marker='.', ms=3, alpha=0.8)
        ax.axhline(y=0, color='#e74c3c', linestyle='--', lw=1.0, alpha=0.7)

        ax.set_xlabel('样本索引 (原始顺序)', **font_kwargs)
        ax.set_ylabel('残差 (真实值 - 预测值)', **font_kwargs)
        ax.set_title('残差分析图', **font_kwargs)
        legend = ax.legend(prop=FONT_PROP, fontsize=9)
        plt.tight_layout()

    except Exception as e:
        print(f"绘制残差图时出错: {e}")
        ax.text(0.5, 0.5, f'绘图错误: {e}', ha='center', va='center', color='red', **font_kwargs)

    return fig


def plot_feature_importance(feature_importance_dict, top_n=20):
    """绘制特征重要性图"""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=80)
    apply_plot_style(ax)
    font_kwargs = {'fontproperties': FONT_PROP} if FONT_PROP else {}

    if not feature_importance_dict:
        ax.text(0.5, 0.5, '无特征重要性数据', ha='center', va='center', color='#7f8c8d', **font_kwargs)
        return fig

    try:
        sorted_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_importance[:top_n]
        features = [x[0] for x in top_features]
        importances = [x[1] for x in top_features]

        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, importances, align='center', color='#3498db', alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, **font_kwargs)
        ax.invert_yaxis()
        ax.set_xlabel('重要性值', **font_kwargs)
        ax.set_title(f'特征重要性 (Top {min(top_n, len(features))})', **font_kwargs)

        for bar in bars:
            width = bar.get_width()
            ax.text(width * 1.01, bar.get_y() + bar.get_height() / 2., f'{width:.4f}',
                    va='center', ha='left', fontsize=8)

        ax.set_xlim(right=max(importances) * 1.15)
        plt.tight_layout()

    except Exception as e:
        print(f"绘制特征重要性图时出错: {e}")
        ax.text(0.5, 0.5, f'绘图错误: {e}', ha='center', va='center', color='red', **font_kwargs)

    return fig


# --- 数据处理函数 ---
def process_folder_data(folder_path, progress_callback=None):
    """处理文件夹数据，整合多个CSV/Excel文件"""
    try:
        file_paths = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.csv', '.xlsx', '.xls')):
                    file_paths.append(os.path.join(root, file))

        if not file_paths:
            return None, "未在文件夹中找到CSV或Excel文件。"

        all_data, group_labels = [], []
        total_files = len(file_paths)

        for i, file_path in enumerate(file_paths):
            if progress_callback:
                progress_callback(int((i + 1) / total_files * 100))

            try:
                df = pd.read_csv(file_path) if file_path.lower().endswith('.csv') else pd.read_excel(file_path)
                if df.empty:
                    continue

                for col in df.select_dtypes(include=['object']).columns:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='ignore')
                    except:
                        pass

                df.dropna(axis=1, how='all', inplace=True)
                df.dropna(inplace=True)

                if df.empty:
                    continue

                numeric_df = df.select_dtypes(include=np.number)
                if numeric_df.empty:
                    continue

                all_data.append(numeric_df)
                group_label = os.path.splitext(os.path.basename(file_path))[0]
                group_labels.extend([group_label] * len(numeric_df))

            except Exception as e:
                st.warning(f"处理文件 {os.path.basename(file_path)} 时跳过，错误: {e}")

        if not all_data:
            return None, "未找到有效的数据文件或所有文件处理失败。"

        common_columns = set(all_data[0].columns)
        for df in all_data[1:]:
            common_columns.intersection_update(set(df.columns))

        if not common_columns:
            return None, "文件之间没有共同的数值列，无法合并数据。"

        common_columns = list(common_columns)
        filtered_data = [df[common_columns] for df in all_data]
        X = pd.concat(filtered_data, ignore_index=True)
        groups = pd.Series(group_labels, name='group', index=X.index)

        return {'X': X, 'groups': groups}, None

    except Exception as e:
        import traceback
        return None, f"处理文件夹时出错: {str(e)}\n{traceback.format_exc()}"


# --- 主函数入口 ---
if __name__ == "__main__":
    show_regression_training_page()