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
import matplotlib.font_manager as fm
import platform
import joblib
# 在其他导入语句后添加
from io import BytesIO
import base64
# For classification algorithms
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE, SelectFromModel
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import warnings # 添加导入
import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
import scipy.stats as stats
from scipy.stats import zscore
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

# --- 全局设置 ---
# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning, message="Glyph.*? missing from current font")
warnings.filterwarnings("ignore", category=FutureWarning)

# 在文件开头添加分类器信息配置
CLASSIFIER_INFO = {
    "decision_tree": {
        "name": "决策树 (Decision Tree)",
        "description": "基于特征对数据进行递归分割的树形结构算法",
        "advantages": [
            "易于理解和解释，可以可视化",
            "不需要数据预处理（如标准化）",
            "可以处理数值型和类别型特征",
            "能够处理多分类问题"
        ],
        "disadvantages": [
            "容易过拟合",
            "对噪声数据敏感",
            "决策边界是轴平行的"
        ],
        "suitable_for": "中小型数据集，需要模型可解释性的场景"
    },
    "random_forest": {
        "name": "随机森林 (Random Forest)",
        "description": "集成多个决策树，通过投票机制提高准确性",
        "advantages": [
            "准确率高，不容易过拟合",
            "可以评估特征重要性",
            "能够处理高维数据",
            "对缺失数据不敏感"
        ],
        "disadvantages": [
            "模型较大，预测速度相对较慢",
            "在某些噪声较大的数据上可能过拟合"
        ],
        "suitable_for": "大多数分类任务，特别适合需要特征重要性分析的场景"
    },
    "adaboost": {
        "name": "自适应增强 (AdaBoost)",
        "description": "通过组合多个弱分类器构建强分类器",
        "advantages": [
            "能够提高弱分类器的性能",
            "不容易过拟合",
            "可以使用各种类型的分类器作为弱分类器"
        ],
        "disadvantages": [
            "对噪声和异常值敏感",
            "训练时间可能较长"
        ],
        "suitable_for": "二分类问题，数据质量较高的场景"
    },
    "gradient_boosting": {
        "name": "梯度提升树 (GBDT)",
        "description": "通过逐步改进的方式构建集成模型",
        "advantages": [
            "预测准确率高",
            "能够处理非线性关系",
            "对异常值的鲁棒性较好"
        ],
        "disadvantages": [
            "训练时间长",
            "需要仔细调参以避免过拟合",
            "难以并行化"
        ],
        "suitable_for": "竞赛和需要高精度的场景"
    },
    "catboost": {
        "name": "CatBoost",
        "description": "专门处理类别特征的梯度提升算法",
        "advantages": [
            "自动处理类别特征",
            "减少过拟合",
            "训练速度快",
            "默认参数表现良好"
        ],
        "disadvantages": [
            "模型文件较大",
            "对于纯数值特征可能不如XGBoost"
        ],
        "suitable_for": "包含大量类别特征的数据集"
    },
    "xgboost": {
        "name": "极限梯度提升 (XGBoost)",
        "description": "高效的梯度提升实现",
        "advantages": [
            "速度快，性能高",
            "内置正则化减少过拟合",
            "可以处理缺失值",
            "支持并行计算"
        ],
        "disadvantages": [
            "参数较多，调参复杂",
            "对类别特征需要预处理"
        ],
        "suitable_for": "大规模数据集，竞赛常用"
    },
    "lightgbm": {
        "name": "轻量级梯度提升 (LightGBM)",
        "description": "基于直方图的高效梯度提升算法",
        "advantages": [
            "训练速度极快",
            "内存使用少",
            "准确率高",
            "支持类别特征"
        ],
        "disadvantages": [
            "可能在小数据集上过拟合",
            "对噪声敏感"
        ],
        "suitable_for": "大规模数据集，需要快速训练的场景"
    },
    "extra_trees": {
        "name": "极端随机树 (Extra Trees)",
        "description": "比随机森林更随机的集成方法",
        "advantages": [
            "训练速度比随机森林快",
            "减少过拟合",
            "在某些数据集上表现优于随机森林"
        ],
        "disadvantages": [
            "可能需要更多的树来达到相同精度",
            "模型解释性较差"
        ],
        "suitable_for": "高维数据，需要快速训练的场景"
    },
    "knn": {
        "name": "K近邻 (KNN)",
        "description": "基于实例的懒惰学习算法",
        "advantages": [
            "简单直观，易于理解",
            "对非线性数据效果好",
            "可以处理多分类问题"
        ],
        "disadvantages": [
            "计算成本高（预测慢）",
            "对高维数据效果差（维度灾难）",
            "对不平衡数据敏感"
        ],
        "suitable_for": "小型数据集，特征维度不高的场景"
    },
    "svm": {
        "name": "支持向量机 (SVM)",
        "description": "寻找最优分类超平面的算法",
        "advantages": [
            "在高维空间表现良好",
            "使用核技巧可处理非线性问题",
            "泛化能力强"
        ],
        "disadvantages": [
            "大规模数据训练慢",
            "对参数和核函数选择敏感",
            "难以处理多分类（需要一对一或一对多）"
        ],
        "suitable_for": "中小型数据集，特别是高维数据"
    },
    "neural_network": {
        "name": "神经网络 (MLP)",
        "description": "多层感知器，深度学习的基础",
        "advantages": [
            "可以学习非线性关系",
            "适应性强",
            "可以处理大规模特征"
        ],
        "disadvantages": [
            "需要大量数据",
            "训练时间长",
            "难以解释",
            "需要调整多个超参数"
        ],
        "suitable_for": "大型复杂数据集，非线性关系强的场景"
    },
    "naive_bayes": {
        "name": "朴素贝叶斯",
        "description": "基于贝叶斯定理的概率分类器",
        "advantages": [
            "训练和预测速度快",
            "对小数据集表现良好",
            "可以处理多分类问题",
            "对缺失数据不敏感"
        ],
        "disadvantages": [
            "假设特征相互独立（现实中很少）",
            "对输入数据的分布敏感"
        ],
        "suitable_for": "文本分类、垃圾邮件过滤等"
    },
    "logistic_regression": {
        "name": "逻辑回归",
        "description": "线性分类模型，输出概率",
        "advantages": [
            "简单快速",
            "可解释性强",
            "不需要调整很多超参数",
            "输出概率便于阈值调整"
        ],
        "disadvantages": [
            "只能处理线性可分问题",
            "对特征缩放敏感",
            "容易欠拟合"
        ],
        "suitable_for": "线性可分问题，需要概率输出的场景"
        },
    "bp_neural_network": {
        "name": "BP神经网络 (Backpropagation)",
        "description": "基于反向传播算法的多层感知器神经网络",
        "advantages": [
            "可以学习复杂的非线性关系",
            "适用于各种分类问题",
            "可以自动学习特征表示",
            "支持多层结构"
        ],
        "disadvantages": [
            "容易过拟合",
            "需要大量数据",
            "训练时间较长",
            "需要调整很多超参数"
        ],
        "suitable_for": "中大型数据集，复杂分类问题"
    },
    "rnn": {
        "name": "循环神经网络 (RNN)",
        "description": "专门处理序列数据的神经网络",
        "advantages": [
            "可以处理变长序列",
            "具有记忆功能",
            "适合时序数据",
            "参数共享"
        ],
        "disadvantages": [
            "梯度消失问题",
            "训练速度慢",
            "难以处理长序列",
            "对数据格式要求高"
        ],
        "suitable_for": "时序数据、序列分类问题"
    },
    "cnn": {
        "name": "卷积神经网络 (CNN)",
        "description": "使用卷积操作提取特征的神经网络",
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
        "suitable_for": "图像数据、空间结构数据、特征具有局部相关性的数据"
    },
    "lstm": {
        "name": "长短期记忆网络 (LSTM)",
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
        "suitable_for": "长时序数据、需要长期记忆的序列问题"
    },
    "gru": {
        "name": "门控循环单元 (GRU)",
        "description": "LSTM的简化版本，计算效率更高",
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
        "suitable_for": "中长序列数据、计算资源有限的序列分类"
    }
}

# --- 数据预处理配置 ---
PREPROCESSING_OPTIONS = {
    "outlier_detection": {
        "name": "异常值检测",
        "methods": {
            "isolation_forest": "孤立森林",
            "elliptic_envelope": "椭圆包络",
            "one_class_svm": "单类SVM",
            "z_score": "Z分数法",
            "iqr": "四分位距法"
        }
    },
    "feature_scaling": {
        "name": "特征缩放",
        "methods": {
            "standard": "标准化 (StandardScaler)",
            "minmax": "最小-最大缩放",
            "robust": "鲁棒缩放",
            "power": "幂变换",
            "quantile": "分位数变换"
        }
    },
    "feature_selection": {
        "name": "特征选择",
        "methods": {
            "k_best": "K最佳特征",
            "mutual_info": "互信息",
            "rfe": "递归特征消除",
            "model_based": "基于模型的选择",
            "pca": "主成分分析"
        }
    },
    "imbalance_handling": {
        "name": "不平衡数据处理",
        "methods": {
            "smote": "SMOTE过采样",
            "adasyn": "ADASYN过采样",
            "borderline_smote": "边界SMOTE",
            "random_under": "随机欠采样",
            "tomek": "Tomek链接",
            "edited_nn": "编辑最近邻",
            "smote_enn": "SMOTE+ENN",
            "smote_tomek": "SMOTE+Tomek"
        }
    },
    "missing_value_handling": {
        "name": "缺失值处理",
        "methods": {
            "drop": "删除缺失值",
            "mean": "均值填充",
            "median": "中位数填充",
            "mode": "众数填充",
            "knn": "KNN填充",
            "forward_fill": "前向填充",
            "backward_fill": "后向填充"
        }
    }
}

# --- 字体设置 ---
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
            # 尝试查找字体路径
            font_path = fm.findfont(fm.FontProperties(family=font_name), fallback_to_default=False) # 不回退到默认
            # 检查路径是否有效且不是 DejaVuSans (有时会导致问题)
            if font_path and os.path.exists(font_path) and 'DejaVuSans' not in font_path:
                print(f"字体日志: 使用字体 '{font_name}' 在路径: {font_path}")
                # 设置 sans-serif 字体族，将找到的字体放在列表最前面
                plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
                # 显式设置 font.family 为 sans-serif
                plt.rcParams['font.family'] = 'sans-serif'
                font_prop = fm.FontProperties(fname=font_path) # 使用路径创建 FontProperties
                font_found = True
                break # 找到合适的字体后退出
        except Exception as e:
            print(f"字体日志: 尝试字体 {font_name} 失败: {e}")

    if not font_found:
        print("字体日志: 警告: 未找到支持中文的字体，将使用系统默认字体，中文可能无法正常显示。")
        # 尝试设置一个通用字体作为后备
        try:
            plt.rcParams['font.family'] = 'sans-serif'
            print("字体日志: 已设置字体族为 'sans-serif' 作为后备。")
        except Exception as e_fallback:
            print(f"字体日志: 设置后备字体失败: {e_fallback}")


    # 修复负号显示
    plt.rcParams['axes.unicode_minus'] = False

    return font_prop


# 使用改进的字体设置
FONT_PROP = setup_better_chinese_font()

# 设置英文字体 (作为中文后备)
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial'] + plt.rcParams['font.sans-serif']


# --- 辅助绘图函数 ---
def apply_plot_style(ax):
    """应用统一的绘图样式"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.7, color='#bdc3c7') # 调整网格颜色
    ax.tick_params(axis='both', which='major', labelsize=9, colors='#34495e') # 调整刻度颜色
    ax.xaxis.label.set_fontsize(10)
    ax.yaxis.label.set_fontsize(10)
    ax.title.set_fontsize(12)
    ax.title.set_fontweight('bold')
    ax.xaxis.label.set_fontweight('bold')
    ax.yaxis.label.set_fontweight('bold')

    # 设置标签和标题字体 (如果 FONT_PROP 有效)
    font_kwargs = {'fontproperties': FONT_PROP} if FONT_PROP else {}
    if font_kwargs:
        ax.xaxis.label.set_fontproperties(FONT_PROP)
        ax.yaxis.label.set_fontproperties(FONT_PROP)
        ax.title.set_fontproperties(FONT_PROP)
        # 设置刻度标签字体
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(FONT_PROP)

    return ax


def create_figure_with_safe_dimensions(width_inches, height_inches, max_dpi=80):
    """创建不会超出Matplotlib限制的图形尺寸"""
    max_pixels = 65000
    width_dpi = max_pixels / width_inches if width_inches > 0 else max_dpi
    height_dpi = max_pixels / height_inches if height_inches > 0 else max_dpi
    safe_dpi = min(width_dpi, height_dpi, max_dpi)
    fig, ax = plt.subplots(figsize=(width_inches, height_inches), dpi=safe_dpi)
    return fig, ax


# --- 绘图函数 ---
def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """绘制混淆矩阵"""
    fig, ax = create_figure_with_safe_dimensions(10, 8)
    font_kwargs = {'fontproperties': FONT_PROP} if FONT_PROP else {}

    try:
        cm = confusion_matrix(y_true, y_pred)

        if class_names is None:
            # 尝试从 y_true 和 y_pred 中推断类别名称
            unique_labels = sorted(list(set(list(y_true) + list(y_pred))))
            class_names = [str(label) for label in unique_labels]
            if len(class_names) != cm.shape[0]: # 如果推断不一致，回退到数字
                 class_names = [str(i) for i in range(cm.shape[0])]

        # 创建热力图
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax, annot_kws={"size": 8}) # 减小注释字体

        ax.set_xlabel('预测类别', fontsize=10, fontweight='bold', **font_kwargs)
        ax.set_ylabel('真实类别', fontsize=10, fontweight='bold', **font_kwargs)
        ax.set_title('混淆矩阵', fontsize=12, fontweight='bold', **font_kwargs)

        # 旋转标签以提高可读性
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9, **font_kwargs) # 减小刻度字体
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9, **font_kwargs) # 减小刻度字体

        apply_plot_style(ax) # 应用统一样式 (在设置标签后)
        plt.tight_layout()
        return fig

    except Exception as e:
        print(f"绘制混淆矩阵时出错: {e}")
        ax.text(0.5, 0.5, f'绘制混淆矩阵时出错: {e}',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12, color='red', **font_kwargs)
        return fig


def plot_feature_importance(feature_importance_dict, top_n=20):
    """绘制特征重要性"""
    fig, ax = create_figure_with_safe_dimensions(10, 8)
    font_kwargs = {'fontproperties': FONT_PROP} if FONT_PROP else {}

    try:
        if not feature_importance_dict:
            ax.text(0.5, 0.5, '无可用特征重要性数据',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12, color='#7f8c8d', **font_kwargs)
            return fig

        # 按重要性排序
        sorted_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

        # 获取前N个特征
        top_features = sorted_importance[:top_n]
        features = [x[0] for x in top_features]
        importances = [x[1] for x in top_features]

        y_pos = range(len(features))

        # 绘制水平条形图
        bars = ax.barh(y_pos, importances, align='center', color='#3498db', alpha=0.8)

        # 在条形右侧添加值标签
        for i, bar in enumerate(bars):
            ax.text(bar.get_width() + bar.get_width() * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f'{importances[i]:.4f}',
                    va='center', fontsize=8)

        # 设置y轴刻度标签
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=9, **font_kwargs) # 减小刻度字体
        ax.invert_yaxis() # 重要性高的在顶部

        # 将x轴限制在略高于最大重要性值处
        ax.set_xlim(0, max(importances) * 1.15) # 留出空间给标签

        # 标签和标题
        ax.set_xlabel('重要性', fontsize=10, fontweight='bold', **font_kwargs)
        ax.set_title(f'特征重要性 (Top {min(top_n, len(features))})',
                     fontsize=12, fontweight='bold', **font_kwargs)

        apply_plot_style(ax) # 应用统一样式
        plt.tight_layout()
        return fig

    except Exception as e:
        print(f"绘制特征重要性时出错: {e}")
        ax.text(0.5, 0.5, f'绘制特征重要性时出错: {e}',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12, color='red', **font_kwargs)
        return fig


def plot_roc_curve(y_true, y_proba, class_names=None):
    """绘制ROC曲线"""
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    fig, ax = create_figure_with_safe_dimensions(10, 8)
    font_kwargs = {'fontproperties': FONT_PROP} if FONT_PROP else {}

    try:
        # 获取类别数
        n_classes = y_proba.shape[1]

        if class_names is None:
            # 尝试从 y_true 推断类别名称
            unique_labels = sorted(list(set(y_true)))
            class_names = [str(label) for label in unique_labels]
            if len(class_names) != n_classes: # 如果推断不一致，回退到数字
                 class_names = [str(i) for i in range(n_classes)]


        # 将真实标签进行二值化（one-hot编码）
        # 需要提供 classes 参数以处理标签不从0开始或不连续的情况
        all_possible_labels = sorted(list(set(y_true)))
        y_bin = label_binarize(y_true, classes=all_possible_labels)

        # 如果是二分类，label_binarize 只返回一列，需要手动构造两列
        if n_classes == 2 and y_bin.shape[1] == 1:
            y_bin = np.hstack((1 - y_bin, y_bin))
        # 如果多分类但 y_bin 列数仍不匹配 y_proba (可能因为某些类没出现在 y_true 中)
        elif y_bin.shape[1] != n_classes:
             # 这是一个复杂情况，可能需要重新映射或发出警告
             print(f"警告: ROC曲线的真实标签二值化后的列数 ({y_bin.shape[1]}) 与预测概率的列数 ({n_classes}) 不匹配。曲线可能不准确。")
             # 尝试基于 all_possible_labels 重建 y_bin
             y_bin_corrected = np.zeros((len(y_true), n_classes))
             for idx, label in enumerate(y_true):
                 if label in all_possible_labels:
                      class_idx = all_possible_labels.index(label)
                      # 确保 class_idx 在 y_bin_corrected 的范围内
                      if class_idx < y_bin_corrected.shape[1]:
                           y_bin_corrected[idx, class_idx] = 1
             y_bin = y_bin_corrected


        # 为每个类别计算ROC曲线和AUC值
        colors = plt.cm.get_cmap('tab10', n_classes) # 使用tab10颜色映射

        for i in range(n_classes):
             # 检查 y_bin 是否有足够的列
             if i < y_bin.shape[1]:
                 fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                 roc_auc = auc(fpr, tpr)

                 # 确保 class_names 有对应的名称
                 label_name = class_names[i] if i < len(class_names) else f'类别 {i}'

                 ax.plot(fpr, tpr, color=colors(i), lw=2,
                         label=f'{label_name} (AUC = {roc_auc:.3f})')
             else:
                 print(f"警告: 类别 {i} 在二值化标签中缺失，无法绘制其ROC曲线。")


        # 添加对角线
        ax.plot([0, 1], [0, 1], 'k--', lw=1)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('假阳性率', fontsize=10, fontweight='bold', **font_kwargs)
        ax.set_ylabel('真阳性率', fontsize=10, fontweight='bold', **font_kwargs)
        ax.set_title('ROC曲线', fontsize=12, fontweight='bold', **font_kwargs)

        # 添加图例
        legend = ax.legend(loc="lower right", fontsize=8)
        if FONT_PROP:
            for text in legend.get_texts():
                text.set_fontproperties(FONT_PROP)

        apply_plot_style(ax) # 应用统一样式
        plt.tight_layout()
        return fig

    except Exception as e:
        print(f"绘制ROC曲线时出错: {e}")
        import traceback
        print(traceback.format_exc()) # 打印详细错误
        ax.text(0.5, 0.5, f'绘制ROC曲线时出错: {e}',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12, color='red', **font_kwargs)
        return fig


def plot_class_distribution(y_true, y_pred, class_names=None):
    """绘制类别分布比较（真实值与预测值）"""
    fig, ax = create_figure_with_safe_dimensions(10, 8)
    font_kwargs = {'fontproperties': FONT_PROP} if FONT_PROP else {}

    try:
        # 确定所有类别
        all_labels = sorted(list(set(list(y_true) + list(y_pred))))

        if class_names is None:
            class_names = [str(label) for label in all_labels]
        elif len(class_names) != len(all_labels):
             print("警告: 提供的 class_names 数量与数据中的类别数量不符，将使用数据中的类别。")
             class_names = [str(label) for label in all_labels]

        # 创建类别到名称的映射，以防类别不是简单数字
        label_to_name = dict(zip(all_labels, class_names))

        # 统计每个类别在真实和预测中的出现次数
        true_counts = pd.Series(y_true).value_counts().reindex(all_labels, fill_value=0)
        pred_counts = pd.Series(y_pred).value_counts().reindex(all_labels, fill_value=0)

        # 设置条形的位置
        x = np.arange(len(all_labels))
        width = 0.35

        # 创建条形
        rects1 = ax.bar(x - width / 2, true_counts.values, width, label='真实分布', color='#2ecc71', alpha=0.8)
        rects2 = ax.bar(x + width / 2, pred_counts.values, width, label='预测分布', color='#e74c3c', alpha=0.8)

        # 添加标签、标题和图例
        ax.set_xlabel('类别', fontsize=10, fontweight='bold', **font_kwargs)
        ax.set_ylabel('样本数量', fontsize=10, fontweight='bold', **font_kwargs)
        ax.set_title('类别分布对比', fontsize=12, fontweight='bold', **font_kwargs)
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9, **font_kwargs) # 使用 class_names

        legend = ax.legend(frameon=True, framealpha=0.9, edgecolor='#bdc3c7', # 调整图例边框颜色
                           prop=FONT_PROP, fontsize=9)

        # 在条形顶部添加计数数字
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8) # 减小标签字体

        autolabel(rects1)
        autolabel(rects2)

        apply_plot_style(ax) # 应用统一样式
        plt.tight_layout()
        return fig

    except Exception as e:
        print(f"绘制类别分布时出错: {e}")
        ax.text(0.5, 0.5, f'绘制类别分布时出错: {e}',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12, color='red', **font_kwargs)
        return fig


# --- 数据预处理增强函数 ---
def detect_outliers(X, method='isolation_forest', contamination=0.1, **kwargs):
    """检测异常值"""
    try:
        if method == 'isolation_forest':
            detector = IsolationForest(contamination=contamination, random_state=42)
            outliers = detector.fit_predict(X) == -1
        elif method == 'elliptic_envelope':
            detector = EllipticEnvelope(contamination=contamination, random_state=42)
            outliers = detector.fit_predict(X) == -1
        elif method == 'one_class_svm':
            detector = OneClassSVM(nu=contamination)
            outliers = detector.fit_predict(X) == -1
        elif method == 'z_score':
            threshold = kwargs.get('threshold', 3)
            z_scores = np.abs(zscore(X, axis=0, nan_policy='omit'))
            outliers = (z_scores > threshold).any(axis=1)
        elif method == 'iqr':
            Q1 = X.quantile(0.25)
            Q3 = X.quantile(0.75)
            IQR = Q3 - Q1
            multiplier = kwargs.get('multiplier', 1.5)
            outliers = ((X < (Q1 - multiplier * IQR)) | (X > (Q3 + multiplier * IQR))).any(axis=1)
        else:
            raise ValueError(f"不支持的异常值检测方法: {method}")

        return outliers
    except Exception as e:
        st.error(f"异常值检测出错: {str(e)}")
        return np.zeros(len(X), dtype=bool)

def apply_feature_scaling(X, method='standard', **kwargs):
    """应用特征缩放"""
    try:
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'power':
            scaler = PowerTransformer(method='yeo-johnson')
        elif method == 'quantile':
            scaler = QuantileTransformer(output_distribution='normal')
        else:
            raise ValueError(f"不支持的特征缩放方法: {method}")

        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        return X_scaled, scaler
    except Exception as e:
        st.error(f"特征缩放出错: {str(e)}")
        return X, None

def apply_feature_selection(X, y, method='k_best', n_features=10, **kwargs):
    """应用特征选择"""
    try:
        if method == 'k_best':
            selector = SelectKBest(score_func=f_classif, k=min(n_features, X.shape[1]))
        elif method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=min(n_features, X.shape[1]))
        elif method == 'rfe':
            estimator = kwargs.get('estimator', RandomForestClassifier(n_estimators=50, random_state=42))
            selector = RFE(estimator=estimator, n_features_to_select=min(n_features, X.shape[1]))
        elif method == 'model_based':
            estimator = kwargs.get('estimator', RandomForestClassifier(n_estimators=50, random_state=42))
            selector = SelectFromModel(estimator=estimator, max_features=min(n_features, X.shape[1]))
        elif method == 'pca':
            selector = PCA(n_components=min(n_features, X.shape[1]))
        else:
            raise ValueError(f"不支持的特征选择方法: {method}")

        if method == 'pca':
            X_selected = pd.DataFrame(
                selector.fit_transform(X, y),
                columns=[f'PC{i+1}' for i in range(selector.n_components_)],
                index=X.index
            )
        else:
            X_selected = pd.DataFrame(
                selector.fit_transform(X, y),
                columns=X.columns[selector.get_support()],
                index=X.index
            )

        return X_selected, selector
    except Exception as e:
        st.error(f"特征选择出错: {str(e)}")
        return X, None

def handle_imbalanced_data(X, y, method='smote', **kwargs):
    """处理不平衡数据"""
    try:
        if method == 'smote':
            sampler = SMOTE(random_state=42, **kwargs)
        elif method == 'adasyn':
            sampler = ADASYN(random_state=42, **kwargs)
        elif method == 'borderline_smote':
            sampler = BorderlineSMOTE(random_state=42, **kwargs)
        elif method == 'random_under':
            sampler = RandomUnderSampler(random_state=42, **kwargs)
        elif method == 'tomek':
            sampler = TomekLinks(**kwargs)
        elif method == 'edited_nn':
            sampler = EditedNearestNeighbours(**kwargs)
        elif method == 'smote_enn':
            sampler = SMOTEENN(random_state=42, **kwargs)
        elif method == 'smote_tomek':
            sampler = SMOTETomek(random_state=42, **kwargs)
        else:
            raise ValueError(f"不支持的不平衡数据处理方法: {method}")

        X_resampled, y_resampled = sampler.fit_resample(X, y)

        # 转换回DataFrame和Series
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        y_resampled = pd.Series(y_resampled, name=y.name)

        return X_resampled, y_resampled
    except Exception as e:
        st.error(f"不平衡数据处理出错: {str(e)}")
        return X, y

def handle_missing_values(X, method='mean', **kwargs):
    """处理缺失值"""
    try:
        if method == 'drop':
            X_filled = X.dropna()
        elif method in ['mean', 'median', 'most_frequent']:
            strategy = 'mean' if method == 'mean' else 'median' if method == 'median' else 'most_frequent'
            imputer = SimpleImputer(strategy=strategy)
            X_filled = pd.DataFrame(
                imputer.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        elif method == 'knn':
            n_neighbors = kwargs.get('n_neighbors', 5)
            imputer = KNNImputer(n_neighbors=n_neighbors)
            X_filled = pd.DataFrame(
                imputer.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        elif method == 'forward_fill':
            X_filled = X.fillna(method='ffill')
        elif method == 'backward_fill':
            X_filled = X.fillna(method='bfill')
        else:
            raise ValueError(f"不支持的缺失值处理方法: {method}")

        return X_filled
    except Exception as e:
        st.error(f"缺失值处理出错: {str(e)}")
        return X

# --- 模型训练与评估函数 ---
# 修复 train_model 函数中序列模型的数据处理逻辑
def train_model(X_train, X_test, y_train, y_test, model_type='catboost', params=None):
    """训练分类模型并返回结果"""
    if params is None:
        params = {}

    random_state = params.get('random_state', 42)
    model_args = {'random_state': random_state}

    try:
        # 创建分类器实例
        if model_type == 'decision_tree':
            dt_params = {k: params[k] for k in ['max_depth', 'min_samples_split', 'min_samples_leaf', 'criterion'] if
                         k in params}
            model = DecisionTreeClassifier(**dt_params, **model_args)

        elif model_type == 'random_forest':
            rf_params = {k: params[k] for k in ['n_estimators', 'max_depth', 'min_samples_split'] if k in params}
            model = RandomForestClassifier(**rf_params, **model_args)

        elif model_type == 'adaboost':
            ada_params = {k: params[k] for k in ['n_estimators', 'learning_rate', 'algorithm'] if k in params}
            model = AdaBoostClassifier(**ada_params, **model_args)

        elif model_type == 'gradient_boosting':
            gb_params = {k: params[k] for k in ['n_estimators', 'learning_rate', 'max_depth', 'subsample'] if
                         k in params}
            model = GradientBoostingClassifier(**gb_params, **model_args)

        elif model_type == 'catboost':
            cat_params = {k: params[k] for k in ['iterations', 'learning_rate', 'depth'] if k in params}
            model = CatBoostClassifier(
                loss_function='MultiClass',
                verbose=0,
                **cat_params,
                **model_args
            )

        elif model_type == 'xgboost':
            xgb_params = {k: params[k] for k in ['n_estimators', 'learning_rate', 'max_depth'] if k in params}
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            y_test_encoded = le.transform(y_test)
            num_class = len(le.classes_)
            model = XGBClassifier(
                objective='multi:softprob',
                num_class=num_class,
                eval_metric='mlogloss',
                use_label_encoder=False,
                **xgb_params,
                **model_args
            )

        elif model_type == 'lightgbm':
            lgb_params = {k: params[k] for k in ['n_estimators', 'learning_rate', 'num_leaves', 'max_depth'] if
                          k in params}
            model = LGBMClassifier(**lgb_params, **model_args)

        elif model_type == 'extra_trees':
            et_params = {k: params[k] for k in ['n_estimators', 'max_depth', 'min_samples_split'] if k in params}
            model = ExtraTreesClassifier(**et_params, **model_args)

        elif model_type == 'knn':
            knn_params = {k: params[k] for k in ['n_neighbors', 'weights', 'algorithm', 'p'] if k in params}
            model = KNeighborsClassifier(**knn_params)

        elif model_type == 'svm':
            svm_params = {k: params[k] for k in ['C', 'kernel', 'gamma'] if k in params}
            model = SVC(probability=True, **svm_params, **model_args)

        elif model_type == 'neural_network':
            nn_params = {k: params[k] for k in
                         ['hidden_layer_sizes', 'activation', 'solver', 'alpha', 'learning_rate_init', 'max_iter'] if
                         k in params}
            model = MLPClassifier(**nn_params, **model_args)

        elif model_type == 'naive_bayes':
            nb_type = params.get('nb_type', 'gaussian')
            if nb_type == 'gaussian':
                model = GaussianNB()
            else:
                model = MultinomialNB(**{k: params[k] for k in ['alpha'] if k in params})

        elif model_type == 'logistic_regression':
            lr_params = {k: params[k] for k in ['C', 'penalty', 'solver', 'max_iter'] if k in params}
            model = LogisticRegression(**lr_params, **model_args)

        elif model_type in ['bp_neural_network', 'rnn', 'cnn', 'lstm', 'gru']:
            # 深度学习模型处理
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            y_test_encoded = le.transform(y_test)
            num_classes = len(le.classes_)

            # 数据预处理
            if model_type == 'bp_neural_network':
                # BP神经网络使用标准的表格数据
                X_train_processed = X_train.values.astype(np.float32)
                X_test_processed = X_test.values.astype(np.float32)
                y_train_processed = y_train_encoded
                y_test_processed = y_test_encoded
                input_shape = X_train_processed.shape[1]
                model = build_bp_neural_network(input_shape, num_classes, params)

            elif model_type in ['rnn', 'lstm', 'gru']:
                # 序列模型需要重塑数据
                sequence_length = params.get('sequence_length', 10)

                # 确保数据长度足够
                if len(X_train) >= sequence_length and len(X_test) >= sequence_length:
                    # 方法1：滑动窗口（适合时序相关数据）
                    X_train_seq = []
                    y_train_seq = []
                    for i in range(len(X_train) - sequence_length + 1):
                        X_train_seq.append(X_train.iloc[i:i + sequence_length].values)
                        y_train_seq.append(y_train_encoded[i + sequence_length - 1])

                    X_train_processed = np.array(X_train_seq).astype(np.float32)
                    y_train_processed = np.array(y_train_seq)

                    # 同样处理测试集
                    X_test_seq = []
                    y_test_seq = []
                    for i in range(len(X_test) - sequence_length + 1):
                        X_test_seq.append(X_test.iloc[i:i + sequence_length].values)
                        y_test_seq.append(y_test_encoded[i + sequence_length - 1])

                    X_test_processed = np.array(X_test_seq).astype(np.float32)
                    y_test_processed = np.array(y_test_seq)
                else:
                    # 方法2：简单重塑（适合非时序数据）
                    n_features = X_train.shape[1]
                    new_seq_length = min(sequence_length, n_features)

                    # 如果特征数少于序列长度，进行填充
                    if n_features < sequence_length:
                        X_train_padded = np.pad(X_train.values, ((0, 0), (0, sequence_length - n_features)),
                                                mode='constant')
                        X_test_padded = np.pad(X_test.values, ((0, 0), (0, sequence_length - n_features)),
                                               mode='constant')
                    else:
                        X_train_padded = X_train.values[:, :sequence_length]
                        X_test_padded = X_test.values[:, :sequence_length]

                    X_train_processed = X_train_padded.reshape(-1, sequence_length, 1).astype(np.float32)
                    X_test_processed = X_test_padded.reshape(-1, sequence_length, 1).astype(np.float32)
                    y_train_processed = y_train_encoded
                    y_test_processed = y_test_encoded

                input_shape = (X_train_processed.shape[1], X_train_processed.shape[2])

                # 构建对应的模型
                if model_type == 'rnn':
                    model = build_rnn_model(input_shape, num_classes, params)
                elif model_type == 'lstm':
                    model = build_lstm_model(input_shape, num_classes, params)
                elif model_type == 'gru':
                    model = build_gru_model(input_shape, num_classes, params)

            elif model_type == 'cnn':
                # CNN数据准备
                X_train_processed = prepare_cnn_data(X_train)
                X_test_processed = prepare_cnn_data(X_test)
                y_train_processed = y_train_encoded
                y_test_processed = y_test_encoded
                input_shape = (X_train_processed.shape[1], X_train_processed.shape[2])
                model = build_cnn_model(input_shape, num_classes, params)

            # 验证数据形状一致性
            print(f"训练数据形状: X={X_train_processed.shape}, y={y_train_processed.shape}")
            print(f"测试数据形状: X={X_test_processed.shape}, y={y_test_processed.shape}")

            # 确保样本数量匹配
            assert X_train_processed.shape[0] == y_train_processed.shape[
                0], f"训练集样本数不匹配: X={X_train_processed.shape[0]}, y={y_train_processed.shape[0]}"
            assert X_test_processed.shape[0] == y_test_processed.shape[
                0], f"测试集样本数不匹配: X={X_test_processed.shape[0]}, y={y_test_processed.shape[0]}"

            # 设置回调函数
            callbacks = []
            if params.get('early_stopping', True):
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=params.get('patience', 10),
                    restore_best_weights=True
                )
                callbacks.append(early_stopping)

            # 训练模型
            try:
                history = model.fit(
                    X_train_processed, y_train_processed,
                    validation_data=(X_test_processed, y_test_processed),
                    epochs=params.get('epochs', 100),
                    batch_size=params.get('batch_size', 32),
                    callbacks=callbacks,
                    verbose=0
                )

                # 预测
                y_train_pred_proba = model.predict(X_train_processed, verbose=0)
                y_test_pred_proba = model.predict(X_test_processed, verbose=0)

                # 转换预测结果
                if num_classes == 2:
                    y_train_pred = (y_train_pred_proba > 0.5).astype(int).flatten()
                    y_test_pred = (y_test_pred_proba > 0.5).astype(int).flatten()
                    # 为二分类创建概率矩阵
                    y_train_proba = np.column_stack([1 - y_train_pred_proba.flatten(), y_train_pred_proba.flatten()])
                    y_test_proba = np.column_stack([1 - y_test_pred_proba.flatten(), y_test_pred_proba.flatten()])
                else:
                    y_train_pred = np.argmax(y_train_pred_proba, axis=1)
                    y_test_pred = np.argmax(y_test_pred_proba, axis=1)
                    y_train_proba = y_train_pred_proba
                    y_test_proba = y_test_pred_proba

                # 转换回原始标签
                y_train_pred = le.inverse_transform(y_train_pred)
                y_test_pred = le.inverse_transform(y_test_pred)
                class_names = list(le.classes_)

                # 调整原始标签以匹配处理后的数据长度（重要修复！）
                if model_type in ['rnn', 'lstm', 'gru'] and len(X_train) >= params.get('sequence_length', 10):
                    # 对于滑动窗口方法，需要调整原始标签
                    y_train_original = y_train.iloc[params.get('sequence_length', 10) - 1:].reset_index(drop=True)
                    y_test_original = y_test.iloc[params.get('sequence_length', 10) - 1:].reset_index(drop=True)
                else:
                    y_train_original = y_train
                    y_test_original = y_test

            except Exception as e:
                print(f"深度学习模型训练出错: {e}")
                return {
                    'error': str(e),
                    'model_type': model_type,
                    'params': params,
                    'model': None
                }

        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        # 训练模型（对于传统机器学习模型）
        if model_type not in ['bp_neural_network', 'rnn', 'cnn', 'lstm', 'gru']:
            if model_type == 'xgboost':
                model.fit(X_train, y_train_encoded)
                y_train_pred = le.inverse_transform(model.predict(X_train))
                y_test_pred = le.inverse_transform(model.predict(X_test))
                y_train_proba = model.predict_proba(X_train)
                y_test_proba = model.predict_proba(X_test)
                class_names = list(le.classes_)
                # 对于传统模型，使用原始标签
                y_train_original = y_train
                y_test_original = y_test
            else:
                model.fit(X_train, y_train)
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                y_train_proba = None
                y_test_proba = None
                if hasattr(model, 'predict_proba'):
                    try:
                        y_train_proba = model.predict_proba(X_train)
                        y_test_proba = model.predict_proba(X_test)
                    except Exception:
                        pass
                if hasattr(model, 'classes_'):
                    class_names = list(model.classes_)
                else:
                    class_names = sorted(list(set(y_train) | set(y_test)))
                # 对于传统模型，使用原始标签
                y_train_original = y_train
                y_test_original = y_test

        # 计算评估指标
        avg_method = 'weighted'
        train_accuracy = accuracy_score(y_train_original, y_train_pred)
        test_accuracy = accuracy_score(y_test_original, y_test_pred)
        train_precision = precision_score(y_train_original, y_train_pred, average=avg_method, zero_division=0)
        test_precision = precision_score(y_test_original, y_test_pred, average=avg_method, zero_division=0)
        train_recall = recall_score(y_train_original, y_train_pred, average=avg_method, zero_division=0)
        test_recall = recall_score(y_test_original, y_test_pred, average=avg_method, zero_division=0)
        train_f1 = f1_score(y_train_original, y_train_pred, average=avg_method, zero_division=0)
        test_f1 = f1_score(y_test_original, y_test_pred, average=avg_method, zero_division=0)

        # 获取特征重要性
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            if len(X_train.columns) == len(importance):
                feature_importance = dict(zip(X_train.columns, importance))
        elif model_type == 'logistic_regression' and hasattr(model, 'coef_'):
            importance = np.abs(model.coef_).mean(axis=0)
            if len(X_train.columns) == len(importance):
                feature_importance = dict(zip(X_train.columns, importance))

        # 返回结果
        results = {
            'model': model,
            'model_type': model_type,
            'params': params,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_precision': train_precision,
            'test_precision': test_precision,
            'train_recall': train_recall,
            'test_recall': test_recall,
            'train_f1': train_f1,
            'test_f1': test_f1,
            'feature_importance': feature_importance,
            'confusion_matrix': confusion_matrix(y_test_original, y_test_pred),
            'class_names': class_names,
            'y_train': y_train_original,
            'y_train_pred': y_train_pred,
            'y_test': y_test_original,
            'y_test_pred': y_test_pred,
            'y_train_proba': y_train_proba,
            'y_test_proba': y_test_proba
        }

        return results

    except Exception as e:
        import traceback
        print(f"模型 {model_type} 训练/评估过程中发生错误: {str(e)}")
        print(traceback.format_exc())
        return {
            'error': str(e),
            'model_type': model_type,
            'params': params,
            'model': None
        }


def perform_cross_validation(X, y, model_type='catboost', params=None, cv=5):
    """使用交叉验证评估模型性能"""
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    if params is None:
        params = {}

    random_state = params.get('random_state', 42)
    model_args = {'random_state': random_state}
    avg_method = 'weighted' # 用于多分类指标

    try:
        # 根据模型类型创建分类器 (与 train_model 类似)
        if model_type == 'catboost':
            cat_params = {k: params[k] for k in ['iterations', 'learning_rate', 'depth'] if k in params}
            model = CatBoostClassifier(loss_function='MultiClass', verbose=0, **cat_params, **model_args)
        elif model_type == 'random_forest':
            rf_params = {k: params[k] for k in ['n_estimators', 'max_depth', 'min_samples_split'] if k in params}
            model = RandomForestClassifier(**rf_params, **model_args)
        elif model_type == 'svm':
            svm_params = {k: params[k] for k in ['C', 'kernel', 'gamma'] if k in params}
            # SVM CV 可能很慢，特别是对于大数据集
            model = SVC(probability=True, **svm_params, **model_args)
        elif model_type == 'xgboost':
            xgb_params = {k: params[k] for k in ['n_estimators', 'learning_rate', 'max_depth'] if k in params}
            le = LabelEncoder()
            y_encoded = le.fit_transform(y) # 在整个 y 上编码
            num_class = len(le.classes_)
            model = XGBClassifier(
                 objective='multi:softprob', num_class=num_class, eval_metric='mlogloss',
                 use_label_encoder=False, **xgb_params, **model_args
                 )
            # XGBoost CV 时需要使用编码后的 y
            y_cv = y_encoded
        elif model_type == 'neural_network':
            nn_params = {k: params[k] for k in ['hidden_layer_sizes', 'activation', 'solver', 'alpha', 'learning_rate_init', 'max_iter'] if k in params}
            if 'learning_rate' in params and 'learning_rate_init' not in nn_params:
                 nn_params['learning_rate_init'] = params['learning_rate']
            model = MLPClassifier(**nn_params, **model_args)
        elif model_type in ['bp_neural_network', 'rnn', 'cnn', 'lstm', 'gru']:
            # 深度学习模型的交叉验证比较复杂，这里提供简化版本
            # 由于深度学习模型训练时间较长，建议使用较少的折数

            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

            # 手动实现交叉验证
            cv_strategy = StratifiedKFold(n_splits=min(cv, 3), shuffle=True, random_state=random_state)  # 最多3折

            cv_scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

            for fold, (train_idx, val_idx) in enumerate(cv_strategy.split(X, y_encoded)):
                try:
                    X_train_fold = X.iloc[train_idx]
                    X_val_fold = X.iloc[val_idx]
                    y_train_fold = y_encoded[train_idx]
                    y_val_fold = y_encoded[val_idx]

                    # 数据预处理（根据模型类型）
                    if model_type == 'bp_neural_network':
                        X_train_processed = X_train_fold.values.astype(np.float32)
                        X_val_processed = X_val_fold.values.astype(np.float32)
                        input_shape = X_train_processed.shape[1]
                        model = build_bp_neural_network(input_shape, len(le.classes_), params)

                    elif model_type in ['rnn', 'lstm', 'gru']:
                        sequence_length = params.get('sequence_length', 10)
                        if len(X_train_fold) >= sequence_length:
                            # 滑动窗口方法
                            X_train_seq = []
                            y_train_seq = []
                            for i in range(len(X_train_fold) - sequence_length + 1):
                                X_train_seq.append(X_train_fold.iloc[i:i + sequence_length].values)
                                y_train_seq.append(y_train_fold[i + sequence_length - 1])
                            X_train_processed = np.array(X_train_seq).astype(np.float32)
                            y_train_fold = np.array(y_train_seq)

                            X_val_seq = []
                            y_val_seq = []
                            for i in range(len(X_val_fold) - sequence_length + 1):
                                X_val_seq.append(X_val_fold.iloc[i:i + sequence_length].values)
                                y_val_seq.append(y_val_fold[i + sequence_length - 1])
                            X_val_processed = np.array(X_val_seq).astype(np.float32)
                            y_val_fold = np.array(y_val_seq)
                        else:
                            # 简单重塑
                            n_features = X_train_fold.shape[1]
                            new_seq_length = min(sequence_length, n_features)
                            X_train_processed = X_train_fold.values[:, :new_seq_length].reshape(-1, new_seq_length,
                                                                                                1).astype(np.float32)
                            X_val_processed = X_val_fold.values[:, :new_seq_length].reshape(-1, new_seq_length,
                                                                                            1).astype(np.float32)

                        input_shape = (X_train_processed.shape[1], X_train_processed.shape[2])

                        if model_type == 'rnn':
                            model = build_rnn_model(input_shape, len(le.classes_), params)
                        elif model_type == 'lstm':
                            model = build_lstm_model(input_shape, len(le.classes_), params)
                        elif model_type == 'gru':
                            model = build_gru_model(input_shape, len(le.classes_), params)

                    elif model_type == 'cnn':
                        X_train_processed = prepare_cnn_data(X_train_fold)
                        X_val_processed = prepare_cnn_data(X_val_fold)
                        input_shape = (X_train_processed.shape[1], X_train_processed.shape[2])
                        model = build_cnn_model(input_shape, len(le.classes_), params)

                    # 训练模型（减少epochs以加快交叉验证）
                    model.fit(
                        X_train_processed, y_train_fold,
                        epochs=min(params.get('epochs', 100), 50),  # 最多50个epoch
                        batch_size=params.get('batch_size', 32),
                        verbose=0
                    )

                    # 预测和评估
                    y_pred_proba = model.predict(X_val_processed, verbose=0)
                    if len(le.classes_) == 2:
                        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
                    else:
                        y_pred = np.argmax(y_pred_proba, axis=1)

                    # 计算指标
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    cv_scores['accuracy'].append(accuracy_score(y_val_fold, y_pred))
                    cv_scores['precision'].append(
                        precision_score(y_val_fold, y_pred, average='weighted', zero_division=0))
                    cv_scores['recall'].append(recall_score(y_val_fold, y_pred, average='weighted', zero_division=0))
                    cv_scores['f1'].append(f1_score(y_val_fold, y_pred, average='weighted', zero_division=0))

                except Exception as fold_error:
                    print(f"交叉验证第{fold + 1}折出错: {fold_error}")
                    # 使用NaN填充失败的折
                    for metric in cv_scores:
                        cv_scores[metric].append(np.nan)

            # 转换为numpy数组
            for metric in cv_scores:
                cv_scores[metric] = np.array(cv_scores[metric])

        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        # 使用 StratifiedKFold 进行分类交叉验证，确保类别比例
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

        # 如果是 XGBoost，使用编码后的标签进行 CV
        y_target_for_cv = y_cv if model_type == 'xgboost' else y

        # 执行交叉验证
        # 使用 try-except 包裹每个 cross_val_score 以防某个指标计算失败
        cv_scores = {}
        scoring_metrics = {'accuracy': 'accuracy',
                           'precision': f'precision_{avg_method}',
                           'recall': f'recall_{avg_method}',
                           'f1': f'f1_{avg_method}'}

        for name, scorer in scoring_metrics.items():
            try:
                scores = cross_val_score(model, X, y_target_for_cv, cv=cv_strategy, scoring=scorer, n_jobs=-1) # 使用多核心
                cv_scores[name] = scores
            except Exception as cv_err:
                print(f"警告: 计算交叉验证指标 '{name}' 时出错: {cv_err}")
                cv_scores[name] = np.array([np.nan] * cv) # 填充 NaN

        # 返回交叉验证结果
        cv_results = {
            'cv_accuracy_mean': np.nanmean(cv_scores.get('accuracy', [])),
            'cv_accuracy_std': np.nanstd(cv_scores.get('accuracy', [])),
            'cv_precision_mean': np.nanmean(cv_scores.get('precision', [])),
            'cv_precision_std': np.nanstd(cv_scores.get('precision', [])),
            'cv_recall_mean': np.nanmean(cv_scores.get('recall', [])),
            'cv_recall_std': np.nanstd(cv_scores.get('recall', [])),
            'cv_f1_mean': np.nanmean(cv_scores.get('f1', [])),
            'cv_f1_std': np.nanstd(cv_scores.get('f1', [])),
            'cv_scores': cv_scores # 包含每个折叠的分数
        }

        return cv_results

    except Exception as outer_cv_err:
        print(f"交叉验证过程中发生错误: {outer_cv_err}")
        import traceback
        print(traceback.format_exc())
        # 返回一个包含错误信息的字典
        return {'error': str(outer_cv_err)}


def process_folder_data(folder_path, progress_callback=None):
    """处理包含子文件夹作为类别的文件夹数据"""
    try:
        # 检查文件夹是否有子文件夹
        subfolders = [f for f in os.listdir(folder_path)
                      if os.path.isdir(os.path.join(folder_path, f))]

        if not subfolders:
            return None, "所选文件夹没有包含子文件夹。分类训练需要每个类别有一个单独的子文件夹。"

        # 初始化数据收集
        all_data = []
        labels = []
        file_names = [] # 存储原始文件名

        # 处理每个子文件夹 (类别)
        folder_count = len(subfolders)
        for i, subfolder in enumerate(subfolders):
            subfolder_path = os.path.join(folder_path, subfolder)
            subfolder_files = []

            # 获取当前子文件夹下的所有CSV/Excel文件
            for ext in ['.csv', '.xlsx', '.xls']:
                try:
                    subfolder_files.extend([f for f in os.listdir(subfolder_path)
                                            if f.lower().endswith(ext) and os.path.isfile(os.path.join(subfolder_path, f))]) # 确保是文件
                except FileNotFoundError:
                     print(f"警告: 无法访问子文件夹 {subfolder_path}，已跳过。")
                     continue # 跳过这个子文件夹

            # 处理每个文件
            for file in subfolder_files:
                file_path = os.path.join(subfolder_path, file)
                try:
                    # 加载数据
                    df = pd.read_csv(file_path) if file.lower().endswith('.csv') else pd.read_excel(file_path)

                    # 跳过空文件
                    if df.empty: continue

                    # 尝试转换对象类型列为数值类型
                    for col in df.select_dtypes(include=['object']).columns:
                        try: df[col] = pd.to_numeric(df[col], errors='ignore')
                        except: pass

                    # 删除全为NaN的列
                    df.dropna(axis=1, how='all', inplace=True)
                    # 删除包含任何NaN的行 (重要!)
                    df.dropna(inplace=True)

                    # 清理后如果为空则跳过
                    if df.empty: continue

                    # 只选择数值列作为特征
                    numeric_df = df.select_dtypes(include=np.number)

                    # 如果没有数值列，则跳过
                    if numeric_df.empty: continue

                    # 添加到数据集合
                    all_data.append(numeric_df)
                    # 使用子文件夹名称作为标签，重复次数为当前文件的有效行数
                    labels.extend([subfolder] * len(numeric_df))
                    # 记录文件名
                    file_names.extend([file] * len(numeric_df))

                except Exception as e:
                    st.warning(f"处理文件 {file} 时出错: {e}，已跳过。") # 使用 st.warning

            # 更新进度
            if progress_callback:
                progress_percent = int((i + 1) / folder_count * 100)
                progress_callback(progress_percent)

        # 合并所有数据
        if not all_data:
            return None, "未找到有效的数据文件或所有文件处理失败。"

        # 找到所有数据框共有的数值列
        if not all_data: return None, "没有加载任何有效数据。" # 再次检查以防万一
        common_columns = set(all_data[0].columns)
        for df in all_data[1:]:
            common_columns.intersection_update(set(df.columns))

        if not common_columns:
            return None, "文件之间没有共同的数值列，无法合并数据。"
        common_columns = sorted(list(common_columns)) # 排序以保证一致性
        print(f"找到 {len(common_columns)} 个共同数值列: {common_columns}")

        # 过滤到公共列并连接
        filtered_data = [df[common_columns] for df in all_data]
        X = pd.concat(filtered_data, ignore_index=True)

        # 进行标签编码 (使用子文件夹名称)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(labels)
        y_series = pd.Series(y, name='label', index=X.index) # 确保索引对齐

        # 保存类别映射 (编码后的数字 -> 原始子文件夹名称)
        class_mapping = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))

        # 创建文件名Series (与 X 和 y 对齐)
        file_names_series = pd.Series(file_names, name='file_name', index=X.index)

        # 返回处理后的数据
        return {
            'X': X,
            'y': y_series,
            'file_names': file_names_series, # 包含文件名的 Series
            'class_mapping': class_mapping, # 数字到名称的映射
            'label_encoder': label_encoder, # 保存编码器
            'raw_labels': labels # 原始标签列表 (子文件夹名)
        }, None

    except Exception as e:
        import traceback
        return None, f"处理文件夹时出错: {str(e)}\n{traceback.format_exc()}"


# --- Streamlit UI 函数 ---
def initialize_classification_session_state():
    """初始化分类页面的会话状态变量 - 增强版"""
    defaults = {
        'classification_data': None,
        'column_names': [],
        'selected_input_columns': [],
        'selected_output_column': None,
        'data_source_type': 'file',
        'file_names': None,
        'training_results': None,
        'cv_results': None,
        'model_trained_flag': False,
        'normalize_features': True,
        'test_size': 0.2,
        'scaler': None,
        # 新增：支持多模型
        'selected_models': [],
        'model_params': {},
        'training_results_dict': {},
        # 原有的单模型参数（保留以兼容）
        'current_model_type': 'catboost',
        'use_cv': False,
        'cv_folds': 5,
        'columns_selected_flag': False,
        'temp_selected_input_columns': [],
        'temp_selected_output_column': None,
        'is_selecting_columns': False,
    }

    # 添加所有模型的默认参数
    for model_type in CLASSIFIER_INFO.keys():
        defaults[f'{model_type}_params'] = get_default_params(model_type)

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def show_classification_training_page():
    """显示分类训练页面"""
    initialize_classification_session_state() # 确保状态已初始化

    st.title("分类模型训练")
    st.markdown("---") # 添加分隔线

    # 创建选项卡
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📁 1. 数据导入",
        "📊 2. 特征选择",
        "🤖 3. 智能推荐",
        "⚙️ 4. 模型训练",
        "📈 5. 结果展示"
    ])

    with tab1:
        create_data_import_section()

    with tab2:
        create_column_selection_section()

    with tab3:
        create_smart_recommendation_section()

    with tab4:
        create_model_training_section()

    with tab5:
        create_results_section()


def create_data_import_section():
    """创建数据导入部分UI"""
    st.header("数据源选择")
    st.info("您可以上传单个包含特征和目标列的文件，或者上传一个包含子文件夹的文件夹，每个子文件夹代表一个类别，其中包含该类别的数据文件。")

    # 文件上传列
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("上传文件")
        uploaded_file = st.file_uploader("选择 CSV 或 Excel 文件", type=["csv", "xlsx", "xls"], key="clf_file_uploader")

        if uploaded_file:
             # 使用按钮触发加载，避免每次交互都重新加载
             if st.button("加载文件数据", key="load_file_clf_btn"):
                 with st.spinner(f"正在加载 {uploaded_file.name}..."):
                     try:
                         # 处理上传的文件
                         data = pd.read_csv(uploaded_file) if uploaded_file.name.lower().endswith('.csv') else pd.read_excel(uploaded_file)

                         # 基本清理
                         data.dropna(axis=1, how='all', inplace=True) # 删除全NaN列

                         if data.empty:
                             st.error("上传的文件为空或不包含有效数据。")
                         else:
                             # 在会话状态中存储数据
                             st.session_state.classification_data = data
                             st.session_state.column_names = list(data.columns)
                             st.session_state.data_source_type = "file"
                             st.session_state.file_names = None # 文件模式无文件名
                             # 清除旧结果和状态
                             st.session_state.training_results = None
                             st.session_state.cv_results = None
                             st.session_state.model_trained_flag = False
                             st.session_state.selected_input_columns = [] # 重置选择
                             st.session_state.selected_output_column = None

                             st.success(f"已成功加载: {uploaded_file.name} (包含 {len(data)} 行, {len(data.columns)} 列)")
                             st.rerun() # 重新运行以更新其他选项卡

                     except Exception as e:
                         st.error(f"加载数据时出错: {str(e)}")
                         st.session_state.classification_data = None # 出错时清空

    with col2:
        st.subheader("上传文件夹")
        folder_path = st.text_input("输入包含类别子文件夹的路径", key="clf_folder_path")

        if folder_path and os.path.isdir(folder_path):
            process_button = st.button("处理文件夹数据", key="process_folder_clf_btn")

            if process_button:
                # 显示进度条
                folder_progress = st.progress(0)
                status_text = st.empty()
                def update_folder_progress(p):
                    folder_progress.progress(p / 100)
                    status_text.text(f"正在处理文件夹... {p}%")

                with st.spinner("正在处理文件夹..."):
                    results, error_msg = process_folder_data(
                        folder_path,
                        progress_callback=update_folder_progress
                    )
                    folder_progress.progress(100) # 完成
                    status_text.text("文件夹处理完成。")

                    if results is not None:
                        # 在会话状态中存储处理后的数据字典
                        st.session_state.classification_data = results
                        st.session_state.column_names = list(results['X'].columns) # 特征列名
                        st.session_state.data_source_type = "folder"
                        st.session_state.file_names = results['file_names'] # 文件名Series
                        # 清除旧结果和状态
                        st.session_state.training_results = None
                        st.session_state.cv_results = None
                        st.session_state.model_trained_flag = False
                        st.session_state.selected_input_columns = list(results['X'].columns) # 默认全选特征
                        st.session_state.selected_output_column = 'label' # 文件夹模式下目标列固定为 'label'

                        st.success(
                            f"已成功加载文件夹数据: {len(results['X'])} 行, {len(results['class_mapping'])} 个类别。")

                        # 显示类别映射
                        st.info("类别映射 (数字标签 -> 类别名称):")
                        mapping_df = pd.DataFrame(
                            results['class_mapping'].items(),
                            columns=["类别ID", "类别名称"]
                        ).sort_values(by="类别ID")
                        st.dataframe(mapping_df, hide_index=True)
                        st.rerun() # 重新运行以更新其他选项卡

                    else:
                        st.error(f"处理文件夹时出错: {error_msg}")
                        st.session_state.classification_data = None # 出错时清空
        elif folder_path:
             st.warning("输入的路径不是一个有效的文件夹。")


    # 显示示例数据
    if st.session_state.classification_data is not None:
        st.markdown("---") # 添加分隔线
        st.subheader("数据预览 (前5行)")
        try:
            if st.session_state.data_source_type == "file":
                st.dataframe(st.session_state.classification_data.head())
            elif isinstance(st.session_state.classification_data, dict) and 'X' in st.session_state.classification_data:
                 # 文件夹模式下，显示特征 X 和 目标 y
                 preview_df = st.session_state.classification_data['X'].head().copy()
                 # 添加原始标签和编码后的标签以便预览
                 if 'raw_labels' in st.session_state.classification_data:
                      preview_df['原始类别'] = st.session_state.classification_data['raw_labels'][:5]
                 if 'y' in st.session_state.classification_data:
                      preview_df['编码标签 (y)'] = st.session_state.classification_data['y'].head()
                 st.dataframe(preview_df)
            else:
                 st.warning("无法预览数据，数据格式不符合预期。")
        except Exception as preview_e:
             st.error(f"预览数据时出错: {preview_e}")


def create_column_selection_section():
    """创建列选择部分UI"""
    st.header("特征和目标列选择")

    if st.session_state.classification_data is None:
        st.info("请先在数据导入选项卡中加载数据。")
        return

    all_columns = st.session_state.column_names
    if not all_columns:
        st.warning("未能从加载的数据中获取列名。")
        return

    st.info("请选择用于模型训练的输入特征列。对于文件上传模式，还需选择目标（类别标签）列。")

    # 初始化防止刷新问题的标志
    if 'columns_selected_flag' not in st.session_state:
        st.session_state.columns_selected_flag = False

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("输入特征 (X)")
        # 文件夹模式下，目标列 'label' 和 文件名 'file_name' 不应作为输入特征
        input_options = all_columns
        default_inputs = st.session_state.selected_input_columns
        if st.session_state.data_source_type == "folder":
            # 自动排除 'label' 和 'file_name' (如果存在于原始列名中)
            input_options = [col for col in all_columns if col not in ['label', 'file_name']]
            # 如果 session state 中的选择包含这些，也移除
            default_inputs = [col for col in default_inputs if col in input_options]

        # 禁用自动回调导致的重新渲染
        multiselect_key = "input_col_multi_clf"

        # 使用 multiselect 选择输入特征，但不直接更新 session_state
        selected_inputs = st.multiselect(
            "选择一个或多个输入特征列",
            input_options,  # 提供可选的特征列
            default=default_inputs,  # 保留之前的有效选择
            key=multiselect_key
        )

        if selected_inputs:
            st.write(f"已选择 {len(selected_inputs)} 个输入特征。")
        else:
            st.warning("请至少选择一个输入特征。")

    with col2:
        st.subheader("目标列 (Y - 类别标签)")
        if st.session_state.data_source_type == "file":
            # 文件模式下，用户需要选择目标列
            # 目标列不能是输入特征
            output_options = [col for col in all_columns if col not in selected_inputs]
            # 尝试保留之前的选择
            current_output = st.session_state.selected_output_column
            current_output_index = 0
            if current_output in output_options:
                current_output_index = output_options.index(current_output) + 1  # +1 for None

            # 禁用自动回调导致的重新渲染
            selectbox_key = "output_col_select_clf"

            selected_output = st.selectbox(
                "选择一个目标（类别标签）列",
                [None] + output_options,  # 添加 None 选项
                index=current_output_index,
                key=selectbox_key
            )

            if selected_output:
                st.write(f"已选择 '{selected_output}' 作为目标列。")
            else:
                st.warning("请选择一个目标列。")
        else:
            # 文件夹模式下，目标列是固定的
            st.info("文件夹模式下，目标（类别标签）已自动从子文件夹名称生成，内部表示为 'label' 列。")
            selected_output = 'label'  # 使用临时变量

    # 确认按钮 - 将选择保存到 session_state
    if st.button("✅ 确认特征选择", key="confirm_columns_btn", use_container_width=True):
        st.session_state.selected_input_columns = selected_inputs
        if st.session_state.data_source_type == "file":
            st.session_state.selected_output_column = selected_output
        else:
            st.session_state.selected_output_column = 'label'

        st.session_state.columns_selected_flag = True
        st.success("特征和目标列已确认！")
        time.sleep(0.5)

    st.markdown("---")
    st.subheader("数据预处理选项")

    # 创建预处理选项卡
    prep_tab1, prep_tab2, prep_tab3, prep_tab4 = st.tabs([
        "🔧 基础设置", "🔍 异常值处理", "⚖️ 特征工程", "📊 数据平衡"
    ])

    with prep_tab1:
        col_prep1, col_prep2 = st.columns(2)
        with col_prep1:
            # 特征缩放方法选择
            scaling_method = st.selectbox(
                "特征缩放方法",
                options=list(PREPROCESSING_OPTIONS["feature_scaling"]["methods"].keys()),
                format_func=lambda x: PREPROCESSING_OPTIONS["feature_scaling"]["methods"][x],
                index=0,
                key="scaling_method_select"
            )

            # 缺失值处理
            missing_method = st.selectbox(
                "缺失值处理方法",
                options=list(PREPROCESSING_OPTIONS["missing_value_handling"]["methods"].keys()),
                format_func=lambda x: PREPROCESSING_OPTIONS["missing_value_handling"]["methods"][x],
                index=1,  # 默认均值填充
                key="missing_method_select"
            )

        with col_prep2:
            # 测试集比例
            test_size = st.slider(
                "测试集比例",
                min_value=0.1,
                max_value=0.5,
                value=st.session_state.test_size,
                step=0.05,
                help="用于模型评估的数据比例。",
                key="test_size_slider_clf"
            )

            # 随机种子
            random_seed = st.number_input(
                "随机种子",
                min_value=0,
                max_value=9999,
                value=42,
                help="用于确保结果可重现",
                key="random_seed_input"
            )

    with prep_tab2:
        st.markdown("### 异常值检测与处理")

        enable_outlier_detection = st.checkbox(
            "启用异常值检测",
            value=False,
            key="enable_outlier_detection"
        )

        if enable_outlier_detection:
            col_out1, col_out2 = st.columns(2)
            with col_out1:
                outlier_method = st.selectbox(
                    "异常值检测方法",
                    options=list(PREPROCESSING_OPTIONS["outlier_detection"]["methods"].keys()),
                    format_func=lambda x: PREPROCESSING_OPTIONS["outlier_detection"]["methods"][x],
                    key="outlier_method_select"
                )

            with col_out2:
                if outlier_method in ['isolation_forest', 'elliptic_envelope', 'one_class_svm']:
                    contamination = st.slider(
                        "异常值比例",
                        min_value=0.01,
                        max_value=0.5,
                        value=0.1,
                        step=0.01,
                        key="contamination_slider"
                    )
                elif outlier_method == 'z_score':
                    z_threshold = st.slider(
                        "Z分数阈值",
                        min_value=1.0,
                        max_value=5.0,
                        value=3.0,
                        step=0.1,
                        key="z_threshold_slider"
                    )
                elif outlier_method == 'iqr':
                    iqr_multiplier = st.slider(
                        "IQR倍数",
                        min_value=1.0,
                        max_value=3.0,
                        value=1.5,
                        step=0.1,
                        key="iqr_multiplier_slider"
                    )

            outlier_action = st.radio(
                "异常值处理方式",
                options=["删除", "保留"],
                index=0,
                key="outlier_action_radio"
            )

    with prep_tab3:
        st.markdown("### 特征工程")

        enable_feature_selection = st.checkbox(
            "启用特征选择",
            value=False,
            key="enable_feature_selection"
        )

        if enable_feature_selection:
            col_feat1, col_feat2 = st.columns(2)
            with col_feat1:
                feature_selection_method = st.selectbox(
                    "特征选择方法",
                    options=list(PREPROCESSING_OPTIONS["feature_selection"]["methods"].keys()),
                    format_func=lambda x: PREPROCESSING_OPTIONS["feature_selection"]["methods"][x],
                    key="feature_selection_method"
                )

            with col_feat2:
                if st.session_state.selected_input_columns:
                    max_features = len(st.session_state.selected_input_columns)
                    n_features_to_select = st.slider(
                        "选择特征数量",
                        min_value=1,
                        max_value=max_features,
                        value=min(10, max_features),
                        key="n_features_slider"
                    )
                else:
                    st.info("请先选择输入特征")

    with prep_tab4:
        st.markdown("### 数据平衡处理")

        enable_imbalance_handling = st.checkbox(
            "启用不平衡数据处理",
            value=False,
            key="enable_imbalance_handling"
        )

        if enable_imbalance_handling:
            col_imb1, col_imb2 = st.columns(2)
            with col_imb1:
                imbalance_method = st.selectbox(
                    "数据平衡方法",
                    options=list(PREPROCESSING_OPTIONS["imbalance_handling"]["methods"].keys()),
                    format_func=lambda x: PREPROCESSING_OPTIONS["imbalance_handling"]["methods"][x],
                    key="imbalance_method_select"
                )

            with col_imb2:
                if imbalance_method in ['smote', 'adasyn', 'borderline_smote']:
                    sampling_strategy = st.selectbox(
                        "采样策略",
                        options=['auto', 'minority', 'not majority', 'all'],
                        index=0,
                        key="sampling_strategy_select"
                    )

    # 确认预处理设置
    if st.button("✅ 确认预处理设置", key="confirm_preproc_btn", use_container_width=True):
        # 保存所有预处理设置到session state
        st.session_state.preprocessing_config = {
            'scaling_method': scaling_method,
            'missing_method': missing_method,
            'test_size': test_size,
            'random_seed': random_seed,
            'enable_outlier_detection': enable_outlier_detection,
            'enable_feature_selection': enable_feature_selection,
            'enable_imbalance_handling': enable_imbalance_handling
        }

        if enable_outlier_detection:
            st.session_state.preprocessing_config['outlier_config'] = {
                'method': outlier_method,
                'action': outlier_action
            }
            if outlier_method in ['isolation_forest', 'elliptic_envelope', 'one_class_svm']:
                st.session_state.preprocessing_config['outlier_config']['contamination'] = contamination
            elif outlier_method == 'z_score':
                st.session_state.preprocessing_config['outlier_config']['threshold'] = z_threshold
            elif outlier_method == 'iqr':
                st.session_state.preprocessing_config['outlier_config']['multiplier'] = iqr_multiplier

        if enable_feature_selection:
            st.session_state.preprocessing_config['feature_selection_config'] = {
                'method': feature_selection_method,
                'n_features': n_features_to_select
            }

        if enable_imbalance_handling:
            st.session_state.preprocessing_config['imbalance_config'] = {
                'method': imbalance_method,
                'sampling_strategy': sampling_strategy if imbalance_method in ['smote', 'adasyn', 'borderline_smote'] else 'auto'
            }

        st.success("预处理设置已确认！")
        time.sleep(0.5)


def create_model_training_section():
    """创建模型训练选项部分UI - 优化版"""
    st.header("模型训练配置")

    # 前置检查
    data_loaded = st.session_state.classification_data is not None
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
    st.subheader("选择分类算法")

    # 使用columns布局
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 可用算法列表")
        # 显示所有可用的分类器
        for model_key, model_info in CLASSIFIER_INFO.items():
            if st.checkbox(model_info['name'], key=f"check_{model_key}"):
                if model_key not in st.session_state.selected_models:
                    st.session_state.selected_models.append(model_key)
                    # 初始化该模型的参数
                    if model_key not in st.session_state.model_params:
                        st.session_state.model_params[model_key] = get_default_params(model_key)
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
            model_tabs = st.tabs([CLASSIFIER_INFO[m]['name'] for m in st.session_state.selected_models])

            for i, (model_key, tab) in enumerate(zip(st.session_state.selected_models, model_tabs)):
                with tab:
                    # 显示模型说明
                    with st.expander("算法说明", expanded=True):
                        info = CLASSIFIER_INFO[model_key]
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
            st.info("请从左侧选择至少一个分类算法")

    # 训练选项
    st.markdown("---")
    st.subheader("高级训练选项")

    # 创建高级选项选项卡
    adv_tab1, adv_tab2, adv_tab3 = st.tabs([
        "🎯 基础训练", "🔧 超参数优化", "🤝 集成学习"
    ])

    with adv_tab1:
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            use_cv = st.checkbox("使用交叉验证评估性能", value=st.session_state.get('use_cv', False))
            st.session_state.use_cv = use_cv

            # 自动模型选择
            auto_select = st.checkbox("自动选择最佳模型", value=False, key="auto_select_models")

        with col_opt2:
            if use_cv:
                cv_folds = st.slider("交叉验证折数", 2, 10, st.session_state.get('cv_folds', 5))
                st.session_state.cv_folds = cv_folds

            if auto_select:
                n_auto_models = st.slider("自动选择模型数量", 3, 8, 5, key="n_auto_models")

    with adv_tab2:
        st.markdown("### 超参数优化设置")

        enable_hyperopt = st.checkbox("启用超参数优化", value=False, key="enable_hyperopt")

        if enable_hyperopt:
            col_hyp1, col_hyp2 = st.columns(2)
            with col_hyp1:
                search_type = st.selectbox(
                    "搜索策略",
                    options=['grid', 'random'],
                    format_func=lambda x: '网格搜索' if x == 'grid' else '随机搜索',
                    key="search_type_select"
                )

                hyperopt_cv_folds = st.slider("优化交叉验证折数", 2, 5, 3, key="hyperopt_cv_folds")

            with col_hyp2:
                if search_type == 'random':
                    n_iter = st.slider("随机搜索迭代次数", 10, 100, 20, key="n_iter_slider")

                # 选择要优化的模型
                optimizable_models = ['random_forest', 'xgboost', 'catboost', 'svm', 'logistic_regression']
                selected_for_opt = st.multiselect(
                    "选择要优化的模型",
                    options=optimizable_models,
                    format_func=lambda x: CLASSIFIER_INFO[x]['name'],
                    key="models_for_optimization"
                )

    with adv_tab3:
        st.markdown("### 集成学习设置")

        enable_ensemble = st.checkbox("启用集成学习", value=False, key="enable_ensemble")

        if enable_ensemble:
            col_ens1, col_ens2 = st.columns(2)
            with col_ens1:
                ensemble_method = st.selectbox(
                    "集成方法",
                    options=['voting', 'bagging'],
                    format_func=lambda x: '投票集成' if x == 'voting' else '装袋集成',
                    key="ensemble_method_select"
                )

            with col_ens2:
                if ensemble_method == 'voting':
                    voting_type = st.selectbox(
                        "投票类型",
                        options=['soft', 'hard'],
                        format_func=lambda x: '软投票(概率)' if x == 'soft' else '硬投票(类别)',
                        key="voting_type_select"
                    )

        # 保存高级训练配置
        st.session_state.advanced_training_config = {
            'auto_select': auto_select,
            'n_auto_models': n_auto_models if auto_select else 0,
            'enable_hyperopt': enable_hyperopt,
            'search_type': search_type if enable_hyperopt else 'grid',
            'hyperopt_cv_folds': hyperopt_cv_folds if enable_hyperopt else 3,
            'n_iter': n_iter if enable_hyperopt and search_type == 'random' else 20,
            'selected_for_opt': selected_for_opt if enable_hyperopt else [],
            'enable_ensemble': enable_ensemble,
            'ensemble_method': ensemble_method if enable_ensemble else 'voting',
            'voting_type': voting_type if enable_ensemble and ensemble_method == 'voting' else 'soft'
        }

    # 训练按钮
    st.markdown("---")

    # 根据配置显示不同的训练按钮
    advanced_config = st.session_state.get('advanced_training_config', {})

    if advanced_config.get('auto_select', False):
        if st.button("🤖 自动选择并训练最佳模型", type="primary", use_container_width=True):
            train_with_auto_selection()
    elif advanced_config.get('enable_hyperopt', False) and advanced_config.get('selected_for_opt'):
        if st.button("🔧 训练并优化选中模型", type="primary", use_container_width=True):
            train_with_hyperparameter_optimization()
    else:
        if st.button("🚀 开始训练所选模型", type="primary", use_container_width=True):
            if not st.session_state.selected_models:
                st.error("请至少选择一个分类算法")
                return
            train_selected_models()

    # 集成学习按钮
    if advanced_config.get('enable_ensemble', False) and st.session_state.get('training_results_dict'):
        if st.button("🤝 创建集成模型", use_container_width=True):
            create_and_evaluate_ensemble()

    # 显示训练进度和结果摘要
    if st.session_state.training_results_dict:
        st.markdown("---")
        st.subheader("训练结果摘要")
        display_results_summary()


def get_default_params(model_type):
    """获取模型的默认参数"""
    defaults = {
        'decision_tree': {'max_depth': 5, 'min_samples_split': 2, 'min_samples_leaf': 1, 'criterion': 'gini'},
        'random_forest': {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2},
        'adaboost': {'n_estimators': 50, 'learning_rate': 1.0, 'algorithm': 'SAMME.R'},
        'gradient_boosting': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'subsample': 1.0},
        'catboost': {'iterations': 100, 'learning_rate': 0.1, 'depth': 6},
        'xgboost': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6},
        'lightgbm': {'n_estimators': 100, 'learning_rate': 0.1, 'num_leaves': 31, 'max_depth': -1},
        'extra_trees': {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2},
        'knn': {'n_neighbors': 5, 'weights': 'uniform', 'algorithm': 'auto', 'p': 2},
        'svm': {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'},
        'neural_network': {'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam',
                           'alpha': 0.0001, 'learning_rate_init': 0.001, 'max_iter': 200},
        'naive_bayes': {'nb_type': 'gaussian', 'alpha': 1.0},
        'logistic_regression': {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs', 'max_iter': 100},
        'bp_neural_network': {
            'hidden_layers': '128,64,32',
            'activation': 'relu',
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'optimizer': 'adam',
            'early_stopping': True,
            'patience': 10
        },
        'rnn': {
            'rnn_units': 64,
            'num_layers': 2,
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'optimizer': 'adam',
            'sequence_length': 10,
            'early_stopping': True,
            'patience': 10
        },
        'cnn': {
            'conv_layers': '32,64,128',
            'kernel_size': 3,
            'pool_size': 2,
            'dense_units': 128,
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'optimizer': 'adam',
            'early_stopping': True,
            'patience': 10
        },
        'lstm': {
            'lstm_units': 64,
            'num_layers': 2,
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'optimizer': 'adam',
            'sequence_length': 10,
            'early_stopping': True,
            'patience': 10
        },
        'gru': {
            'gru_units': 64,
            'num_layers': 2,
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'optimizer': 'adam',
            'sequence_length': 10,
            'early_stopping': True,
            'patience': 10
        }
    }
    return defaults.get(model_type, {})


def create_param_widgets(model_type, key_prefix):
    """为不同模型创建参数输入控件"""
    params = {}

    if model_type == "decision_tree":
        col1, col2, col3 = st.columns(3)
        with col1:
            params['max_depth'] = st.slider("最大深度", 1, 20, 5, key=f"{key_prefix}_max_depth")
        with col2:
            params['min_samples_split'] = st.slider("最小分裂样本数", 2, 20, 2, key=f"{key_prefix}_min_split")
        with col3:
            params['criterion'] = st.selectbox("分裂标准", ["gini", "entropy"], key=f"{key_prefix}_criterion")

    elif model_type == "random_forest":
        col1, col2, col3 = st.columns(3)
        with col1:
            params['n_estimators'] = st.slider("树的数量", 10, 500, 100, 10, key=f"{key_prefix}_n_est")
        with col2:
            max_depth = st.slider("最大深度 (0=无限制)", 0, 30, 0, 1, key=f"{key_prefix}_max_depth")
            params['max_depth'] = None if max_depth == 0 else max_depth
        with col3:
            params['min_samples_split'] = st.slider("最小分裂样本数", 2, 20, 2, key=f"{key_prefix}_min_split")

    elif model_type == "adaboost":
        col1, col2, col3 = st.columns(3)
        with col1:
            params['n_estimators'] = st.slider("弱学习器数量", 10, 200, 50, 10, key=f"{key_prefix}_n_est")
        with col2:
            params['learning_rate'] = st.slider("学习率", 0.01, 2.0, 1.0, 0.01, key=f"{key_prefix}_lr")
        with col3:
            params['algorithm'] = st.selectbox("算法", ["SAMME", "SAMME.R"], index=1, key=f"{key_prefix}_algo")

    elif model_type == "gradient_boosting":
        col1, col2 = st.columns(2)
        with col1:
            params['n_estimators'] = st.slider("树的数量", 10, 500, 100, 10, key=f"{key_prefix}_n_est")
            params['learning_rate'] = st.slider("学习率", 0.01, 0.3, 0.1, 0.01, key=f"{key_prefix}_lr")
        with col2:
            params['max_depth'] = st.slider("最大深度", 1, 10, 3, key=f"{key_prefix}_max_depth")
            params['subsample'] = st.slider("子采样比例", 0.5, 1.0, 1.0, 0.1, key=f"{key_prefix}_subsample")

    elif model_type == "catboost":
        col1, col2, col3 = st.columns(3)
        with col1:
            params['iterations'] = st.slider("迭代次数", 50, 1000, 100, 50, key=f"{key_prefix}_iter")
        with col2:
            params['learning_rate'] = st.slider("学习率", 0.01, 0.3, 0.1, 0.01, key=f"{key_prefix}_lr")
        with col3:
            params['depth'] = st.slider("树深度", 1, 10, 6, key=f"{key_prefix}_depth")

    elif model_type == "xgboost":
        col1, col2, col3 = st.columns(3)
        with col1:
            params['n_estimators'] = st.slider("树的数量", 10, 500, 100, 10, key=f"{key_prefix}_n_est")
        with col2:
            params['learning_rate'] = st.slider("学习率", 0.01, 0.3, 0.1, 0.01, key=f"{key_prefix}_lr")
        with col3:
            params['max_depth'] = st.slider("最大深度", 1, 15, 6, key=f"{key_prefix}_max_depth")

    elif model_type == "lightgbm":
        col1, col2 = st.columns(2)
        with col1:
            params['n_estimators'] = st.slider("树的数量", 10, 500, 100, 10, key=f"{key_prefix}_n_est")
            params['learning_rate'] = st.slider("学习率", 0.01, 0.3, 0.1, 0.01, key=f"{key_prefix}_lr")
        with col2:
            params['num_leaves'] = st.slider("叶子节点数", 10, 300, 31, key=f"{key_prefix}_num_leaves")
            params['max_depth'] = st.slider("最大深度 (-1=无限制)", -1, 20, -1, key=f"{key_prefix}_max_depth")

    elif model_type == "extra_trees":
        col1, col2, col3 = st.columns(3)
        with col1:
            params['n_estimators'] = st.slider("树的数量", 10, 500, 100, 10, key=f"{key_prefix}_n_est")
        with col2:
            max_depth = st.slider("最大深度 (0=无限制)", 0, 30, 0, key=f"{key_prefix}_max_depth")
            params['max_depth'] = None if max_depth == 0 else max_depth
        with col3:
            params['min_samples_split'] = st.slider("最小分裂样本数", 2, 20, 2, key=f"{key_prefix}_min_split")

    elif model_type == "knn":
        col1, col2 = st.columns(2)
        with col1:
            params['n_neighbors'] = st.slider("邻居数量", 1, 30, 5, key=f"{key_prefix}_n_neighbors")
            params['weights'] = st.selectbox("权重", ["uniform", "distance"], key=f"{key_prefix}_weights")
        with col2:
            params['algorithm'] = st.selectbox("算法", ["auto", "ball_tree", "kd_tree", "brute"],
                                               key=f"{key_prefix}_algo")
            params['p'] = st.selectbox("距离度量", [1, 2], index=1, key=f"{key_prefix}_p",
                                       format_func=lambda x: "曼哈顿距离" if x == 1 else "欧氏距离")

    elif model_type == "svm":
        col1, col2, col3 = st.columns(3)
        with col1:
            params['C'] = st.slider("正则化参数 C", 0.1, 10.0, 1.0, 0.1, key=f"{key_prefix}_c")
        with col2:
            params['kernel'] = st.selectbox("核函数", ["rbf", "linear", "poly", "sigmoid"], key=f"{key_prefix}_kernel")
        with col3:
            if params['kernel'] in ['rbf', 'poly', 'sigmoid']:
                gamma_options = ["scale", "auto", 0.001, 0.01, 0.1, 1.0]
                params['gamma'] = st.selectbox("Gamma", gamma_options, key=f"{key_prefix}_gamma")

    elif model_type == "neural_network":
        col1, col2 = st.columns(2)
        with col1:
            hidden_str = st.text_input("隐藏层结构 (例: 100,50)", "100", key=f"{key_prefix}_hidden")
            try:
                params['hidden_layer_sizes'] = tuple(int(x.strip()) for x in hidden_str.split(',') if x.strip())
            except:
                params['hidden_layer_sizes'] = (100,)
            params['activation'] = st.selectbox("激活函数", ["relu", "tanh", "logistic"],
                                                key=f"{key_prefix}_activation")
        with col2:
            params['solver'] = st.selectbox("优化器", ["adam", "sgd", "lbfgs"], key=f"{key_prefix}_solver")
            params['alpha'] = st.slider("正则化强度", 0.0001, 1.0, 0.0001, 0.0001, format="%.4f",
                                        key=f"{key_prefix}_alpha")
            params['max_iter'] = st.slider("最大迭代次数", 100, 1000, 200, 100, key=f"{key_prefix}_max_iter")

    elif model_type == "naive_bayes":
        params['nb_type'] = st.selectbox("贝叶斯类型", ["gaussian", "multinomial"], key=f"{key_prefix}_nb_type")
        if params['nb_type'] == 'multinomial':
            params['alpha'] = st.slider("平滑参数", 0.0, 10.0, 1.0, 0.1, key=f"{key_prefix}_alpha")

    elif model_type == "logistic_regression":
        col1, col2 = st.columns(2)
        with col1:
            params['C'] = st.slider("正则化参数 C", 0.01, 10.0, 1.0, 0.01, key=f"{key_prefix}_c")
            params['penalty'] = st.selectbox("正则化类型", ["l2", "l1", "elasticnet", "none"],
                                             key=f"{key_prefix}_penalty")
        with col2:
            params['solver'] = st.selectbox("求解器", ["lbfgs", "liblinear", "newton-cg", "sag", "saga"],
                                            key=f"{key_prefix}_solver")
            params['max_iter'] = st.slider("最大迭代次数", 50, 1000, 100, 50, key=f"{key_prefix}_max_iter")

    elif model_type == "bp_neural_network":
        col1, col2 = st.columns(2)
        with col1:
            hidden_layers = st.text_input("隐藏层结构 (例: 128,64,32)", "128,64,32", key=f"{key_prefix}_hidden")
            params['hidden_layers'] = hidden_layers
            params['activation'] = st.selectbox("激活函数", ["relu", "tanh", "sigmoid"], key=f"{key_prefix}_activation")
            params['dropout_rate'] = st.slider("Dropout率", 0.0, 0.8, 0.2, 0.1, key=f"{key_prefix}_dropout")
            params['learning_rate'] = st.slider("学习率", 0.0001, 0.01, 0.001, 0.0001, format="%.4f",
                                                key=f"{key_prefix}_lr")
        with col2:
            params['batch_size'] = st.selectbox("批次大小", [16, 32, 64, 128], index=1, key=f"{key_prefix}_batch")
            params['epochs'] = st.slider("训练轮数", 50, 500, 100, 10, key=f"{key_prefix}_epochs")
            params['optimizer'] = st.selectbox("优化器", ["adam", "sgd", "rmsprop"], key=f"{key_prefix}_optimizer")
            params['early_stopping'] = st.checkbox("早停", True, key=f"{key_prefix}_early_stop")
            if params['early_stopping']:
                params['patience'] = st.slider("早停耐心值", 5, 50, 10, 5, key=f"{key_prefix}_patience")

    elif model_type == "rnn":
        col1, col2 = st.columns(2)
        with col1:
            params['rnn_units'] = st.slider("RNN单元数", 16, 256, 64, 16, key=f"{key_prefix}_units")
            params['num_layers'] = st.slider("层数", 1, 5, 2, key=f"{key_prefix}_layers")
            params['dropout_rate'] = st.slider("Dropout率", 0.0, 0.8, 0.2, 0.1, key=f"{key_prefix}_dropout")
            params['sequence_length'] = st.slider("序列长度", 5, 50, 10, 5, key=f"{key_prefix}_seq_len")
        with col2:
            params['learning_rate'] = st.slider("学习率", 0.0001, 0.01, 0.001, 0.0001, format="%.4f",
                                                key=f"{key_prefix}_lr")
            params['batch_size'] = st.selectbox("批次大小", [16, 32, 64, 128], index=1, key=f"{key_prefix}_batch")
            params['epochs'] = st.slider("训练轮数", 50, 500, 100, 10, key=f"{key_prefix}_epochs")
            params['optimizer'] = st.selectbox("优化器", ["adam", "sgd", "rmsprop"], key=f"{key_prefix}_optimizer")

    elif model_type == "cnn":
        col1, col2 = st.columns(2)
        with col1:
            conv_layers = st.text_input("卷积层通道 (例: 32,64,128)", "32,64,128", key=f"{key_prefix}_conv")
            params['conv_layers'] = conv_layers
            params['kernel_size'] = st.slider("卷积核大小", 2, 7, 3, key=f"{key_prefix}_kernel")
            params['pool_size'] = st.slider("池化大小", 2, 4, 2, key=f"{key_prefix}_pool")
            params['dense_units'] = st.slider("全连接层单元", 32, 512, 128, 32, key=f"{key_prefix}_dense")
        with col2:
            params['dropout_rate'] = st.slider("Dropout率", 0.0, 0.8, 0.2, 0.1, key=f"{key_prefix}_dropout")
            params['learning_rate'] = st.slider("学习率", 0.0001, 0.01, 0.001, 0.0001, format="%.4f",
                                                key=f"{key_prefix}_lr")
            params['batch_size'] = st.selectbox("批次大小", [16, 32, 64, 128], index=1, key=f"{key_prefix}_batch")
            params['epochs'] = st.slider("训练轮数", 50, 500, 100, 10, key=f"{key_prefix}_epochs")

    elif model_type == "lstm":
        col1, col2 = st.columns(2)
        with col1:
            params['lstm_units'] = st.slider("LSTM单元数", 16, 256, 64, 16, key=f"{key_prefix}_units")
            params['num_layers'] = st.slider("层数", 1, 5, 2, key=f"{key_prefix}_layers")
            params['dropout_rate'] = st.slider("Dropout率", 0.0, 0.8, 0.2, 0.1, key=f"{key_prefix}_dropout")
            params['sequence_length'] = st.slider("序列长度", 5, 50, 10, 5, key=f"{key_prefix}_seq_len")
        with col2:
            params['learning_rate'] = st.slider("学习率", 0.0001, 0.01, 0.001, 0.0001, format="%.4f",
                                                key=f"{key_prefix}_lr")
            params['batch_size'] = st.selectbox("批次大小", [16, 32, 64, 128], index=1, key=f"{key_prefix}_batch")
            params['epochs'] = st.slider("训练轮数", 50, 500, 100, 10, key=f"{key_prefix}_epochs")
            params['optimizer'] = st.selectbox("优化器", ["adam", "sgd", "rmsprop"], key=f"{key_prefix}_optimizer")

    elif model_type == "gru":
        col1, col2 = st.columns(2)
        with col1:
            params['gru_units'] = st.slider("GRU单元数", 16, 256, 64, 16, key=f"{key_prefix}_units")
            params['num_layers'] = st.slider("层数", 1, 5, 2, key=f"{key_prefix}_layers")
            params['dropout_rate'] = st.slider("Dropout率", 0.0, 0.8, 0.2, 0.1, key=f"{key_prefix}_dropout")
            params['sequence_length'] = st.slider("序列长度", 5, 50, 10, 5, key=f"{key_prefix}_seq_len")
        with col2:
            params['learning_rate'] = st.slider("学习率", 0.0001, 0.01, 0.001, 0.0001, format="%.4f",
                                                key=f"{key_prefix}_lr")
            params['batch_size'] = st.selectbox("批次大小", [16, 32, 64, 128], index=1, key=f"{key_prefix}_batch")
            params['epochs'] = st.slider("训练轮数", 50, 500, 100, 10, key=f"{key_prefix}_epochs")
            params['optimizer'] = st.selectbox("优化器", ["adam", "sgd", "rmsprop"], key=f"{key_prefix}_optimizer")

    params['random_state'] = 42
    return params


# 添加到代码中的辅助函数

def build_bp_neural_network(input_shape, num_classes, params):
    """构建BP神经网络"""
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
    if num_classes == 2:
        model.add(layers.Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    else:
        model.add(layers.Dense(num_classes, activation='softmax'))
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']

    # 编译模型
    optimizer_map = {
        'adam': Adam(learning_rate=params['learning_rate']),
        'sgd': SGD(learning_rate=params['learning_rate']),
        'rmsprop': RMSprop(learning_rate=params['learning_rate'])
    }

    model.compile(optimizer=optimizer_map[params['optimizer']], loss=loss, metrics=metrics)
    return model


def build_rnn_model(input_shape, num_classes, params):
    """构建RNN模型"""
    model = Sequential()

    # RNN层
    for i in range(params['num_layers']):
        return_sequences = i < params['num_layers'] - 1
        model.add(layers.SimpleRNN(
            params['rnn_units'],
            return_sequences=return_sequences,
            input_shape=input_shape if i == 0 else None,
            dropout=params['dropout_rate']
        ))

    # 输出层
    if num_classes == 2:
        model.add(layers.Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
    else:
        model.add(layers.Dense(num_classes, activation='softmax'))
        loss = 'sparse_categorical_crossentropy'

    # 编译模型
    optimizer_map = {
        'adam': Adam(learning_rate=params['learning_rate']),
        'sgd': SGD(learning_rate=params['learning_rate']),
        'rmsprop': RMSprop(learning_rate=params['learning_rate'])
    }

    model.compile(optimizer=optimizer_map[params['optimizer']], loss=loss, metrics=['accuracy'])
    return model


def build_cnn_model(input_shape, num_classes, params):
    """构建CNN模型"""
    model = Sequential()

    # 解析卷积层
    try:
        conv_filters = [int(x.strip()) for x in params['conv_layers'].split(',') if x.strip()]
    except:
        conv_filters = [32, 64, 128]

    # 卷积层
    for i, filters in enumerate(conv_filters):
        if i == 0:
            model.add(layers.Conv1D(filters, params['kernel_size'], activation='relu', input_shape=input_shape))
        else:
            model.add(layers.Conv1D(filters, params['kernel_size'], activation='relu'))
        model.add(layers.MaxPooling1D(params['pool_size']))
        if params['dropout_rate'] > 0:
            model.add(layers.Dropout(params['dropout_rate']))

    # 展平和全连接层
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(params['dense_units'], activation='relu'))
    if params['dropout_rate'] > 0:
        model.add(layers.Dropout(params['dropout_rate']))

    # 输出层
    if num_classes == 2:
        model.add(layers.Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
    else:
        model.add(layers.Dense(num_classes, activation='softmax'))
        loss = 'sparse_categorical_crossentropy'

    # 编译模型
    optimizer_map = {
        'adam': Adam(learning_rate=params['learning_rate']),
        'sgd': SGD(learning_rate=params['learning_rate']),
        'rmsprop': RMSprop(learning_rate=params['learning_rate'])
    }

    model.compile(optimizer=optimizer_map[params['optimizer']], loss=loss, metrics=['accuracy'])
    return model


def build_lstm_model(input_shape, num_classes, params):
    """构建LSTM模型"""
    model = Sequential()

    # LSTM层
    for i in range(params['num_layers']):
        return_sequences = i < params['num_layers'] - 1
        model.add(layers.LSTM(
            params['lstm_units'],
            return_sequences=return_sequences,
            input_shape=input_shape if i == 0 else None,
            dropout=params['dropout_rate']
        ))

    # 输出层
    if num_classes == 2:
        model.add(layers.Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
    else:
        model.add(layers.Dense(num_classes, activation='softmax'))
        loss = 'sparse_categorical_crossentropy'

    # 编译模型
    optimizer_map = {
        'adam': Adam(learning_rate=params['learning_rate']),
        'sgd': SGD(learning_rate=params['learning_rate']),
        'rmsprop': RMSprop(learning_rate=params['learning_rate'])
    }

    model.compile(optimizer=optimizer_map[params['optimizer']], loss=loss, metrics=['accuracy'])
    return model


def build_gru_model(input_shape, num_classes, params):
    """构建GRU模型"""
    model = Sequential()

    # GRU层
    for i in range(params['num_layers']):
        return_sequences = i < params['num_layers'] - 1
        model.add(layers.GRU(
            params['gru_units'],
            return_sequences=return_sequences,
            input_shape=input_shape if i == 0 else None,
            dropout=params['dropout_rate']
        ))

    # 输出层
    if num_classes == 2:
        model.add(layers.Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
    else:
        model.add(layers.Dense(num_classes, activation='softmax'))
        loss = 'sparse_categorical_crossentropy'

    # 编译模型
    optimizer_map = {
        'adam': Adam(learning_rate=params['learning_rate']),
        'sgd': SGD(learning_rate=params['learning_rate']),
        'rmsprop': RMSprop(learning_rate=params['learning_rate'])
    }

    model.compile(optimizer=optimizer_map[params['optimizer']], loss=loss, metrics=['accuracy'])
    return model


def prepare_sequence_data(X, sequence_length):
    """将表格数据转换为序列数据"""
    # 对于表格数据，我们可以使用滑动窗口方法
    # 或者简单地将特征重塑为序列格式

    if len(X) < sequence_length:
        # 如果数据量少于序列长度，进行填充
        sequence_length = len(X)

    sequences = []
    labels_seq = []

    # 使用滑动窗口方法
    for i in range(len(X) - sequence_length + 1):
        sequences.append(X.iloc[i:i + sequence_length].values)

    return np.array(sequences)


def prepare_cnn_data(X):
    """为CNN准备数据"""
    # 将特征数据重塑为适合1D卷积的格式
    # 每个样本被视为一个序列，特征作为序列元素
    return np.expand_dims(X.values, axis=2)  # (samples, features, 1)


def apply_advanced_preprocessing(X, y=None, config=None):
    """应用高级数据预处理"""
    if config is None:
        config = st.session_state.get('preprocessing_config', {})

    X_processed = X.copy()
    y_processed = y.copy() if y is not None else None
    preprocessing_info = {}

    # 1. 处理缺失值
    missing_method = config.get('missing_method', 'mean')
    if X_processed.isnull().any().any():
        st.info(f"检测到缺失值，使用{PREPROCESSING_OPTIONS['missing_value_handling']['methods'][missing_method]}处理")
        X_processed = handle_missing_values(X_processed, method=missing_method)
        preprocessing_info['missing_values_handled'] = True

    # 2. 异常值检测和处理
    if config.get('enable_outlier_detection', False):
        outlier_config = config.get('outlier_config', {})
        method = outlier_config.get('method', 'isolation_forest')
        action = outlier_config.get('action', '删除')

        st.info(f"使用{PREPROCESSING_OPTIONS['outlier_detection']['methods'][method]}检测异常值")

        outlier_params = {}
        if method in ['isolation_forest', 'elliptic_envelope', 'one_class_svm']:
            outlier_params['contamination'] = outlier_config.get('contamination', 0.1)
        elif method == 'z_score':
            outlier_params['threshold'] = outlier_config.get('threshold', 3.0)
        elif method == 'iqr':
            outlier_params['multiplier'] = outlier_config.get('multiplier', 1.5)

        outliers = detect_outliers(X_processed, method=method, **outlier_params)
        n_outliers = outliers.sum()

        if n_outliers > 0:
            st.warning(f"检测到 {n_outliers} 个异常值 ({n_outliers/len(X_processed)*100:.1f}%)")

            if action == '删除':
                X_processed = X_processed[~outliers]
                if y_processed is not None:
                    y_processed = y_processed[~outliers]
                st.info(f"已删除 {n_outliers} 个异常值")

            preprocessing_info['outliers_detected'] = n_outliers
            preprocessing_info['outliers_removed'] = n_outliers if action == '删除' else 0

    # 3. 特征选择
    if config.get('enable_feature_selection', False) and y_processed is not None:
        feature_config = config.get('feature_selection_config', {})
        method = feature_config.get('method', 'k_best')
        n_features = feature_config.get('n_features', 10)

        st.info(f"使用{PREPROCESSING_OPTIONS['feature_selection']['methods'][method]}进行特征选择")

        X_processed, feature_selector = apply_feature_selection(
            X_processed, y_processed, method=method, n_features=n_features
        )

        if feature_selector is not None:
            preprocessing_info['feature_selection_applied'] = True
            preprocessing_info['selected_features'] = list(X_processed.columns)
            preprocessing_info['feature_selector'] = feature_selector

    # 4. 特征缩放
    scaling_method = config.get('scaling_method', 'standard')
    X_processed, scaler = apply_feature_scaling(X_processed, method=scaling_method)
    preprocessing_info['scaling_method'] = scaling_method
    preprocessing_info['scaler'] = scaler

    # 5. 不平衡数据处理
    if config.get('enable_imbalance_handling', False) and y_processed is not None:
        imbalance_config = config.get('imbalance_config', {})
        method = imbalance_config.get('method', 'smote')
        sampling_strategy = imbalance_config.get('sampling_strategy', 'auto')

        # 检查数据是否不平衡
        class_counts = y_processed.value_counts()
        imbalance_ratio = class_counts.max() / class_counts.min()

        if imbalance_ratio > 1.5:  # 如果最大类别是最小类别的1.5倍以上
            st.info(f"检测到数据不平衡（比例: {imbalance_ratio:.2f}），使用{PREPROCESSING_OPTIONS['imbalance_handling']['methods'][method]}处理")

            X_processed, y_processed = handle_imbalanced_data(
                X_processed, y_processed, method=method, sampling_strategy=sampling_strategy
            )

            preprocessing_info['imbalance_handled'] = True
            preprocessing_info['original_class_distribution'] = class_counts.to_dict()
            preprocessing_info['new_class_distribution'] = y_processed.value_counts().to_dict()

    return X_processed, y_processed, preprocessing_info

# --- 自动模型选择和优化功能 ---
def get_hyperparameter_space(model_type):
    """获取模型的超参数搜索空间"""
    param_spaces = {
        'random_forest': {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'xgboost': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'subsample': [0.8, 0.9, 1.0]
        },
        'catboost': {
            'iterations': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'depth': [4, 6, 8, 10]
        },
        'svm': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'linear', 'poly'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
        },
        'logistic_regression': {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['liblinear', 'saga']
        }
    }
    return param_spaces.get(model_type, {})

def perform_hyperparameter_optimization(X_train, X_test, y_train, y_test, model_type, search_type='grid', cv_folds=3, n_iter=20):
    """执行超参数优化"""
    try:
        # 获取基础模型
        if model_type == 'random_forest':
            base_model = RandomForestClassifier(random_state=42)
        elif model_type == 'xgboost':
            base_model = XGBClassifier(random_state=42, eval_metric='mlogloss', use_label_encoder=False)
        elif model_type == 'catboost':
            base_model = CatBoostClassifier(verbose=0, random_state=42)
        elif model_type == 'svm':
            base_model = SVC(probability=True, random_state=42)
        elif model_type == 'logistic_regression':
            base_model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            return None, f"模型 {model_type} 不支持超参数优化"

        # 获取参数空间
        param_space = get_hyperparameter_space(model_type)
        if not param_space:
            return None, f"模型 {model_type} 没有定义参数空间"

        # 选择搜索策略
        if search_type == 'grid':
            search = GridSearchCV(
                base_model,
                param_space,
                cv=cv_folds,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
        else:  # random search
            search = RandomizedSearchCV(
                base_model,
                param_space,
                n_iter=n_iter,
                cv=cv_folds,
                scoring='accuracy',
                n_jobs=-1,
                random_state=42,
                verbose=1
            )

        # 执行搜索
        search.fit(X_train, y_train)

        # 获取最佳模型和参数
        best_model = search.best_estimator_
        best_params = search.best_params_
        best_score = search.best_score_

        # 在测试集上评估
        test_score = best_model.score(X_test, y_test)

        return {
            'best_model': best_model,
            'best_params': best_params,
            'best_cv_score': best_score,
            'test_score': test_score,
            'search_results': search.cv_results_
        }, None

    except Exception as e:
        return None, f"超参数优化出错: {str(e)}"

def create_ensemble_model(models_dict, method='voting'):
    """创建集成模型"""
    try:
        # 过滤出成功训练的模型
        valid_models = [(name, results['model']) for name, results in models_dict.items()
                       if results.get('model') is not None]

        if len(valid_models) < 2:
            return None, "需要至少2个有效模型才能创建集成"

        if method == 'voting':
            # 投票集成
            ensemble = VotingClassifier(
                estimators=valid_models,
                voting='soft'  # 使用概率投票
            )
        elif method == 'bagging':
            # 选择最佳模型作为基学习器
            best_model_name = max(models_dict.keys(),
                                key=lambda x: models_dict[x].get('test_accuracy', 0))
            base_estimator = models_dict[best_model_name]['model']

            ensemble = BaggingClassifier(
                base_estimator=base_estimator,
                n_estimators=10,
                random_state=42
            )
        else:
            return None, f"不支持的集成方法: {method}"

        return ensemble, None

    except Exception as e:
        return None, f"创建集成模型出错: {str(e)}"

def auto_select_best_models(X, y, n_models=5, cv_folds=3):
    """自动选择最佳模型"""
    # 定义候选模型
    candidate_models = {
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'xgboost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss', use_label_encoder=False),
        'catboost': CatBoostClassifier(iterations=100, verbose=0, random_state=42),
        'svm': SVC(probability=True, random_state=42),
        'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'extra_trees': ExtraTreesClassifier(n_estimators=100, random_state=42),
        'neural_network': MLPClassifier(hidden_layer_sizes=(100,), random_state=42, max_iter=500)
    }

    model_scores = {}

    # 使用交叉验证评估每个模型
    for name, model in candidate_models.items():
        try:
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy', n_jobs=-1)
            model_scores[name] = {
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'model': model
            }
        except Exception as e:
            st.warning(f"模型 {name} 评估失败: {str(e)}")
            continue

    # 按平均分数排序，选择前n个模型
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1]['mean_score'], reverse=True)
    best_models = dict(sorted_models[:n_models])

    return best_models

def train_with_auto_selection():
    """使用自动模型选择进行训练"""
    # 准备数据
    if st.session_state.data_source_type == "file":
        data_source = st.session_state.classification_data
        X = data_source[st.session_state.selected_input_columns].copy()
        y = data_source[st.session_state.selected_output_column].copy()
    else:
        data_dict = st.session_state.classification_data
        X = data_dict['X'][st.session_state.selected_input_columns].copy()
        y = data_dict['y'].copy()

    # 基础数据清理
    X = X.apply(pd.to_numeric, errors='coerce')
    if y.dtype == 'object':
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index, name=y.name)
        st.session_state.label_encoder = le

    # 应用预处理
    X_processed, y_processed, preprocessing_info = apply_advanced_preprocessing(X, y)

    st.subheader("自动模型选择")
    with st.spinner("正在评估候选模型..."):
        n_models = st.session_state.get('advanced_training_config', {}).get('n_auto_models', 5)
        best_models = auto_select_best_models(X_processed, y_processed, n_models=n_models)

        if not best_models:
            st.error("自动模型选择失败")
            return

        st.success(f"已自动选择 {len(best_models)} 个最佳模型")

        # 显示选择的模型
        model_df = pd.DataFrame([
            {
                '模型': CLASSIFIER_INFO.get(name, {}).get('name', name),
                '交叉验证准确率': f"{info['mean_score']:.4f} ± {info['std_score']:.4f}"
            }
            for name, info in best_models.items()
        ])
        st.dataframe(model_df, use_container_width=True, hide_index=True)

    # 更新选中的模型
    st.session_state.selected_models = list(best_models.keys())
    for model_name in best_models.keys():
        if model_name not in st.session_state.model_params:
            st.session_state.model_params[model_name] = get_default_params(model_name)

    # 训练选中的模型
    train_selected_models()

def train_with_hyperparameter_optimization():
    """使用超参数优化进行训练"""
    # 准备数据
    if st.session_state.data_source_type == "file":
        data_source = st.session_state.classification_data
        X = data_source[st.session_state.selected_input_columns].copy()
        y = data_source[st.session_state.selected_output_column].copy()
    else:
        data_dict = st.session_state.classification_data
        X = data_dict['X'][st.session_state.selected_input_columns].copy()
        y = data_dict['y'].copy()

    # 基础数据清理
    X = X.apply(pd.to_numeric, errors='coerce')
    if y.dtype == 'object':
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index, name=y.name)
        st.session_state.label_encoder = le

    # 应用预处理
    X_processed, y_processed, preprocessing_info = apply_advanced_preprocessing(X, y)

    # 分割数据
    test_size = st.session_state.get('preprocessing_config', {}).get('test_size', 0.2)
    random_seed = st.session_state.get('preprocessing_config', {}).get('random_seed', 42)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed,
        test_size=test_size,
        random_state=random_seed,
        stratify=y_processed
    )

    # 获取优化配置
    advanced_config = st.session_state.get('advanced_training_config', {})
    selected_for_opt = advanced_config.get('selected_for_opt', [])
    search_type = advanced_config.get('search_type', 'grid')
    cv_folds = advanced_config.get('hyperopt_cv_folds', 3)
    n_iter = advanced_config.get('n_iter', 20)

    st.subheader("超参数优化")

    # 为每个选中的模型进行优化
    optimization_results = {}
    progress_bar = st.progress(0)

    for i, model_type in enumerate(selected_for_opt):
        st.write(f"正在优化 {CLASSIFIER_INFO[model_type]['name']}...")

        result, error = perform_hyperparameter_optimization(
            X_train, X_test, y_train, y_test,
            model_type, search_type, cv_folds, n_iter
        )

        if result:
            optimization_results[model_type] = result
            st.success(f"{CLASSIFIER_INFO[model_type]['name']} 优化完成 - 最佳CV分数: {result['best_cv_score']:.4f}")
        else:
            st.error(f"{CLASSIFIER_INFO[model_type]['name']} 优化失败: {error}")

        progress_bar.progress((i + 1) / len(selected_for_opt))

    # 保存优化结果
    st.session_state.optimization_results = optimization_results

    # 显示优化结果摘要
    if optimization_results:
        st.subheader("优化结果摘要")
        opt_df = pd.DataFrame([
            {
                '模型': CLASSIFIER_INFO[model_type]['name'],
                '最佳CV分数': f"{result['best_cv_score']:.4f}",
                '测试分数': f"{result['test_score']:.4f}",
                '最佳参数': str(result['best_params'])
            }
            for model_type, result in optimization_results.items()
        ])
        st.dataframe(opt_df, use_container_width=True, hide_index=True)

def create_and_evaluate_ensemble():
    """创建和评估集成模型"""
    if not st.session_state.get('training_results_dict'):
        st.error("请先训练一些基础模型")
        return

    advanced_config = st.session_state.get('advanced_training_config', {})
    ensemble_method = advanced_config.get('ensemble_method', 'voting')

    st.subheader("集成模型创建")

    with st.spinner("正在创建集成模型..."):
        ensemble_model, error = create_ensemble_model(
            st.session_state.training_results_dict,
            method=ensemble_method
        )

        if error:
            st.error(f"集成模型创建失败: {error}")
            return

        # 准备测试数据（使用与基础模型相同的预处理）
        if st.session_state.data_source_type == "file":
            data_source = st.session_state.classification_data
            X = data_source[st.session_state.selected_input_columns].copy()
            y = data_source[st.session_state.selected_output_column].copy()
        else:
            data_dict = st.session_state.classification_data
            X = data_dict['X'][st.session_state.selected_input_columns].copy()
            y = data_dict['y'].copy()

        # 应用相同的预处理
        X_processed, y_processed, _ = apply_advanced_preprocessing(X, y)

        # 分割数据
        test_size = st.session_state.get('preprocessing_config', {}).get('test_size', 0.2)
        random_seed = st.session_state.get('preprocessing_config', {}).get('random_seed', 42)

        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed,
            test_size=test_size,
            random_state=random_seed,
            stratify=y_processed
        )

        # 训练集成模型
        ensemble_model.fit(X_train, y_train)

        # 评估集成模型
        train_score = ensemble_model.score(X_train, y_train)
        test_score = ensemble_model.score(X_test, y_test)

        # 保存集成模型结果
        ensemble_results = {
            'model': ensemble_model,
            'model_type': f'ensemble_{ensemble_method}',
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'ensemble_method': ensemble_method,
            'base_models': list(st.session_state.training_results_dict.keys())
        }

        st.session_state.training_results_dict[f'ensemble_{ensemble_method}'] = ensemble_results

        st.success(f"集成模型创建成功！")
        st.write(f"训练准确率: {train_score:.4f}")
        st.write(f"测试准确率: {test_score:.4f}")

        # 与基础模型比较
        st.subheader("与基础模型性能比较")
        comparison_data = []

        for model_name, results in st.session_state.training_results_dict.items():
            if results.get('model') is not None:
                comparison_data.append({
                    '模型': CLASSIFIER_INFO.get(model_name, {}).get('name', model_name),
                    '测试准确率': results.get('test_accuracy', 0),
                    '是否集成': '是' if 'ensemble' in model_name else '否'
                })

        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            comp_df = comp_df.sort_values('测试准确率', ascending=False)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

# --- 评估和验证增强功能 ---
def calculate_advanced_metrics(y_true, y_pred, y_proba=None, class_names=None):
    """计算高级评估指标"""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, log_loss, matthews_corrcoef, cohen_kappa_score,
        balanced_accuracy_score, classification_report
    )

    metrics = {}

    # 基础指标
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # 高级指标
    metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)

    # 如果有概率预测
    if y_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:  # 二分类
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                metrics['log_loss'] = log_loss(y_true, y_proba)
            else:  # 多分类
                metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
                metrics['roc_auc_ovo'] = roc_auc_score(y_true, y_proba, multi_class='ovo', average='weighted')
                metrics['log_loss'] = log_loss(y_true, y_proba)
        except Exception as e:
            st.warning(f"计算概率相关指标时出错: {str(e)}")

    # 分类报告
    try:
        metrics['classification_report'] = classification_report(
            y_true, y_pred,
            target_names=class_names if class_names else None,
            output_dict=True,
            zero_division=0
        )
    except Exception as e:
        st.warning(f"生成分类报告时出错: {str(e)}")

    return metrics

def analyze_model_interpretability(model, X, feature_names=None, model_type=None):
    """分析模型可解释性"""
    interpretability_results = {}

    try:
        # 特征重要性
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            if feature_names and len(feature_names) == len(importance):
                interpretability_results['feature_importance'] = dict(zip(feature_names, importance))
        elif hasattr(model, 'coef_') and model_type == 'logistic_regression':
            # 逻辑回归系数
            coef = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
            if feature_names and len(feature_names) == len(coef):
                interpretability_results['feature_importance'] = dict(zip(feature_names, coef))

        # 模型复杂度分析
        if hasattr(model, 'tree_'):  # 决策树类模型
            interpretability_results['model_complexity'] = {
                'n_nodes': model.tree_.node_count,
                'max_depth': model.tree_.max_depth,
                'n_leaves': model.tree_.n_leaves
            }
        elif hasattr(model, 'n_estimators'):  # 集成模型
            interpretability_results['model_complexity'] = {
                'n_estimators': model.n_estimators
            }
        elif hasattr(model, 'support_'):  # SVM
            interpretability_results['model_complexity'] = {
                'n_support_vectors': len(model.support_),
                'support_vector_ratio': len(model.support_) / len(X)
            }

    except Exception as e:
        st.warning(f"模型解释性分析出错: {str(e)}")

    return interpretability_results

def perform_learning_curve_analysis(model, X, y, cv=5):
    """执行学习曲线分析"""
    from sklearn.model_selection import learning_curve

    try:
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y,
            train_sizes=train_sizes,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42
        )

        return {
            'train_sizes': train_sizes_abs,
            'train_scores_mean': train_scores.mean(axis=1),
            'train_scores_std': train_scores.std(axis=1),
            'val_scores_mean': val_scores.mean(axis=1),
            'val_scores_std': val_scores.std(axis=1)
        }
    except Exception as e:
        st.warning(f"学习曲线分析出错: {str(e)}")
        return None

def perform_validation_curve_analysis(model, X, y, param_name, param_range, cv=5):
    """执行验证曲线分析"""
    from sklearn.model_selection import validation_curve

    try:
        train_scores, val_scores = validation_curve(
            model, X, y,
            param_name=param_name,
            param_range=param_range,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1
        )

        return {
            'param_range': param_range,
            'train_scores_mean': train_scores.mean(axis=1),
            'train_scores_std': train_scores.std(axis=1),
            'val_scores_mean': val_scores.mean(axis=1),
            'val_scores_std': val_scores.std(axis=1)
        }
    except Exception as e:
        st.warning(f"验证曲线分析出错: {str(e)}")
        return None

def detect_overfitting_underfitting(train_score, val_score, threshold=0.05):
    """检测过拟合和欠拟合"""
    score_gap = train_score - val_score

    if score_gap > threshold:
        if train_score > 0.9:
            return "过拟合", f"训练分数({train_score:.3f})明显高于验证分数({val_score:.3f})"
        else:
            return "轻微过拟合", f"存在轻微的过拟合倾向"
    elif train_score < 0.7 and val_score < 0.7:
        return "欠拟合", f"训练和验证分数都较低，模型可能过于简单"
    else:
        return "良好拟合", f"训练和验证分数接近且合理"

def analyze_class_imbalance_impact(y_true, y_pred, class_names=None):
    """分析类别不平衡对模型性能的影响"""
    from sklearn.metrics import precision_recall_fscore_support

    # 计算每个类别的分布
    unique_classes, class_counts = np.unique(y_true, return_counts=True)
    class_distribution = dict(zip(unique_classes, class_counts))

    # 计算每个类别的性能指标
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, zero_division=0)

    class_performance = {}
    for i, class_label in enumerate(unique_classes):
        class_name = class_names[i] if class_names and i < len(class_names) else f"Class_{class_label}"
        class_performance[class_name] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1_score': f1[i],
            'support': support[i],
            'frequency': class_counts[i] / len(y_true)
        }

    # 分析不平衡程度
    max_count = max(class_counts)
    min_count = min(class_counts)
    imbalance_ratio = max_count / min_count

    return {
        'class_distribution': class_distribution,
        'class_performance': class_performance,
        'imbalance_ratio': imbalance_ratio,
        'is_imbalanced': imbalance_ratio > 2.0
    }

def train_selected_models():
    """训练所有选中的模型 - 增强版"""
    # 准备数据
    if st.session_state.data_source_type == "file":
        data_source = st.session_state.classification_data
        X = data_source[st.session_state.selected_input_columns].copy()
        y = data_source[st.session_state.selected_output_column].copy()
    else:
        data_dict = st.session_state.classification_data
        X = data_dict['X'][st.session_state.selected_input_columns].copy()
        y = data_dict['y'].copy()

    # 基础数据清理
    X = X.apply(pd.to_numeric, errors='coerce')
    if X.isnull().values.any():
        st.warning("输入特征中检测到非数值数据，已尝试转换为数值。")

    if y.dtype == 'object':
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index, name=y.name)
        st.session_state.label_encoder = le

    # 应用高级预处理
    st.subheader("数据预处理")
    with st.expander("预处理详情", expanded=True):
        X_processed, y_processed, preprocessing_info = apply_advanced_preprocessing(X, y)

        # 显示预处理结果
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("原始样本数", len(X))
            st.metric("处理后样本数", len(X_processed))
        with col2:
            st.metric("原始特征数", X.shape[1])
            st.metric("处理后特征数", X_processed.shape[1])
        with col3:
            if 'outliers_detected' in preprocessing_info:
                st.metric("检测异常值", preprocessing_info['outliers_detected'])
            if 'imbalance_handled' in preprocessing_info:
                st.success("✅ 数据平衡处理完成")

    # 分割数据
    test_size = st.session_state.get('preprocessing_config', {}).get('test_size', 0.2)
    random_seed = st.session_state.get('preprocessing_config', {}).get('random_seed', 42)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed,
        test_size=test_size,
        random_state=random_seed,
        stratify=y_processed
    )

    # 保存预处理信息
    st.session_state.preprocessing_info = preprocessing_info

    # 创建进度条
    progress_bar = st.progress(0)
    status_text = st.empty()

    # 训练每个选中的模型
    num_models = len(st.session_state.selected_models)
    for i, model_type in enumerate(st.session_state.selected_models):
        status_text.text(f"正在训练 {CLASSIFIER_INFO[model_type]['name']}...")

        # 训练模型
        params = st.session_state.model_params[model_type]
        results = train_model(X_train, X_test, y_train, y_test, model_type, params)

        # 保存结果和预处理信息
        if results.get('model') is not None:
            results['preprocessing_info'] = preprocessing_info
        st.session_state.training_results_dict[model_type] = results

        # 交叉验证
        if st.session_state.use_cv and results.get('model') is not None:
            cv_results = perform_cross_validation(X_processed, y_processed, model_type, params, st.session_state.cv_folds)
            st.session_state.training_results_dict[model_type]['cv_results'] = cv_results

        # 更新进度
        progress_bar.progress((i + 1) / num_models)

    status_text.text("所有模型训练完成！")
    st.session_state.model_trained_flag = True


def display_results_summary():
    """显示所有模型的结果摘要"""
    # 创建结果比较表格
    results_data = []
    for model_type, results in st.session_state.training_results_dict.items():
        if results.get('model') is not None:
            row = {
                '模型': CLASSIFIER_INFO[model_type]['name'],
                '训练准确率': f"{results.get('train_accuracy', 0):.4f}",
                '测试准确率': f"{results.get('test_accuracy', 0):.4f}",
                '测试F1分数': f"{results.get('test_f1', 0):.4f}"
            }

            # 添加交叉验证结果
            if 'cv_results' in results and results['cv_results']:
                cv = results['cv_results']
                row['CV准确率'] = f"{cv.get('cv_accuracy_mean', 0):.4f} ± {cv.get('cv_accuracy_std', 0):.4f}"
            else:
                row['CV准确率'] = "N/A"

            results_data.append(row)

    if results_data:
        df = pd.DataFrame(results_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # 找出最佳模型
        best_model_idx = df['测试准确率'].apply(lambda x: float(x)).idxmax()
        best_model = df.iloc[best_model_idx]['模型']
        st.success(f"🏆 最佳模型：{best_model}")


def create_detailed_results_tab():
    """创建详细结果选项卡 - 增强版"""
    st.subheader("模型详细结果")

    # 选择要查看的模型
    available_models = [model_type for model_type, results in st.session_state.training_results_dict.items()
                        if results.get('model') is not None]

    if not available_models:
        st.warning("没有可用的训练结果")
        return

    selected_model = st.selectbox(
        "选择要查看详细结果的模型",
        available_models,
        format_func=lambda x: CLASSIFIER_INFO.get(x, {}).get('name', x)
    )

    if selected_model:
        results = st.session_state.training_results_dict[selected_model]

        # 创建详细分析选项卡
        detail_tab1, detail_tab2, detail_tab3, detail_tab4 = st.tabs([
            "📊 基础指标", "🔍 高级指标", "🧠 模型解释", "📈 性能分析"
        ])

        with detail_tab1:
            # 基础性能指标
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 训练集指标")
                metrics_train = {
                    "准确率": f"{results.get('train_accuracy', 0):.4f}",
                    "精确率": f"{results.get('train_precision', 0):.4f}",
                    "召回率": f"{results.get('train_recall', 0):.4f}",
                    "F1分数": f"{results.get('train_f1', 0):.4f}"
                }
                for metric, value in metrics_train.items():
                    st.metric(metric, value)

            with col2:
                st.markdown("### 测试集指标")
                metrics_test = {
                    "准确率": f"{results.get('test_accuracy', 0):.4f}",
                    "精确率": f"{results.get('test_precision', 0):.4f}",
                    "召回率": f"{results.get('test_recall', 0):.4f}",
                    "F1分数": f"{results.get('test_f1', 0):.4f}"
                }
                for metric, value in metrics_test.items():
                    st.metric(metric, value)

            # 过拟合/欠拟合检测
            train_acc = results.get('train_accuracy', 0)
            test_acc = results.get('test_accuracy', 0)
            fitting_status, fitting_desc = detect_overfitting_underfitting(train_acc, test_acc)

            if fitting_status == "过拟合":
                st.error(f"⚠️ {fitting_status}: {fitting_desc}")
            elif fitting_status == "欠拟合":
                st.warning(f"⚠️ {fitting_status}: {fitting_desc}")
            else:
                st.success(f"✅ {fitting_status}: {fitting_desc}")

        with detail_tab2:
            # 高级评估指标
            st.markdown("### 高级评估指标")

            y_test = results.get('y_test')
            y_test_pred = results.get('y_test_pred')
            y_test_proba = results.get('y_test_proba')
            class_names = results.get('class_names')

            if y_test is not None and y_test_pred is not None:
                advanced_metrics = calculate_advanced_metrics(
                    y_test, y_test_pred, y_test_proba, class_names
                )

                # 显示高级指标
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("平衡准确率", f"{advanced_metrics.get('balanced_accuracy', 0):.4f}")
                    st.metric("Matthews相关系数", f"{advanced_metrics.get('matthews_corrcoef', 0):.4f}")

                with col2:
                    st.metric("Cohen's Kappa", f"{advanced_metrics.get('cohen_kappa', 0):.4f}")
                    if 'roc_auc' in advanced_metrics:
                        st.metric("ROC AUC", f"{advanced_metrics['roc_auc']:.4f}")

                with col3:
                    if 'log_loss' in advanced_metrics:
                        st.metric("对数损失", f"{advanced_metrics['log_loss']:.4f}")
                    if 'roc_auc_ovr' in advanced_metrics:
                        st.metric("ROC AUC (OvR)", f"{advanced_metrics['roc_auc_ovr']:.4f}")

                # 分类报告
                if 'classification_report' in advanced_metrics:
                    st.markdown("### 详细分类报告")
                    report_df = pd.DataFrame(advanced_metrics['classification_report']).transpose()
                    st.dataframe(report_df.round(4), use_container_width=True)

                # 类别不平衡分析
                imbalance_analysis = analyze_class_imbalance_impact(y_test, y_test_pred, class_names)

                if imbalance_analysis['is_imbalanced']:
                    st.warning(f"⚠️ 检测到类别不平衡 (比例: {imbalance_analysis['imbalance_ratio']:.2f})")

                    # 显示每个类别的性能
                    st.markdown("### 各类别性能分析")
                    class_perf_data = []
                    for class_name, perf in imbalance_analysis['class_performance'].items():
                        class_perf_data.append({
                            '类别': class_name,
                            '精确率': f"{perf['precision']:.4f}",
                            '召回率': f"{perf['recall']:.4f}",
                            'F1分数': f"{perf['f1_score']:.4f}",
                            '样本数': perf['support'],
                            '频率': f"{perf['frequency']:.2%}"
                        })

                    class_perf_df = pd.DataFrame(class_perf_data)
                    st.dataframe(class_perf_df, use_container_width=True, hide_index=True)

        with detail_tab3:
            # 模型解释性分析
            st.markdown("### 模型解释性分析")

            model = results.get('model')
            if model is not None:
                # 获取特征名称
                if st.session_state.data_source_type == "file":
                    feature_names = st.session_state.selected_input_columns
                else:
                    feature_names = list(st.session_state.classification_data['X'].columns)

                # 如果进行了特征选择，更新特征名称
                preprocessing_info = results.get('preprocessing_info', {})
                if 'selected_features' in preprocessing_info:
                    feature_names = preprocessing_info['selected_features']

                interpretability = analyze_model_interpretability(
                    model, None, feature_names, selected_model
                )

                # 特征重要性
                if 'feature_importance' in interpretability:
                    st.markdown("#### 特征重要性")
                    importance_data = interpretability['feature_importance']

                    # 排序并显示
                    sorted_importance = sorted(importance_data.items(), key=lambda x: x[1], reverse=True)

                    importance_df = pd.DataFrame(sorted_importance, columns=['特征', '重要性'])
                    st.dataframe(importance_df.head(20), use_container_width=True, hide_index=True)

                    # 可视化特征重要性
                    if len(sorted_importance) > 0:
                        fig = plot_feature_importance(importance_data, top_n=15)
                        st.pyplot(fig)

                # 模型复杂度
                if 'model_complexity' in interpretability:
                    st.markdown("#### 模型复杂度")
                    complexity = interpretability['model_complexity']

                    complexity_cols = st.columns(len(complexity))
                    for i, (key, value) in enumerate(complexity.items()):
                        with complexity_cols[i]:
                            st.metric(key.replace('_', ' ').title(), value)

        with detail_tab4:
            # 性能分析
            st.markdown("### 性能分析")

            # 学习曲线分析（如果有足够数据）
            if st.button("生成学习曲线", key=f"learning_curve_{selected_model}"):
                with st.spinner("正在生成学习曲线..."):
                    # 重新获取数据进行学习曲线分析
                    if st.session_state.data_source_type == "file":
                        data_source = st.session_state.classification_data
                        X = data_source[st.session_state.selected_input_columns].copy()
                        y = data_source[st.session_state.selected_output_column].copy()
                    else:
                        data_dict = st.session_state.classification_data
                        X = data_dict['X'][st.session_state.selected_input_columns].copy()
                        y = data_dict['y'].copy()

                    # 应用相同的预处理
                    X_processed, y_processed, _ = apply_advanced_preprocessing(X, y)

                    # 获取模型
                    model = results['model']

                    # 生成学习曲线
                    learning_curve_data = perform_learning_curve_analysis(model, X_processed, y_processed)

                    if learning_curve_data:
                        # 绘制学习曲线
                        fig, ax = plt.subplots(figsize=(10, 6))

                        train_sizes = learning_curve_data['train_sizes']
                        train_mean = learning_curve_data['train_scores_mean']
                        train_std = learning_curve_data['train_scores_std']
                        val_mean = learning_curve_data['val_scores_mean']
                        val_std = learning_curve_data['val_scores_std']

                        ax.plot(train_sizes, train_mean, 'o-', color='blue', label='训练分数')
                        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')

                        ax.plot(train_sizes, val_mean, 'o-', color='red', label='验证分数')
                        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')

                        ax.set_xlabel('训练样本数')
                        ax.set_ylabel('准确率')
                        ax.set_title(f'{CLASSIFIER_INFO.get(selected_model, {}).get("name", selected_model)} 学习曲线')
                        ax.legend()
                        ax.grid(True, alpha=0.3)

                        st.pyplot(fig)
                        plt.close(fig)

# --- 状态管理辅助函数 ---
def stable_selectbox(label, options, format_func=None, state_key=None, key=None, help=None, on_change=None):
    """
    创建一个真正稳定的selectbox，完全避免界面闪烁

    Args:
        label: selectbox的标签
        options: 选项列表
        format_func: 格式化函数
        state_key: session state中的键名
        key: streamlit组件的key
        help: 帮助文本
        on_change: 值变化时的回调函数

    Returns:
        选中的值
    """
    if state_key is None:
        state_key = f"stable_selectbox_{key}" if key else f"stable_selectbox_{hash(label)}"

    # 初始化session state
    if state_key not in st.session_state and options:
        st.session_state[state_key] = options[0]

    # 确保当前值在选项中
    current_value = st.session_state.get(state_key)
    if current_value not in options and options:
        st.session_state[state_key] = options[0]
        current_value = options[0]

    # 使用callback来避免重新渲染
    def handle_change():
        if on_change:
            on_change()

    # 创建selectbox，使用on_change回调
    if options:
        index = options.index(current_value) if current_value in options else 0

        # 使用唯一的key避免重复渲染
        unique_key = f"{key}_{hash(str(options))}" if key else f"selectbox_{hash(str(options))}"

        selected = st.selectbox(
            label,
            options,
            index=index,
            format_func=format_func,
            key=unique_key,
            help=help,
            on_change=handle_change
        )

        # 只在真正变化时更新session state
        if selected != st.session_state.get(state_key):
            st.session_state[state_key] = selected

        return selected
    else:
        st.warning(f"没有可用的选项: {label}")
        return None

def stable_text_input(label, value="", state_key=None, key=None, help=None, **kwargs):
    """
    创建一个稳定的text_input，保持输入状态

    Args:
        label: 输入框标签
        value: 默认值
        state_key: session state中的键名
        key: streamlit组件的key
        help: 帮助文本
        **kwargs: 其他参数

    Returns:
        输入的值
    """
    if state_key is None:
        state_key = f"stable_text_input_{key}" if key else f"stable_text_input_{hash(label)}"

    # 初始化session state
    if state_key not in st.session_state:
        st.session_state[state_key] = value

    # 创建text_input
    input_value = st.text_input(
        label,
        value=st.session_state[state_key],
        key=key,
        help=help,
        **kwargs
    )

    # 更新session state
    st.session_state[state_key] = input_value

    return input_value

class NoFlickerComponentManager:
    """无闪烁组件管理器"""

    def __init__(self):
        self.component_states = {}
        self.render_cache = {}

    def create_stable_form(self, form_key, components_config):
        """
        创建一个完全稳定的表单，避免任何闪烁

        Args:
            form_key: 表单的唯一标识
            components_config: 组件配置列表

        Returns:
            表单数据字典
        """
        form_state_key = f"form_state_{form_key}"

        # 初始化表单状态
        if form_state_key not in st.session_state:
            st.session_state[form_state_key] = {}

        form_data = {}

        # 使用容器来避免重新渲染
        with st.container():
            for config in components_config:
                component_type = config['type']
                component_key = config['key']
                component_label = config['label']

                # 为每个组件创建稳定的状态
                state_key = f"{form_state_key}_{component_key}"

                if component_type == 'selectbox':
                    options = config['options']
                    format_func = config.get('format_func')

                    # 初始化状态
                    if state_key not in st.session_state and options:
                        st.session_state[state_key] = options[0]

                    current_value = st.session_state.get(state_key)
                    if current_value not in options and options:
                        st.session_state[state_key] = options[0]
                        current_value = options[0]

                    if options:
                        index = options.index(current_value) if current_value in options else 0

                        # 使用稳定的key
                        stable_key = f"{component_key}_stable_{hash(str(options))}"

                        selected = st.selectbox(
                            component_label,
                            options,
                            index=index,
                            format_func=format_func,
                            key=stable_key
                        )

                        # 只在真正变化时更新
                        if selected != st.session_state.get(state_key):
                            st.session_state[state_key] = selected
                            # 触发相关组件的更新回调
                            if 'on_change' in config:
                                config['on_change'](selected)

                        form_data[component_key] = selected

                elif component_type == 'text_input':
                    default_value = config.get('value', '')

                    # 初始化状态
                    if state_key not in st.session_state:
                        st.session_state[state_key] = default_value

                    input_value = st.text_input(
                        component_label,
                        value=st.session_state[state_key],
                        key=f"{component_key}_input"
                    )

                    st.session_state[state_key] = input_value
                    form_data[component_key] = input_value

                elif component_type == 'text_area':
                    default_value = config.get('value', '')

                    # 初始化状态
                    if state_key not in st.session_state:
                        st.session_state[state_key] = default_value

                    text_value = st.text_area(
                        component_label,
                        value=st.session_state[state_key],
                        key=f"{component_key}_area"
                    )

                    st.session_state[state_key] = text_value
                    form_data[component_key] = text_value

        return form_data

# --- 部署和生产化功能 ---
class ModelVersionManager:
    """模型版本管理器"""

    def __init__(self, base_path="models"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        self.metadata_file = os.path.join(base_path, "model_metadata.json")
        self.load_metadata()

    def load_metadata(self):
        """加载模型元数据"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            except:
                self.metadata = {}
        else:
            self.metadata = {}

    def save_metadata(self):
        """保存模型元数据"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def save_model(self, model_data, model_name, version=None, tags=None):
        """保存模型版本"""
        if version is None:
            version = self.get_next_version(model_name)

        model_id = f"{model_name}_v{version}"
        model_path = os.path.join(self.base_path, f"{model_id}.joblib")

        # 保存模型文件
        joblib.dump(model_data, model_path)

        # 更新元数据
        if model_name not in self.metadata:
            self.metadata[model_name] = {"versions": {}}

        self.metadata[model_name]["versions"][version] = {
            "model_id": model_id,
            "file_path": model_path,
            "created_at": datetime.now().isoformat(),
            "tags": tags or [],
            "metrics": model_data.get('metrics', {}),
            "model_type": model_data.get('model_type', 'unknown'),
            "status": "active"
        }

        self.save_metadata()
        return model_id

    def get_next_version(self, model_name):
        """获取下一个版本号"""
        if model_name not in self.metadata:
            return "1.0"

        versions = list(self.metadata[model_name]["versions"].keys())
        if not versions:
            return "1.0"

        # 简单的版本递增逻辑
        latest_version = max(versions, key=lambda x: float(x))
        major, minor = map(int, latest_version.split('.'))
        return f"{major}.{minor + 1}"

    def load_model(self, model_name, version=None):
        """加载指定版本的模型"""
        if model_name not in self.metadata:
            return None, f"模型 {model_name} 不存在"

        if version is None:
            # 加载最新版本
            versions = list(self.metadata[model_name]["versions"].keys())
            version = max(versions, key=lambda x: float(x))

        if version not in self.metadata[model_name]["versions"]:
            return None, f"版本 {version} 不存在"

        model_info = self.metadata[model_name]["versions"][version]
        model_path = model_info["file_path"]

        if not os.path.exists(model_path):
            return None, f"模型文件不存在: {model_path}"

        try:
            model_data = joblib.load(model_path)
            return model_data, None
        except Exception as e:
            return None, f"加载模型失败: {str(e)}"

    def list_models(self):
        """列出所有模型"""
        return list(self.metadata.keys())

    def list_versions(self, model_name):
        """列出模型的所有版本"""
        if model_name not in self.metadata:
            return []
        return list(self.metadata[model_name]["versions"].keys())

    def get_model_info(self, model_name, version=None):
        """获取模型信息"""
        if model_name not in self.metadata:
            return None

        if version is None:
            versions = list(self.metadata[model_name]["versions"].keys())
            version = max(versions, key=lambda x: float(x))

        return self.metadata[model_name]["versions"].get(version)

class ModelMonitor:
    """模型性能监控器"""

    def __init__(self, log_path="model_logs"):
        self.log_path = log_path
        os.makedirs(log_path, exist_ok=True)

    def log_prediction(self, model_id, input_data, prediction, confidence=None, timestamp=None):
        """记录预测日志"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        log_entry = {
            "timestamp": timestamp,
            "model_id": model_id,
            "input_data": input_data.tolist() if hasattr(input_data, 'tolist') else input_data,
            "prediction": prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
            "confidence": confidence.tolist() if hasattr(confidence, 'tolist') else confidence
        }

        log_file = os.path.join(self.log_path, f"{model_id}_predictions.jsonl")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

    def log_performance(self, model_id, metrics, timestamp=None):
        """记录性能指标"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        log_entry = {
            "timestamp": timestamp,
            "model_id": model_id,
            "metrics": metrics
        }

        log_file = os.path.join(self.log_path, f"{model_id}_performance.jsonl")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

    def get_performance_history(self, model_id, days=30):
        """获取性能历史"""
        log_file = os.path.join(self.log_path, f"{model_id}_performance.jsonl")
        if not os.path.exists(log_file):
            return []

        cutoff_date = datetime.now() - pd.Timedelta(days=days)
        performance_data = []

        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    entry_date = datetime.fromisoformat(entry['timestamp'])
                    if entry_date >= cutoff_date:
                        performance_data.append(entry)
                except:
                    continue

        return performance_data

def create_model_deployment_interface():
    """创建模型部署界面"""
    st.subheader("模型部署与管理")

    # 初始化版本管理器
    if 'version_manager' not in st.session_state:
        st.session_state.version_manager = ModelVersionManager()

    # 创建部署选项卡
    deploy_tab1, deploy_tab2, deploy_tab3, deploy_tab4 = st.tabs([
        "📦 模型保存", "🚀 模型部署", "📊 性能监控", "🔄 版本管理"
    ])

    with deploy_tab1:
        st.markdown("### 保存训练好的模型")

        if not st.session_state.get('training_results_dict'):
            st.info("请先训练一些模型")
            return

        # 选择要保存的模型
        available_models = [name for name, results in st.session_state.training_results_dict.items()
                           if results.get('model') is not None]

        if available_models:
            # 使用容器和缓存来避免闪烁
            model_save_container = st.container()

            with model_save_container:
                # 初始化持久状态
                if 'model_save_selected' not in st.session_state:
                    st.session_state.model_save_selected = available_models[0]
                if 'model_save_name_value' not in st.session_state:
                    st.session_state.model_save_name_value = available_models[0]
                if 'model_save_version_value' not in st.session_state:
                    st.session_state.model_save_version_value = ""
                if 'model_save_tags_value' not in st.session_state:
                    st.session_state.model_save_tags_value = "production,latest"
                if 'model_save_desc_value' not in st.session_state:
                    st.session_state.model_save_desc_value = ""

                # 确保选择的模型在可用列表中
                if st.session_state.model_save_selected not in available_models:
                    st.session_state.model_save_selected = available_models[0]
                    st.session_state.model_save_name_value = available_models[0]

                # 模型选择 - 使用固定的key和index
                current_index = available_models.index(st.session_state.model_save_selected)

                selected_for_save = st.selectbox(
                    "选择要保存的模型",
                    available_models,
                    index=current_index,
                    format_func=lambda x: CLASSIFIER_INFO.get(x, {}).get('name', x),
                    key="model_save_selectbox_fixed"
                )

                # 只在真正变化时更新状态和模型名称
                if selected_for_save != st.session_state.model_save_selected:
                    st.session_state.model_save_selected = selected_for_save
                    # 只在用户没有手动修改模型名称时才自动更新
                    if st.session_state.model_save_name_value in available_models:
                        st.session_state.model_save_name_value = selected_for_save

                col1, col2 = st.columns(2)
                with col1:
                    model_name = st.text_input(
                        "模型名称",
                        value=st.session_state.model_save_name_value,
                        key="model_name_input_fixed"
                    )
                    if model_name != st.session_state.model_save_name_value:
                        st.session_state.model_save_name_value = model_name

                    version = st.text_input(
                        "版本号 (留空自动生成)",
                        value=st.session_state.model_save_version_value,
                        key="model_version_input_fixed"
                    )
                    if version != st.session_state.model_save_version_value:
                        st.session_state.model_save_version_value = version

                with col2:
                    tags = st.text_input(
                        "标签 (用逗号分隔)",
                        value=st.session_state.model_save_tags_value,
                        key="model_tags_input_fixed"
                    )
                    if tags != st.session_state.model_save_tags_value:
                        st.session_state.model_save_tags_value = tags

                    description = st.text_area(
                        "模型描述",
                        value=st.session_state.model_save_desc_value,
                        key="model_description_input_fixed"
                    )
                    if description != st.session_state.model_save_desc_value:
                        st.session_state.model_save_desc_value = description

            if st.button("💾 保存模型", use_container_width=True, key="save_model_btn_fixed"):
                if st.session_state.model_save_name_value:
                    results = st.session_state.training_results_dict[st.session_state.model_save_selected]

                    # 准备保存数据
                    save_data = {
                        'model': results['model'],
                        'model_type': st.session_state.model_save_selected,
                        'feature_names': st.session_state.selected_input_columns,
                        'class_names': results.get('class_names'),
                        'scaler': st.session_state.get('scaler'),
                        'label_encoder': st.session_state.get('label_encoder'),
                        'preprocessing_info': results.get('preprocessing_info', {}),
                        'params': results.get('params', {}),
                        'metrics': {
                            'test_accuracy': results.get('test_accuracy'),
                            'test_precision': results.get('test_precision'),
                            'test_recall': results.get('test_recall'),
                            'test_f1': results.get('test_f1')
                        },
                        'description': st.session_state.model_save_desc_value,
                        'created_at': datetime.now().isoformat()
                    }

                    tag_list = [tag.strip() for tag in st.session_state.model_save_tags_value.split(',') if tag.strip()]
                    version_used = st.session_state.model_save_version_value if st.session_state.model_save_version_value else None

                    try:
                        model_id = st.session_state.version_manager.save_model(
                            save_data, st.session_state.model_save_name_value, version_used, tag_list
                        )
                        st.success(f"模型已保存: {model_id}")
                    except Exception as e:
                        st.error(f"保存模型失败: {str(e)}")
                else:
                    st.error("请输入模型名称")

    with deploy_tab2:
        st.markdown("### 模型部署")

        # 列出可用模型
        available_models = st.session_state.version_manager.list_models()

        if available_models:
            # 使用容器避免闪烁
            deploy_container = st.container()

            with deploy_container:
                # 初始化部署状态
                if 'deploy_selected_model' not in st.session_state:
                    st.session_state.deploy_selected_model = available_models[0]
                if 'deploy_selected_version' not in st.session_state:
                    st.session_state.deploy_selected_version = ""

                # 确保选择的模型在可用列表中
                if st.session_state.deploy_selected_model not in available_models:
                    st.session_state.deploy_selected_model = available_models[0]
                    st.session_state.deploy_selected_version = ""

                # 模型选择
                current_model_index = available_models.index(st.session_state.deploy_selected_model)
                selected_model = st.selectbox(
                    "选择要部署的模型",
                    available_models,
                    index=current_model_index,
                    key="deploy_model_selectbox_fixed"
                )

                # 检查模型是否变化
                if selected_model != st.session_state.deploy_selected_model:
                    st.session_state.deploy_selected_model = selected_model
                    st.session_state.deploy_selected_version = ""  # 重置版本选择

                if selected_model:
                    versions = st.session_state.version_manager.list_versions(selected_model)

                    if versions:
                        # 确保选择的版本在可用列表中
                        if not st.session_state.deploy_selected_version or st.session_state.deploy_selected_version not in versions:
                            st.session_state.deploy_selected_version = versions[0]

                        current_version_index = versions.index(st.session_state.deploy_selected_version)
                        selected_version = st.selectbox(
                            "选择版本",
                            versions,
                            index=current_version_index,
                            key="deploy_version_selectbox_fixed"
                        )

                        # 更新版本状态
                        if selected_version != st.session_state.deploy_selected_version:
                            st.session_state.deploy_selected_version = selected_version

                if selected_version:
                    model_info = st.session_state.version_manager.get_model_info(selected_model, selected_version)

                    # 显示模型信息
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**模型类型**: {model_info.get('model_type', 'Unknown')}")
                        st.write(f"**创建时间**: {model_info.get('created_at', 'Unknown')}")
                        st.write(f"**状态**: {model_info.get('status', 'Unknown')}")

                    with col2:
                        metrics = model_info.get('metrics', {})
                        if metrics:
                            st.write("**性能指标**:")
                            for metric, value in metrics.items():
                                if value is not None:
                                    st.write(f"- {metric}: {value:.4f}")

                    # 部署选项
                    st.markdown("#### 部署配置")
                    deployment_type = st.selectbox(
                        "部署类型",
                        ["本地API", "批量预测", "实时预测"]
                    )

                    if deployment_type == "本地API":
                        api_port = st.number_input("API端口", min_value=8000, max_value=9999, value=8080)

                        if st.button("🚀 启动API服务"):
                            st.info("API服务启动功能需要额外的Flask/FastAPI集成")
                            # 这里可以集成Flask或FastAPI来创建REST API

                    elif deployment_type == "批量预测":
                        st.markdown("##### 批量预测设置")
                        uploaded_file = st.file_uploader("上传预测数据", type=['csv', 'xlsx'])

                        if uploaded_file and st.button("开始批量预测"):
                            try:
                                # 加载模型
                                model_data, error = st.session_state.version_manager.load_model(
                                    selected_model, selected_version
                                )

                                if error:
                                    st.error(f"加载模型失败: {error}")
                                else:
                                    # 加载预测数据
                                    if uploaded_file.name.endswith('.csv'):
                                        pred_data = pd.read_csv(uploaded_file)
                                    else:
                                        pred_data = pd.read_excel(uploaded_file)

                                    # 执行预测
                                    model = model_data['model']
                                    feature_names = model_data['feature_names']
                                    scaler = model_data.get('scaler')

                                    # 准备数据
                                    X_pred = pred_data[feature_names]
                                    if scaler:
                                        X_pred_scaled = scaler.transform(X_pred)
                                    else:
                                        X_pred_scaled = X_pred

                                    # 预测
                                    predictions = model.predict(X_pred_scaled)
                                    if hasattr(model, 'predict_proba'):
                                        probabilities = model.predict_proba(X_pred_scaled)
                                        max_proba = probabilities.max(axis=1)
                                    else:
                                        max_proba = None

                                    # 准备结果
                                    result_df = pred_data.copy()
                                    result_df['预测结果'] = predictions
                                    if max_proba is not None:
                                        result_df['预测置信度'] = max_proba

                                    st.success("批量预测完成！")
                                    st.dataframe(result_df.head(10))

                                    # 提供下载
                                    csv = result_df.to_csv(index=False)
                                    st.download_button(
                                        "下载预测结果",
                                        csv,
                                        f"predictions_{selected_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        "text/csv"
                                    )

                            except Exception as e:
                                st.error(f"批量预测失败: {str(e)}")
        else:
            st.info("没有可用的已保存模型")

    with deploy_tab3:
        st.markdown("### 性能监控")

        # 初始化监控器
        if 'model_monitor' not in st.session_state:
            st.session_state.model_monitor = ModelMonitor()

        # 选择要监控的模型
        available_models = st.session_state.version_manager.list_models()

        if available_models:
            # 使用session state保持监控模型选择状态
            if 'selected_monitor_model' not in st.session_state:
                st.session_state.selected_monitor_model = available_models[0]

            selected_model = st.selectbox(
                "选择监控模型",
                available_models,
                index=available_models.index(st.session_state.selected_monitor_model) if st.session_state.selected_monitor_model in available_models else 0,
                key="monitor_model_select"
            )

            # 更新session state
            if selected_model != st.session_state.selected_monitor_model:
                st.session_state.selected_monitor_model = selected_model
                # 重置版本选择
                if 'selected_monitor_version' in st.session_state:
                    del st.session_state.selected_monitor_version

            if selected_model:
                versions = st.session_state.version_manager.list_versions(selected_model)

                if versions:
                    # 使用session state保持版本选择状态
                    if 'selected_monitor_version' not in st.session_state:
                        st.session_state.selected_monitor_version = versions[0]

                    selected_version = st.selectbox(
                        "选择版本",
                        versions,
                        index=versions.index(st.session_state.selected_monitor_version) if st.session_state.selected_monitor_version in versions else 0,
                        key="monitor_version_select"
                    )

                    # 更新session state
                    if selected_version != st.session_state.selected_monitor_version:
                        st.session_state.selected_monitor_version = selected_version

                if selected_version:
                    model_id = f"{selected_model}_v{selected_version}"

                    # 性能历史
                    st.markdown("#### 性能历史")
                    days = st.slider("查看天数", 1, 90, 30)

                    performance_history = st.session_state.model_monitor.get_performance_history(model_id, days)

                    if performance_history:
                        # 转换为DataFrame
                        perf_df = pd.DataFrame(performance_history)
                        perf_df['timestamp'] = pd.to_datetime(perf_df['timestamp'])

                        # 提取指标
                        metrics_data = []
                        for _, row in perf_df.iterrows():
                            metrics = row['metrics']
                            record = {'timestamp': row['timestamp']}
                            record.update(metrics)
                            metrics_data.append(record)

                        metrics_df = pd.DataFrame(metrics_data)

                        # 绘制性能趋势
                        if len(metrics_df) > 1:
                            fig, axes = plt.subplots(2, 2, figsize=(12, 8))

                            # 准确率趋势
                            if 'test_accuracy' in metrics_df.columns:
                                axes[0, 0].plot(metrics_df['timestamp'], metrics_df['test_accuracy'], 'b-o')
                                axes[0, 0].set_title('准确率趋势')
                                axes[0, 0].set_ylabel('准确率')
                                axes[0, 0].tick_params(axis='x', rotation=45)

                            # F1分数趋势
                            if 'test_f1' in metrics_df.columns:
                                axes[0, 1].plot(metrics_df['timestamp'], metrics_df['test_f1'], 'g-o')
                                axes[0, 1].set_title('F1分数趋势')
                                axes[0, 1].set_ylabel('F1分数')
                                axes[0, 1].tick_params(axis='x', rotation=45)

                            # 精确率趋势
                            if 'test_precision' in metrics_df.columns:
                                axes[1, 0].plot(metrics_df['timestamp'], metrics_df['test_precision'], 'r-o')
                                axes[1, 0].set_title('精确率趋势')
                                axes[1, 0].set_ylabel('精确率')
                                axes[1, 0].tick_params(axis='x', rotation=45)

                            # 召回率趋势
                            if 'test_recall' in metrics_df.columns:
                                axes[1, 1].plot(metrics_df['timestamp'], metrics_df['test_recall'], 'm-o')
                                axes[1, 1].set_title('召回率趋势')
                                axes[1, 1].set_ylabel('召回率')
                                axes[1, 1].tick_params(axis='x', rotation=45)

                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)

                        # 显示最新指标
                        st.markdown("#### 最新性能指标")
                        latest_metrics = performance_history[-1]['metrics']

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("准确率", f"{latest_metrics.get('test_accuracy', 0):.4f}")
                        with col2:
                            st.metric("精确率", f"{latest_metrics.get('test_precision', 0):.4f}")
                        with col3:
                            st.metric("召回率", f"{latest_metrics.get('test_recall', 0):.4f}")
                        with col4:
                            st.metric("F1分数", f"{latest_metrics.get('test_f1', 0):.4f}")
                    else:
                        st.info("暂无性能监控数据")

                    # 手动记录性能
                    st.markdown("#### 手动记录性能")
                    if st.button("记录当前模型性能"):
                        # 获取当前训练结果
                        if selected_model in st.session_state.training_results_dict:
                            results = st.session_state.training_results_dict[selected_model]
                            metrics = {
                                'test_accuracy': results.get('test_accuracy'),
                                'test_precision': results.get('test_precision'),
                                'test_recall': results.get('test_recall'),
                                'test_f1': results.get('test_f1')
                            }

                            st.session_state.model_monitor.log_performance(model_id, metrics)
                            st.success("性能指标已记录")
        else:
            st.info("没有可监控的模型")

    with deploy_tab4:
        st.markdown("### 版本管理")

        # 模型列表
        available_models = st.session_state.version_manager.list_models()

        if available_models:
            st.markdown("#### 已保存的模型")

            for model_name in available_models:
                with st.expander(f"📦 {model_name}"):
                    versions = st.session_state.version_manager.list_versions(model_name)

                    version_data = []
                    for version in versions:
                        model_info = st.session_state.version_manager.get_model_info(model_name, version)
                        version_data.append({
                            '版本': version,
                            '模型类型': model_info.get('model_type', 'Unknown'),
                            '创建时间': model_info.get('created_at', 'Unknown')[:19],
                            '状态': model_info.get('status', 'Unknown'),
                            '标签': ', '.join(model_info.get('tags', []))
                        })

                    if version_data:
                        version_df = pd.DataFrame(version_data)
                        st.dataframe(version_df, use_container_width=True, hide_index=True)

                        # 版本操作
                        col1, col2 = st.columns(2)
                        with col1:
                            # 使用session state保持版本选择状态
                            version_state_key = f"selected_version_{model_name}"
                            if version_state_key not in st.session_state:
                                st.session_state[version_state_key] = versions[0] if versions else None

                            if versions:
                                selected_version = st.selectbox(
                                    f"选择{model_name}的版本",
                                    versions,
                                    index=versions.index(st.session_state[version_state_key]) if st.session_state[version_state_key] in versions else 0,
                                    key=f"version_select_{model_name}"
                                )

                                # 更新session state
                                if selected_version != st.session_state[version_state_key]:
                                    st.session_state[version_state_key] = selected_version
                            else:
                                selected_version = None

                        with col2:
                            if selected_version and st.button(f"加载版本 {selected_version}", key=f"load_{model_name}_{selected_version}"):
                                model_data, error = st.session_state.version_manager.load_model(model_name, selected_version)
                                if error:
                                    st.error(f"加载失败: {error}")
                                else:
                                    st.success(f"已加载 {model_name} v{selected_version}")
                                    # 可以在这里将模型加载到当前会话中
        else:
            st.info("没有已保存的模型")

        # 清理功能
        st.markdown("#### 维护操作")
        if st.button("🗑️ 清理旧版本", help="删除超过30天的旧版本"):
            st.warning("清理功能需要谨慎实现，建议手动管理")

def create_advanced_settings_tab():
    """创建高级设置选项卡"""
    st.subheader("高级设置与配置")

    # 创建设置选项卡
    settings_tab1, settings_tab2, settings_tab3 = st.tabs([
        "⚙️ 系统设置", "🔧 算法配置", "📊 数据设置"
    ])

    with settings_tab1:
        st.markdown("### 系统配置")

        col1, col2 = st.columns(2)
        with col1:
            # 并行处理设置
            n_jobs = st.slider("并行处理核心数", -1, 8, -1, help="-1表示使用所有可用核心")

            # 内存设置
            memory_limit = st.selectbox("内存限制", ["无限制", "1GB", "2GB", "4GB", "8GB"])

            # 缓存设置
            enable_cache = st.checkbox("启用结果缓存", value=True)

        with col2:
            # 日志级别
            log_level = st.selectbox("日志级别", ["DEBUG", "INFO", "WARNING", "ERROR"])

            # 自动保存
            auto_save_interval = st.slider("自动保存间隔(分钟)", 1, 60, 5)

            # 图表设置
            plot_style = st.selectbox("图表样式", ["default", "seaborn", "ggplot", "bmh"])

        # 保存系统设置
        if st.button("💾 保存系统设置"):
            system_config = {
                'n_jobs': n_jobs,
                'memory_limit': memory_limit,
                'enable_cache': enable_cache,
                'log_level': log_level,
                'auto_save_interval': auto_save_interval,
                'plot_style': plot_style
            }
            st.session_state.system_config = system_config
            st.success("系统设置已保存")

    with settings_tab2:
        st.markdown("### 算法默认配置")

        # 选择要配置的算法
        algorithm_to_config = st.selectbox(
            "选择算法",
            list(CLASSIFIER_INFO.keys()),
            format_func=lambda x: CLASSIFIER_INFO[x]['name']
        )

        if algorithm_to_config:
            st.markdown(f"#### {CLASSIFIER_INFO[algorithm_to_config]['name']} 默认参数")

            # 显示当前默认参数
            current_defaults = get_default_params(algorithm_to_config)

            # 创建参数编辑界面
            new_params = {}
            for param, value in current_defaults.items():
                if isinstance(value, bool):
                    new_params[param] = st.checkbox(param, value=value, key=f"config_{param}")
                elif isinstance(value, int):
                    new_params[param] = st.number_input(param, value=value, key=f"config_{param}")
                elif isinstance(value, float):
                    new_params[param] = st.number_input(param, value=value, format="%.4f", key=f"config_{param}")
                elif isinstance(value, str):
                    new_params[param] = st.text_input(param, value=value, key=f"config_{param}")
                else:
                    st.text(f"{param}: {value} (不可编辑)")

            if st.button(f"保存 {CLASSIFIER_INFO[algorithm_to_config]['name']} 配置"):
                # 保存新的默认参数
                if 'algorithm_configs' not in st.session_state:
                    st.session_state.algorithm_configs = {}
                st.session_state.algorithm_configs[algorithm_to_config] = new_params
                st.success("算法配置已保存")

    with settings_tab3:
        st.markdown("### 数据处理设置")

        col1, col2 = st.columns(2)
        with col1:
            # 数据验证设置
            st.markdown("#### 数据验证")
            validate_data_types = st.checkbox("验证数据类型", value=True)
            check_missing_values = st.checkbox("检查缺失值", value=True)
            detect_outliers_auto = st.checkbox("自动检测异常值", value=False)

            # 预处理默认设置
            st.markdown("#### 预处理默认值")
            default_test_size = st.slider("默认测试集比例", 0.1, 0.5, 0.2, 0.05)
            default_cv_folds = st.slider("默认交叉验证折数", 2, 10, 5)

        with col2:
            # 特征工程设置
            st.markdown("#### 特征工程")
            auto_feature_selection = st.checkbox("自动特征选择", value=False)
            auto_scaling = st.checkbox("自动特征缩放", value=True)

            # 数据增强设置
            st.markdown("#### 数据增强")
            enable_data_augmentation = st.checkbox("启用数据增强", value=False)
            augmentation_factor = st.slider("增强倍数", 1.0, 5.0, 2.0, 0.5)

        # 保存数据设置
        if st.button("💾 保存数据设置"):
            data_config = {
                'validate_data_types': validate_data_types,
                'check_missing_values': check_missing_values,
                'detect_outliers_auto': detect_outliers_auto,
                'default_test_size': default_test_size,
                'default_cv_folds': default_cv_folds,
                'auto_feature_selection': auto_feature_selection,
                'auto_scaling': auto_scaling,
                'enable_data_augmentation': enable_data_augmentation,
                'augmentation_factor': augmentation_factor
            }
            st.session_state.data_config = data_config
            st.success("数据设置已保存")


def create_feature_analysis_tab():
    """创建特征分析选项卡"""
    st.subheader("特征重要性分析")

    # 收集有特征重要性的模型
    models_with_importance = []
    for model_type, results in st.session_state.training_results_dict.items():
        if results.get('model') is not None and results.get('feature_importance'):
            models_with_importance.append(model_type)

    if not models_with_importance:
        st.info("当前训练的模型中没有支持特征重要性分析的模型")
        st.warning("决策树、随机森林、梯度提升等模型支持特征重要性分析")
        return

    # 选择模型
    selected_model = st.selectbox(
        "选择模型查看特征重要性",
        models_with_importance,
        format_func=lambda x: CLASSIFIER_INFO[x]['name']
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
                        if results.get('model') is not None]

    if not available_models:
        st.warning("没有可用的模型结果")
        return

    selected_model = st.selectbox(
        "选择要可视化的模型",
        available_models,
        format_func=lambda x: CLASSIFIER_INFO[x]['name'],
        key="viz_model_select"
    )

    if selected_model:
        results = st.session_state.training_results_dict[selected_model]

        # 获取数据
        y_test = results.get('y_test')
        y_test_pred = results.get('y_test_pred')
        y_test_proba = results.get('y_test_proba')
        class_names = results.get('class_names')

        # 选择可视化类型
        viz_options = ["混淆矩阵", "类别分布对比"]
        if y_test_proba is not None:
            viz_options.append("ROC曲线")

        viz_type = st.selectbox("选择可视化类型", viz_options, key="viz_type_select")

        # 生成可视化
        try:
            if viz_type == "混淆矩阵":
                fig = plot_confusion_matrix(y_test, y_test_pred, class_names)
            elif viz_type == "ROC曲线":
                fig = plot_roc_curve(y_test, y_test_proba, class_names)
            elif viz_type == "类别分布对比":
                fig = plot_class_distribution(y_test, y_test_pred, class_names)

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
                        if results.get('model') is not None]

    if not available_models:
        st.warning("没有可导出的模型")
        return

    selected_model = st.selectbox(
        "选择要导出的模型",
        available_models,
        format_func=lambda x: CLASSIFIER_INFO[x]['name'],
        key="export_model_select"
    )

    if selected_model:
        results = st.session_state.training_results_dict[selected_model]

        # 显示模型信息
        st.info(f"""
        **模型类型**: {CLASSIFIER_INFO[selected_model]['name']}  
        **测试准确率**: {results.get('test_accuracy', 0):.4f}  
        **测试F1分数**: {results.get('test_f1', 0):.4f}
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
                        'class_names': results.get('class_names'),
                        'scaler': st.session_state.get('scaler'),
                        'label_encoder': st.session_state.get('label_encoder'),
                        'params': results.get('params'),
                        'metrics': {
                            'test_accuracy': results.get('test_accuracy'),
                            'test_precision': results.get('test_precision'),
                            'test_recall': results.get('test_recall'),
                            'test_f1': results.get('test_f1')
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
            if st.button("导出测试集预测结果 (.csv)", key="export_predictions"):
                try:
                    # 创建预测结果DataFrame
                    pred_df = pd.DataFrame({
                        '真实值': results['y_test'],
                        '预测值': results['y_test_pred']
                    })

                    # 添加概率（如果有）
                    if results.get('y_test_proba') is not None:
                        proba_cols = [f'概率_类别{i}' for i in range(results['y_test_proba'].shape[1])]
                        proba_df = pd.DataFrame(results['y_test_proba'], columns=proba_cols, index=pred_df.index)
                        pred_df = pd.concat([pred_df, proba_df], axis=1)

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

        # 模型部署代码示例
        st.markdown("### 模型部署代码示例")
        with st.expander("查看如何使用导出的模型"):
            st.code(f"""
import joblib
import pandas as pd
import numpy as np

# 加载模型
model_data = joblib.load('your_model_file.joblib')
model = model_data['model']
scaler = model_data.get('scaler')
feature_names = model_data['feature_names']

# 准备新数据（确保特征顺序与训练时一致）
new_data = pd.DataFrame({{
    # 填入您的特征数据
    'feature1': [value1],
    'feature2': [value2],
    # ...
}})

# 确保特征顺序正确
new_data = new_data[feature_names]

# 标准化（如果训练时使用了）
if scaler is not None:
    new_data_scaled = scaler.transform(new_data)
else:
    new_data_scaled = new_data

# 预测
predictions = model.predict(new_data_scaled)
probabilities = model.predict_proba(new_data_scaled)  # 如果需要概率

print(f"预测结果: {{predictions}}")
print(f"预测概率: {{probabilities}}")
            """, language='python')

def create_results_section():
    """创建结果展示部分UI - 增强版"""
    st.header("结果展示与分析")

    if not st.session_state.model_trained_flag or not st.session_state.training_results_dict:
        st.info("请先在模型训练选项卡中训练模型以查看结果。")
        return

    # 创建子选项卡
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 模型比较", "📈 详细结果", "🔍 特征分析", "📉 可视化",
        "💾 导出", "🚀 部署", "⚙️ 高级设置"
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

    with tab6:
        create_model_deployment_interface()

    with tab7:
        create_advanced_settings_tab()

# --- 智能推荐系统 ---
def analyze_data_characteristics(X, y):
    """分析数据特征，为模型推荐提供依据"""
    characteristics = {}

    # 数据规模
    n_samples, n_features = X.shape
    characteristics['n_samples'] = n_samples
    characteristics['n_features'] = n_features
    characteristics['data_size'] = 'small' if n_samples < 1000 else 'medium' if n_samples < 10000 else 'large'
    characteristics['feature_size'] = 'low' if n_features < 10 else 'medium' if n_features < 100 else 'high'

    # 类别分布
    class_counts = pd.Series(y).value_counts()
    characteristics['n_classes'] = len(class_counts)
    characteristics['class_balance'] = class_counts.max() / class_counts.min()
    characteristics['is_binary'] = len(class_counts) == 2
    characteristics['is_imbalanced'] = characteristics['class_balance'] > 2.0

    # 特征特性
    numeric_features = X.select_dtypes(include=[np.number]).columns
    characteristics['n_numeric_features'] = len(numeric_features)
    characteristics['feature_types'] = 'numeric' if len(numeric_features) == n_features else 'mixed'

    # 数据质量
    missing_ratio = X.isnull().sum().sum() / (n_samples * n_features)
    characteristics['missing_ratio'] = missing_ratio
    characteristics['has_missing'] = missing_ratio > 0

    # 特征相关性
    if len(numeric_features) > 1:
        corr_matrix = X[numeric_features].corr().abs()
        # 排除对角线
        corr_matrix = corr_matrix.where(~np.eye(corr_matrix.shape[0], dtype=bool))
        max_corr = corr_matrix.max().max()
        characteristics['max_correlation'] = max_corr
        characteristics['high_correlation'] = max_corr > 0.8

    return characteristics

def recommend_models(data_characteristics):
    """基于数据特征推荐模型"""
    recommendations = []

    n_samples = data_characteristics['n_samples']
    n_features = data_characteristics['n_features']
    data_size = data_characteristics['data_size']
    is_binary = data_characteristics['is_binary']
    is_imbalanced = data_characteristics['is_imbalanced']
    has_missing = data_characteristics['has_missing']
    high_correlation = data_characteristics.get('high_correlation', False)

    # 基于数据规模的推荐
    if data_size == 'small':
        recommendations.extend([
            {'model': 'logistic_regression', 'priority': 'high', 'reason': '小数据集适合简单模型'},
            {'model': 'naive_bayes', 'priority': 'high', 'reason': '小数据集上表现良好'},
            {'model': 'knn', 'priority': 'medium', 'reason': '适合小数据集的非参数方法'},
            {'model': 'decision_tree', 'priority': 'medium', 'reason': '可解释性强，适合小数据'}
        ])
    elif data_size == 'medium':
        recommendations.extend([
            {'model': 'random_forest', 'priority': 'high', 'reason': '中等数据集的最佳选择'},
            {'model': 'xgboost', 'priority': 'high', 'reason': '高性能，适合中等规模数据'},
            {'model': 'svm', 'priority': 'medium', 'reason': '在中等数据上表现良好'},
            {'model': 'neural_network', 'priority': 'medium', 'reason': '可以学习复杂模式'}
        ])
    else:  # large
        recommendations.extend([
            {'model': 'xgboost', 'priority': 'high', 'reason': '大数据集上性能优异'},
            {'model': 'lightgbm', 'priority': 'high', 'reason': '大数据集训练速度快'},
            {'model': 'catboost', 'priority': 'high', 'reason': '自动处理类别特征'},
            {'model': 'neural_network', 'priority': 'medium', 'reason': '大数据集可发挥深度学习优势'}
        ])

    # 基于类别数量的推荐
    if is_binary:
        recommendations.append({
            'model': 'logistic_regression', 'priority': 'high',
            'reason': '二分类问题的经典选择'
        })

    # 基于数据不平衡的推荐
    if is_imbalanced:
        recommendations.extend([
            {'model': 'random_forest', 'priority': 'high', 'reason': '对不平衡数据鲁棒'},
            {'model': 'xgboost', 'priority': 'high', 'reason': '可通过参数处理不平衡'},
            {'preprocessing': 'imbalance_handling', 'priority': 'high', 'reason': '建议使用SMOTE等方法'}
        ])

    # 基于缺失值的推荐
    if has_missing:
        recommendations.extend([
            {'model': 'xgboost', 'priority': 'high', 'reason': '内置缺失值处理'},
            {'model': 'catboost', 'priority': 'high', 'reason': '自动处理缺失值'},
            {'preprocessing': 'missing_value_handling', 'priority': 'high', 'reason': '建议预处理缺失值'}
        ])

    # 基于特征相关性的推荐
    if high_correlation:
        recommendations.extend([
            {'preprocessing': 'feature_selection', 'priority': 'medium', 'reason': '高相关性特征建议降维'},
            {'model': 'random_forest', 'priority': 'medium', 'reason': '对特征相关性不敏感'}
        ])

    # 去重并排序
    unique_recommendations = {}
    for rec in recommendations:
        key = rec.get('model') or rec.get('preprocessing')
        if key not in unique_recommendations or rec['priority'] == 'high':
            unique_recommendations[key] = rec

    return list(unique_recommendations.values())

def create_smart_recommendation_section():
    """创建智能推荐部分"""
    st.subheader("🤖 智能模型推荐")

    if st.session_state.classification_data is None:
        st.info("请先加载数据以获取智能推荐")
        return

    # 分析数据特征
    if st.button("🔍 分析数据并获取推荐", use_container_width=True):
        with st.spinner("正在分析数据特征..."):
            # 准备数据
            if st.session_state.data_source_type == "file":
                data_source = st.session_state.classification_data
                if st.session_state.selected_input_columns and st.session_state.selected_output_column:
                    X = data_source[st.session_state.selected_input_columns].copy()
                    y = data_source[st.session_state.selected_output_column].copy()
                else:
                    st.warning("请先选择输入特征和目标列")
                    return
            else:
                data_dict = st.session_state.classification_data
                X = data_dict['X'][st.session_state.selected_input_columns].copy() if st.session_state.selected_input_columns else data_dict['X'].copy()
                y = data_dict['y'].copy()

            # 基础数据清理
            X = X.apply(pd.to_numeric, errors='coerce')
            if y.dtype == 'object':
                le = LabelEncoder()
                y = pd.Series(le.fit_transform(y), index=y.index, name=y.name)

            # 分析数据特征
            characteristics = analyze_data_characteristics(X, y)

            # 获取推荐
            recommendations = recommend_models(characteristics)

            # 显示数据特征分析
            st.markdown("### 📊 数据特征分析")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("样本数量", characteristics['n_samples'])
                st.metric("特征数量", characteristics['n_features'])

            with col2:
                st.metric("类别数量", characteristics['n_classes'])
                st.metric("数据规模", characteristics['data_size'].upper())

            with col3:
                st.metric("类别平衡比", f"{characteristics['class_balance']:.2f}")
                st.metric("缺失值比例", f"{characteristics['missing_ratio']:.2%}")

            with col4:
                if 'max_correlation' in characteristics:
                    st.metric("最大特征相关性", f"{characteristics['max_correlation']:.3f}")
                st.metric("特征类型", characteristics['feature_types'].upper())

            # 显示数据特征标签
            st.markdown("### 🏷️ 数据特征标签")
            feature_tags = []
            if characteristics['is_binary']:
                feature_tags.append("二分类")
            if characteristics['is_imbalanced']:
                feature_tags.append("类别不平衡")
            if characteristics['has_missing']:
                feature_tags.append("包含缺失值")
            if characteristics.get('high_correlation', False):
                feature_tags.append("高特征相关性")

            if feature_tags:
                tag_html = " ".join([f'<span style="background-color: #f0f2f6; padding: 4px 8px; border-radius: 4px; margin-right: 8px;">{tag}</span>' for tag in feature_tags])
                st.markdown(tag_html, unsafe_allow_html=True)
            else:
                st.success("数据质量良好，无特殊问题")

            # 显示推荐结果
            st.markdown("### 🎯 模型推荐")

            # 分类推荐
            model_recommendations = [r for r in recommendations if 'model' in r]
            preprocessing_recommendations = [r for r in recommendations if 'preprocessing' in r]

            if model_recommendations:
                st.markdown("#### 推荐模型")

                # 按优先级分组
                high_priority = [r for r in model_recommendations if r['priority'] == 'high']
                medium_priority = [r for r in model_recommendations if r['priority'] == 'medium']

                if high_priority:
                    st.markdown("**🔥 强烈推荐**")
                    for rec in high_priority:
                        model_name = CLASSIFIER_INFO[rec['model']]['name']
                        st.success(f"✅ **{model_name}** - {rec['reason']}")

                if medium_priority:
                    st.markdown("**💡 建议考虑**")
                    for rec in medium_priority:
                        model_name = CLASSIFIER_INFO[rec['model']]['name']
                        st.info(f"ℹ️ **{model_name}** - {rec['reason']}")

            if preprocessing_recommendations:
                st.markdown("#### 预处理建议")
                for rec in preprocessing_recommendations:
                    preprocessing_name = PREPROCESSING_OPTIONS[rec['preprocessing']]['name']
                    if rec['priority'] == 'high':
                        st.warning(f"⚠️ **{preprocessing_name}** - {rec['reason']}")
                    else:
                        st.info(f"💡 **{preprocessing_name}** - {rec['reason']}")

            # 一键应用推荐
            st.markdown("### 🚀 快速应用推荐")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("📋 应用推荐模型", use_container_width=True):
                    # 自动选择推荐的高优先级模型
                    recommended_models = [r['model'] for r in high_priority[:3]]  # 最多选择3个
                    st.session_state.selected_models = recommended_models

                    # 初始化模型参数
                    for model in recommended_models:
                        if model not in st.session_state.model_params:
                            st.session_state.model_params[model] = get_default_params(model)

                    st.success(f"已选择推荐模型: {', '.join([CLASSIFIER_INFO[m]['name'] for m in recommended_models])}")

            with col2:
                if st.button("⚙️ 应用预处理建议", use_container_width=True):
                    # 应用预处理建议
                    preprocessing_config = st.session_state.get('preprocessing_config', {})

                    for rec in preprocessing_recommendations:
                        if rec['preprocessing'] == 'imbalance_handling':
                            preprocessing_config['enable_imbalance_handling'] = True
                        elif rec['preprocessing'] == 'missing_value_handling':
                            preprocessing_config['missing_method'] = 'knn'
                        elif rec['preprocessing'] == 'feature_selection':
                            preprocessing_config['enable_feature_selection'] = True

                    st.session_state.preprocessing_config = preprocessing_config
                    st.success("已应用预处理建议")

            # 保存分析结果
            st.session_state.data_analysis = {
                'characteristics': characteristics,
                'recommendations': recommendations,
                'analysis_timestamp': datetime.now().isoformat()
            }


def create_model_comparison_tab():
    """创建模型比较选项卡"""
    st.subheader("模型性能比较")

    # 收集所有模型的指标
    comparison_data = []
    for model_type, results in st.session_state.training_results_dict.items():
        if results.get('model') is not None:
            metrics = {
                '模型': CLASSIFIER_INFO[model_type]['name'],
                '训练准确率': results.get('train_accuracy', 0),
                '测试准确率': results.get('test_accuracy', 0),
                '训练精确率': results.get('train_precision', 0),
                '测试精确率': results.get('test_precision', 0),
                '训练召回率': results.get('train_recall', 0),
                '测试召回率': results.get('test_recall', 0),
                '训练F1': results.get('train_f1', 0),
                '测试F1': results.get('test_f1', 0)
            }
            comparison_data.append(metrics)

    if comparison_data:
        df = pd.DataFrame(comparison_data)

        # 创建比较图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 准确率比较
        ax = axes[0, 0]
        x = range(len(df))
        width = 0.35
        ax.bar([i - width / 2 for i in x], df['训练准确率'], width, label='训练集', alpha=0.8)
        ax.bar([i + width / 2 for i in x], df['测试准确率'], width, label='测试集', alpha=0.8)
        ax.set_xlabel('模型')
        ax.set_ylabel('准确率')
        ax.set_title('准确率比较')
        ax.set_xticks(x)
        ax.set_xticklabels(df['模型'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # F1分数比较
        ax = axes[0, 1]
        ax.bar([i - width / 2 for i in x], df['训练F1'], width, label='训练集', alpha=0.8)
        ax.bar([i + width / 2 for i in x], df['测试F1'], width, label='测试集', alpha=0.8)
        ax.set_xlabel('模型')
        ax.set_ylabel('F1分数')
        ax.set_title('F1分数比较')
        ax.set_xticks(x)
        ax.set_xticklabels(df['模型'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 雷达图 - 测试集指标
        ax = axes[1, 0]
        categories = ['准确率', '精确率', '召回率', 'F1分数']
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        ax = plt.subplot(2, 2, 3, projection='polar')
        for idx, row in df.iterrows():
            values = [row['测试准确率'], row['测试精确率'], row['测试召回率'], row['测试F1']]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=row['模型'])
            ax.fill(angles, values, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('测试集性能雷达图')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)

        # 散点图 - 训练vs测试准确率
        ax = axes[1, 1]
        colors = plt.cm.Set3(np.linspace(0, 1, len(df)))
        for idx, row in df.iterrows():
            ax.scatter(row['训练准确率'], row['测试准确率'],
                       s=200, c=[colors[idx]], alpha=0.6, edgecolors='black', linewidth=2)
            ax.annotate(row['模型'], (row['训练准确率'], row['测试准确率']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)

        # 添加对角线
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('训练准确率')
        ax.set_ylabel('测试准确率')
        ax.set_title('训练vs测试准确率')
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        # 显示详细比较表格
        st.markdown("### 详细指标对比")
        st.dataframe(df.style.highlight_max(axis=0, subset=[col for col in df.columns if col != '模型']),
                     use_container_width=True)


def auto_save_classification_state():
    """自动保存分类训练状态到临时文件"""
    if 'user_id' not in st.session_state or not st.session_state.user_id:
        return

    # 创建用户临时目录
    temp_dir = f"temp/user_{st.session_state.user_id}"
    os.makedirs(temp_dir, exist_ok=True)
    save_path = f"{temp_dir}/classification_state.pkl"

    try:
        # 收集需要保存的状态
        state_to_save = {
            'classification_data': st.session_state.get('classification_data'),
            'column_names': st.session_state.get('column_names', []),
            'selected_input_columns': st.session_state.get('selected_input_columns', []),
            'selected_output_column': st.session_state.get('selected_output_column'),
            'data_source_type': st.session_state.get('data_source_type', 'file'),
            'file_names': st.session_state.get('file_names'),
            'normalize_features': st.session_state.get('normalize_features', True),
            'test_size': st.session_state.get('test_size', 0.2),
            'current_model_type': st.session_state.get('current_model_type', 'catboost'),
            # 模型参数
            'catboost_params': st.session_state.get('catboost_params', {}),
            'random_forest_params': st.session_state.get('random_forest_params', {}),
            'svm_params': st.session_state.get('svm_params', {}),
            'xgboost_params': st.session_state.get('xgboost_params', {}),
            'neural_network_params': st.session_state.get('neural_network_params', {}),
            # 训练结果 (如果有)
            'training_results': st.session_state.get('training_results'),
            'cv_results': st.session_state.get('cv_results'),
            'model_trained_flag': st.session_state.get('model_trained_flag', False)
        }

        # 保存到本地文件
        import joblib
        with open(save_path, 'wb') as f:
            joblib.dump(state_to_save, f)

        # 记录保存时间
        with open(f"{temp_dir}/last_save.txt", 'w') as f:
            f.write(datetime.now().isoformat())

    except Exception as e:
        print(f"自动保存分类训练状态时出错: {e}")


def recover_classification_state():
    """恢复分类训练状态"""
    if 'user_id' not in st.session_state or not st.session_state.user_id:
        return False

    temp_dir = f"temp/user_{st.session_state.user_id}"
    save_path = f"{temp_dir}/classification_state.pkl"

    if os.path.exists(save_path):
        try:
            import joblib
            with open(save_path, 'rb') as f:
                saved_state = joblib.load(f)

            # 恢复状态到session_state
            for key, value in saved_state.items():
                st.session_state[key] = value

            print("分类训练状态已恢复")
            return True
        except Exception as e:
            print(f"恢复分类训练状态时出错: {e}")

    return False

# --- 示例数据集函数 ---
def add_example_datasets():
    """在侧边栏添加示例数据集加载功能"""
    st.sidebar.header("示例数据集")
    st.sidebar.info("选择一个经典数据集快速开始分类训练。")

    # 示例数据集字典
    datasets = {
        "鸢尾花 (Iris)": "iris",
        "葡萄酒 (Wine)": "wine",
        "乳腺癌 (Breast Cancer)": "breast_cancer",
        "手写数字 (Digits)": "digits"
    }

    selected_dataset_name = st.sidebar.selectbox(
        "选择数据集",
        list(datasets.keys()),
        key="example_dataset_select"
    )

    if st.sidebar.button("加载示例数据", key="load_example_btn"):
        dataset_key = datasets[selected_dataset_name]
        with st.sidebar.spinner(f"正在加载 {selected_dataset_name}..."):
            try:
                from sklearn import datasets as sklearn_datasets

                if dataset_key == "iris":
                    data = sklearn_datasets.load_iris(as_frame=True)
                    df = data.frame
                    output_col = 'target' # target 列包含类别 0, 1, 2
                    # 可以选择添加类别名称列
                    # df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

                elif dataset_key == "wine":
                    data = sklearn_datasets.load_wine(as_frame=True)
                    df = data.frame
                    output_col = 'target'

                elif dataset_key == "breast_cancer":
                    data = sklearn_datasets.load_breast_cancer(as_frame=True)
                    df = data.frame
                    output_col = 'target'

                elif dataset_key == "digits":
                    data = sklearn_datasets.load_digits(as_frame=True)
                    df = data.frame
                    output_col = 'target'
                else:
                    st.sidebar.error("选择的数据集无效。")
                    return

                # 清理列名 (移除特殊字符等)
                df.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in df.columns]
                output_col = "".join (c if c.isalnum() else "_" for c in str(output_col)) # 清理目标列名

                # 在会话状态中存储数据
                st.session_state.classification_data = df
                st.session_state.column_names = list(df.columns)
                st.session_state.data_source_type = "file" # 示例数据集按文件处理
                st.session_state.file_names = None

                # 自动选择特征和目标列
                # 输入特征：所有列除了目标列
                st.session_state.selected_input_columns = [col for col in df.columns if col != output_col]
                st.session_state.selected_output_column = output_col

                # 清除旧结果
                st.session_state.training_results = None
                st.session_state.cv_results = None
                st.session_state.model_trained_flag = False

                st.sidebar.success(f"已加载 {selected_dataset_name} ({len(df)} 行)")

                # 重新运行以更新UI和选项卡
                st.rerun()

            except ImportError:
                 st.sidebar.error("需要 'scikit-learn' 库来加载示例数据集。请先安装它。")
            except Exception as e:
                st.sidebar.error(f"加载示例数据集时出错: {str(e)}")


# --- 主函数入口 ---
def show_classification_page():
    """主函数，从主应用调用以显示分类训练页面"""

    # 添加侧边栏标题和示例数据集
    st.sidebar.title("分类模型训练")

    # 添加刷新按钮
    add_refresh_button()

    # 检查是否需要恢复状态
    is_refresh = 'refresh' in st.query_params and st.query_params['refresh'] == 'true'
    if is_refresh and 'is_restored' not in st.session_state:
        st.session_state.is_restored = recover_classification_state()

    add_example_datasets()

    # 显示主页面内容
    show_classification_training_page()

    # 自动保存当前状态
    auto_save_classification_state()


def local_save_session_state(user_id=None):
    """分类模块本地版本的会话状态保存函数"""
    # 如果未提供user_id，尝试从session_state获取
    if user_id is None and 'user_id' in st.session_state:
        user_id = st.session_state.user_id

    if not user_id:
        print("警告: 无法保存会话状态，未提供有效的用户ID")
        return

    # 创建保存目录
    save_dir = f"temp/session_states"
    os.makedirs(save_dir, exist_ok=True)

    # 准备要保存的状态数据
    state_to_save = {
        'current_page': st.session_state.get('current_page', 'classification_training'),
        'classification_data': None,  # 复杂对象不直接保存
        'selected_input_columns': st.session_state.get('selected_input_columns', []),
        'selected_output_column': st.session_state.get('selected_output_column'),
        'data_source_type': st.session_state.get('data_source_type', 'file'),
        'normalize_features': st.session_state.get('normalize_features', True),
        'test_size': st.session_state.get('test_size', 0.2),
        'timestamp': datetime.now().isoformat()
    }

    # 保存到文件
    try:
        save_path = f"{save_dir}/user_{user_id}_state.json"
        with open(save_path, 'w', encoding='utf-8') as f:
            # 只保存可JSON序列化的数据
            json_safe_state = {k: v for k, v in state_to_save.items()
                               if k != 'classification_data'}
            json.dump(json_safe_state, f, ensure_ascii=False, indent=2)

        print(f"会话状态已保存到: {save_path}")
        return True
    except Exception as e:
        print(f"保存会话状态时出错: {e}")
        return False


def add_refresh_button():
    """添加受控刷新按钮，不会丢失状态"""
    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("🔄 刷新页面", key="controlled_refresh"):
            # 保存当前状态
            if 'user_id' in st.session_state:
                # 先保存分类训练状态
                auto_save_classification_state()
                # 使用本地函数保存会话状态
                local_save_session_state(st.session_state.user_id)

            # 设置URL参数标记这是一个受控刷新
            st.query_params['refresh'] = 'true'
            st.query_params['page'] = st.session_state.current_page
            st.session_state.is_refreshing = True
            st.rerun()

def plot_feature_importance(importance_dict, top_n=15):
    """绘制特征重要性图"""
    # 排序特征重要性
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

    # 取前top_n个特征
    top_features = sorted_features[:top_n]

    if not top_features:
        return None

    # 准备数据
    features, importances = zip(*top_features)

    # 创建图表
    fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.4)))

    # 创建水平条形图
    bars = ax.barh(range(len(features)), importances, color='skyblue', alpha=0.8)

    # 设置标签
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel('重要性分数')
    ax.set_title(f'特征重要性 (Top {len(features)})')

    # 添加数值标签
    for i, (bar, importance) in enumerate(zip(bars, importances)):
        ax.text(bar.get_width() + max(importances) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{importance:.4f}', va='center', fontsize=9)

    # 美化图表
    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)

    # 反转y轴，使最重要的特征在顶部
    ax.invert_yaxis()

    plt.tight_layout()
    return fig

# --- 直接运行脚本时的入口 ---
if __name__ == "__main__":
    # 设置页面配置 (仅在直接运行此脚本时需要)
    st.set_page_config(page_title="分类模型训练", layout="wide")
    show_classification_page()
