# data_evaluator.py
import time

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
from scipy import stats

# --- 新增：添加缺失的 Sklearn 导入 ---
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score # 添加了 accuracy_score
from sklearn.model_selection import train_test_split
# --- 结束新增 ---

# --- 尝试导入字体工具 ---
try:
    # 假设 font_utils.py 在同一个目录下
    from font_utils import apply_plot_style, FONT_PROP, create_figure_with_safe_dimensions
    print("字体工具从 font_utils 成功加载 (in data_evaluator)")
except ImportError:
    print("警告: 无法从 font_utils 导入，将在 data_evaluator 中使用备用绘图设置。")
    FONT_PROP = None
    # 定义备用函数，避免后续代码出错
    def apply_plot_style(ax):
        ax.grid(True, linestyle='--', alpha=0.6)
        return ax
    def create_figure_with_safe_dimensions(w, h, dpi=80):
        # 简单的备用实现
        fig, ax = plt.subplots(figsize=(w, h), dpi=dpi)
        return fig, ax
# --- 结束字体导入 ---

# --- 常量定义 ---
MIN_SAMPLES_PER_CLASS = 10 # 分类任务每类最少样本阈值 (示例)
MIN_TOTAL_SAMPLES = 50     # 整体最少样本阈值 (示例)
HIGH_CORRELATION_THRESHOLD = 0.90 # 高相关性阈值 (示例)
LOW_VARIANCE_THRESHOLD = 0.01    # 低方差阈值 (示例)
HIGH_SKEWNESS_THRESHOLD = 1.0 # 偏度阈值
IQR_MULTIPLIER = 1.5      # IQR 方法的乘数
ZSCORE_THRESHOLD = 3        # Z-score 方法的阈值
HIGH_CARDINALITY_THRESHOLD = 50 # 高基数阈值
RARE_CATEGORY_THRESHOLD = 0.01  # 稀有类别频率阈值
LEAKAGE_CORR_THRESHOLD = 0.98   # 潜在数据泄露的相关性阈值 (回归)
LEAKAGE_AUC_THRESHOLD = 0.99    # 潜在数据泄露的AUC阈值 (分类，基于简单模型)
# --- 结束常量定义 ---


# ==============================================================
#               评估检查的辅助函数定义
# ==============================================================

def check_data_size(df):
    """检查数据集大小是否可能足够"""
    findings = []
    n_samples = len(df)
    if n_samples < MIN_TOTAL_SAMPLES:
        findings.append({
            'type': 'warning',
            'message': f"数据量较小 ({n_samples} 个样本)，可能不足以训练出泛化能力强的模型。"
        })
    else:
         findings.append({
            'type': 'info',
            'message': f"数据集包含 {n_samples} 个样本。"
        })
    return findings

def check_class_balance(y, task_type):
    """检查分类任务的类别平衡性"""
    findings = []
    class_counts = None # 初始化
    if task_type != 'Classification':
        return findings, class_counts # 只对分类任务有效

    if y is None or y.empty:
        findings.append({'type': 'error', 'message': '目标变量无效或为空，无法检查类别平衡。'})
        return findings, class_counts

    try:
        class_counts = y.value_counts()
        n_classes = len(class_counts)
        findings.append({'type': 'info', 'message': f"目标变量包含 {n_classes} 个类别。"})

        if n_classes > 1:
            min_count = class_counts.min()
            max_count = class_counts.max()
            imbalance_ratio = min_count / max_count if max_count > 0 else 1

            if min_count < MIN_SAMPLES_PER_CLASS:
                 minority_classes = class_counts[class_counts < MIN_SAMPLES_PER_CLASS].index.tolist()
                 findings.append({
                     'type': 'warning',
                     'message': f"以下类别样本过少 (少于 {MIN_SAMPLES_PER_CLASS} 个): {minority_classes}，最小类别样本数为 {min_count}。这可能影响模型学习这些类别。"
                 })
            if imbalance_ratio < 0.1: # 示例：严重不平衡阈值
                findings.append({
                    'type': 'warning',
                    'message': f"数据存在严重的类别不平衡 (最小/最大比例: {imbalance_ratio:.2f})。考虑使用数据平衡技术（如过采样/欠采样）或调整模型评估指标（如关注F1分数、AUC）。"
                })
            elif imbalance_ratio < 0.5: # 示例：中度不平衡
                 findings.append({
                    'type': 'info',
                    'message': f"数据存在一定的类别不平衡 (最小/最大比例: {imbalance_ratio:.2f})。建议关注模型在少数类上的性能。"
                })
            else:
                findings.append({'type': 'info', 'message': "类别分布相对均衡。"})
        elif n_classes == 1:
            findings.append({'type': 'warning', 'message': "目标变量只包含一个类别，无法进行分类任务。"})
        else: # n_classes == 0
             findings.append({'type': 'error', 'message': "目标变量中未找到有效类别。"})


    except Exception as e:
        findings.append({'type': 'error', 'message': f'检查类别平衡时出错: {e}'})
        class_counts = None # 出错时重置

    return findings, class_counts

def check_missing_values(df):
    """检查整个数据框中的缺失值"""
    findings = []
    if df is None or df.empty:
        findings.append({'type': 'error', 'message': '输入数据框为空，无法检查缺失值。'})
        return findings
    try:
        missing_counts = df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]
        if not missing_cols.empty:
            total_missing = int(missing_cols.sum()) # 确保是整数
            percent_missing = (total_missing / df.size) * 100 # df.size 是总元素数
            findings.append({
                'type': 'warning',
                'message': f"数据中发现 {total_missing} 个缺失值 ({percent_missing:.2f}% of total cells)。"
                           f" 存在缺失值的列: {missing_cols.index.tolist()}。"
                           f" 缺失值会影响大多数机器学习模型，建议使用 '缺失值处理' 功能进行填充或移除。"
            })
        else:
            findings.append({'type': 'info', 'message': "数据中未发现缺失值。"})
    except Exception as e:
        findings.append({'type': 'error', 'message': f"检查缺失值时出错: {e}"})
    return findings

def check_feature_variance(X):
    """检查数值特征的方差是否过低"""
    findings = []
    if X is None or X.empty:
        findings.append({'type': 'error', 'message': '特征数据X为空，无法检查方差。'})
        return findings
    try:
        numeric_X = X.select_dtypes(include=np.number)
        if numeric_X.empty:
            findings.append({'type': 'info', 'message': '未发现数值特征，跳过方差检查。'})
            return findings

        variances = numeric_X.var()
        # 排除方差为NaN或Inf的情况 (例如，如果列只有一个值或全是NaN)
        variances = variances.replace([np.inf, -np.inf], np.nan).dropna()
        low_variance_cols = variances[variances < LOW_VARIANCE_THRESHOLD].index.tolist()

        if low_variance_cols:
            findings.append({
                'type': 'warning',
                'message': f"以下特征方差较低 (低于 {LOW_VARIANCE_THRESHOLD})，可能提供的信息有限: {low_variance_cols}。考虑是否移除这些特征，因为它们对模型预测的贡献可能很小。"
            })
        else:
            findings.append({'type': 'info', 'message': "未检测到数值特征方差过低的情况。"})

    except Exception as e:
        findings.append({'type': 'error', 'message': f"检查特征方差时出错: {e}"})
    return findings


def check_feature_distribution(X):
    """检查数值特征的分布，识别偏态，并生成可视化"""
    findings = []
    visualizations = []
    if X is None or X.empty:
        findings.append({'type': 'error', 'message': '特征数据X为空，无法检查分布。'})
        return findings, visualizations

    numeric_cols = X.select_dtypes(include=np.number).columns
    if numeric_cols.empty:
        findings.append({'type': 'info', 'message': '未发现数值特征，跳过分布检查。'})
        return findings, visualizations

    highly_skewed_cols = []
    for col in numeric_cols:
        # 跳过方差过低的列
        try:
             # 检查方差前先移除NaN
            var_check = X[col].dropna().var()
            if pd.isna(var_check) or var_check < LOW_VARIANCE_THRESHOLD:
                 continue
        except Exception:
             continue # 计算方差出错也跳过

        fig = None # 初始化 fig
        try:
            # 计算偏度，处理NaN值
            skewness = X[col].dropna().skew()
            if pd.isna(skewness):
                 continue
            if abs(skewness) > HIGH_SKEWNESS_THRESHOLD:
                highly_skewed_cols.append(f"{col} ({skewness:.2f})")

            # --- 可视化 ---
            fig, ax = create_figure_with_safe_dimensions(6, 4, dpi=70)
            sns.histplot(X[col].dropna(), kde=True, ax=ax, bins=min(30, X[col].nunique()))
            apply_plot_style(ax)
            ax.set_title(f"特征 '{col}' 分布 (偏度: {skewness:.2f})", fontproperties=FONT_PROP if FONT_PROP else None, fontsize=10)
            ax.set_xlabel(col, fontproperties=FONT_PROP if FONT_PROP else None, fontsize=9)
            ax.set_ylabel("频率", fontproperties=FONT_PROP if FONT_PROP else None, fontsize=9)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
            plt.tight_layout()
            visualizations.append(fig)
            # --- 结束可视化 ---

        except Exception as e:
            findings.append({'type': 'warning', 'message': f"分析特征 '{col}' 的分布时出错: {e}"})
            if fig: plt.close(fig) # 如果绘图出错，尝试关闭

    if highly_skewed_cols:
        findings.append({
            'type': 'warning',
            'message': f"以下数值特征存在明显偏态 (偏度绝对值 > {HIGH_SKEWNESS_THRESHOLD}): {', '.join(highly_skewed_cols)}。"
                       f" 偏态分布可能违反某些模型（如线性回归、LDA）的假设，影响性能。考虑进行数据转换（如对数、平方根、Box-Cox变换）来使其更接近正态分布。"
        })
    else:
        findings.append({'type': 'info', 'message': '数值特征未发现明显偏态。'})

    return findings, visualizations

def check_categorical_features(X):
    """检查分类特征的基数和稀有类别，并生成可视化"""
    findings = []
    visualizations = []
    if X is None or X.empty:
        findings.append({'type': 'error', 'message': '特征数据X为空，无法检查分类特征。'})
        return findings, visualizations

    # 选择对象类型和唯一值较少的数值类型作为潜在分类特征
    categorical_cols = X.select_dtypes(include='object').columns.tolist()
    potential_cat_numeric = [col for col in X.select_dtypes(include=np.number).columns
                             if X[col].nunique() < HIGH_CARDINALITY_THRESHOLD / 2]
    categorical_cols.extend(potential_cat_numeric)
    # 排除布尔类型被误判为分类
    categorical_cols = [col for col in categorical_cols if X[col].dtype != 'bool']
    categorical_cols = sorted(list(set(categorical_cols)))


    if not categorical_cols:
        findings.append({'type': 'info', 'message': '未发现明显的分类特征，跳过相关检查。'})
        return findings, visualizations

    high_cardinality_cols = []
    rare_category_cols = {}

    for col in categorical_cols:
        fig = None # 初始化 fig
        try:
            # 计算唯一值数量，处理NaN
            col_data_clean = X[col].dropna()
            if col_data_clean.empty: continue # 如果列全是NaN
            n_unique = col_data_clean.nunique()

            if n_unique > HIGH_CARDINALITY_THRESHOLD:
                high_cardinality_cols.append(f"{col} ({n_unique})")

            # 计算频率
            value_counts = col_data_clean.value_counts(normalize=True)
            rare_categories = value_counts[value_counts < RARE_CATEGORY_THRESHOLD].index.tolist()
            if rare_categories:
                rare_category_cols[col] = rare_categories

            # --- 可视化 (限制类别数量) ---
            if 1 < n_unique <= 30:
                 fig, ax = create_figure_with_safe_dimensions(6, 4, dpi=70)
                 top_n = 20
                 plot_counts = col_data_clean.value_counts().nlargest(top_n)
                 sns.barplot(x=plot_counts.index.astype(str), y=plot_counts.values, ax=ax, palette="viridis")
                 apply_plot_style(ax)
                 ax.set_title(f"特征 '{col}' 频率 (Top {min(top_n, n_unique)})", fontproperties=FONT_PROP if FONT_PROP else None, fontsize=10)
                 ax.set_xlabel("类别", fontproperties=FONT_PROP if FONT_PROP else None, fontsize=9)
                 ax.set_ylabel("频率", fontproperties=FONT_PROP if FONT_PROP else None, fontsize=9)
                 plt.xticks(rotation=45, ha='right', fontsize=8, fontproperties=FONT_PROP if FONT_PROP else None)
                 plt.yticks(fontsize=8)
                 plt.tight_layout()
                 visualizations.append(fig)
            # --- 结束可视化 ---

        except Exception as e:
            findings.append({'type': 'warning', 'message': f"分析分类特征 '{col}' 时出错: {e}"})
            if fig: plt.close(fig)


    if high_cardinality_cols:
        findings.append({
            'type': 'warning',
            'message': f"以下分类特征基数过高 (唯一值 > {HIGH_CARDINALITY_THRESHOLD}): {', '.join(high_cardinality_cols)}。"
                       f" 这会给独热编码带来困难（维度爆炸），影响模型性能和训练时间。考虑降基数方法：如合并稀有类别、目标编码、频数编码、特征哈希，或将此特征视为文本/ID特征处理。"
        })
    if rare_category_cols:
        rare_info = "; ".join([f"{col}: {str(cats[:3])}..." if len(cats) > 3 else f"{col}: {str(cats)}" for col, cats in rare_category_cols.items()])
        findings.append({
            'type': 'warning',
            'message': f"以下分类特征包含稀有类别 (频率 < {RARE_CATEGORY_THRESHOLD*100}%): {rare_info}。"
                       f" 模型可能难以从稀有类别中学习，这些类别对模型性能贡献有限，还可能增加过拟合风险。考虑将稀有类别合并为 '其他' 类别，或使用特定处理方法。"
        })

    if not high_cardinality_cols and not rare_category_cols and categorical_cols:
         findings.append({'type': 'info', 'message': '分类特征的基数和频率分布在合理范围内。'})

    return findings, visualizations

def check_outliers_summary(X):
    """使用 IQR 方法初步检查数值特征中的离群点"""
    findings = []
    if X is None or X.empty:
        findings.append({'type': 'error', 'message': '特征数据X为空，无法检查离群点。'})
        return findings

    numeric_cols = X.select_dtypes(include=np.number).columns
    if numeric_cols.empty:
        findings.append({'type': 'info', 'message': '未发现数值特征，跳过离群点初步检查。'})
        return findings

    outlier_cols_details = {}
    for col in numeric_cols:
        try:
            col_data = X[col].dropna()
            if col_data.empty: continue

            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            if pd.isna(IQR) or IQR <= 1e-8: continue

            lower_bound = Q1 - IQR_MULTIPLIER * IQR
            upper_bound = Q3 + IQR_MULTIPLIER * IQR
            outliers_mask = (X[col] < lower_bound) | (X[col] > upper_bound)
            num_outliers = outliers_mask.sum()

            if num_outliers > 0:
                percent_outliers = (num_outliers / len(X)) * 100
                if percent_outliers > 1.0:
                    outlier_cols_details[col] = f"{percent_outliers:.1f}%"
        except Exception as e:
             findings.append({'type': 'warning', 'message': f"检查特征 '{col}' 的离群点时出错: {e}"})

    if outlier_cols_details:
         details_str = "; ".join([f"{col} ({perc})" for col, perc in outlier_cols_details.items()])
         findings.append({
             'type': 'warning',
             'message': f"初步检测发现以下数值特征可能包含较多离群点 (基于 >{IQR_MULTIPLIER} 倍 IQR): {details_str}。"
                        f" 离群点会显著影响均值、方差等统计量，对线性模型、SVM、KNN 等算法性能影响较大，可能拉偏模型拟合。建议使用 '异常点发现' 功能详细分析，并考虑处理（如盖帽法、移除、替换为中位数/均值、分箱）或使用对离群点不敏感的鲁棒模型（如树模型）。"
         })
    else:
        findings.append({'type': 'info', 'message': '数值特征初步检查未发现大量离群点 (基于 IQR)。'})

    return findings

def check_feature_correlation(X):
    """检查数值特征间的高相关性，并生成可视化"""
    findings = []
    visualizations = []
    if X is None or X.empty:
        findings.append({'type': 'error', 'message': '特征数据X为空，无法检查相关性。'})
        return findings, visualizations

    numeric_X = X.select_dtypes(include=np.number)
    if numeric_X.shape[1] < 2:
        findings.append({'type': 'info', 'message': "数值特征不足2个，无法计算相关性。"})
        return findings, visualizations

    fig_corr = None # 初始化 fig
    try:
        corr_matrix = numeric_X.corr()
        upper_triangle_mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        upper_triangle_values = corr_matrix.where(upper_triangle_mask)

        high_corr_features = []
        for col in upper_triangle_values.columns:
            correlated_rows = upper_triangle_values.index[upper_triangle_values[col].abs() > HIGH_CORRELATION_THRESHOLD].tolist()
            for row_label in correlated_rows:
                if pd.notna(upper_triangle_values.loc[row_label, col]): # 确保值不是 NaN
                    high_corr_features.append((col, row_label, corr_matrix.loc[row_label, col]))

        if high_corr_features:
             details = ", ".join([f"({p[0]}, {p[1]}: {p[2]:.2f})" for p in high_corr_features])
             findings.append({
                'type': 'warning',
                'message': f"发现以下数值特征对之间存在高相关性 (绝对值 > {HIGH_CORRELATION_THRESHOLD}): {details}。"
                           f" 高度相关的特征可能引入多重共线性问题，影响线性模型系数的稳定性和解释性。考虑移除其中一个特征、使用PCA进行降维，或选用对共线性不敏感的模型（如随机森林、XGBoost）。"
             })
        else:
             findings.append({'type': 'info', 'message': f"未发现数值特征之间存在强相关性 (阈值 > {HIGH_CORRELATION_THRESHOLD})。"})

        # --- 可视化：相关性热力图 ---
        max_heatmap_features = 50
        if numeric_X.shape[1] <= max_heatmap_features:
             plot_cols = numeric_X.columns
             plot_corr_matrix = corr_matrix
        else:
             top_var_cols = numeric_X.var().nlargest(max_heatmap_features).index
             plot_cols = top_var_cols
             plot_corr_matrix = numeric_X[plot_cols].corr()
             findings.append({'type': 'info', 'message': f"特征过多，相关性热力图仅显示方差最大的 {max_heatmap_features} 个特征。"})

        if not plot_cols.empty and not plot_corr_matrix.empty:
             fig_corr, ax_corr = create_figure_with_safe_dimensions(min(10, len(plot_cols)*0.5 + 1), min(8, len(plot_cols)*0.4 + 1), dpi=80)
             sns.heatmap(plot_corr_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax_corr, vmin=-1, vmax=1)
             ax_corr.set_title("特征相关性热力图", fontproperties=FONT_PROP if FONT_PROP else None)
             plt.setp(ax_corr.get_xticklabels(), rotation=45, ha="right", fontproperties=FONT_PROP if FONT_PROP else None, fontsize=7)
             plt.setp(ax_corr.get_yticklabels(), rotation=0, fontproperties=FONT_PROP if FONT_PROP else None, fontsize=7)
             plt.tight_layout()
             visualizations.append(fig_corr)
        # --- 结束可视化 ---

    except Exception as e:
        findings.append({'type': 'error', 'message': f"检查特征相关性时出错: {e}"})
        if fig_corr: plt.close(fig_corr)

    return findings, visualizations

def analyze_target_variable(y, task_type):
    """分析目标变量的分布"""
    findings = []
    visualizations = []
    fig_target_reg = None # 初始化
    fig_target_cls = None # 初始化

    if y is None or y.empty:
        findings.append({'type':'error', 'message':'目标变量无效或为空，无法分析。'})
        return findings, visualizations

    target_name = y.name if y.name else "目标变量"

    try:
        if task_type == 'Regression':
            if not pd.api.types.is_numeric_dtype(y):
                 findings.append({'type':'error', 'message': f"回归任务的目标变量 '{target_name}' 不是数值类型。"}); return findings, visualizations
            # 离群点检查
            y_clean = y.dropna()
            if not y_clean.empty:
                Q1 = y_clean.quantile(0.25); Q3 = y_clean.quantile(0.75); IQR = Q3 - Q1
                if pd.notna(IQR) and IQR > 1e-8:
                    lower = Q1 - IQR_MULTIPLIER * IQR; upper = Q3 + IQR_MULTIPLIER * IQR
                    outliers_y = y[(y < lower) | (y > upper)]
                    if not outliers_y.empty:
                         perc = (len(outliers_y) / len(y)) * 100
                         if perc > 1.0: findings.append({'type':'warning', 'message': f"目标变量 '{target_name}' 发现约 {perc:.1f}% 潜在离群点(IQR)。"})
            # 可视化
            fig_target_reg, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=80)
            sns.histplot(y_clean, kde=True, ax=axes[0]); apply_plot_style(axes[0])
            axes[0].set_title(f"'{target_name}' 分布", fontproperties=FONT_PROP, fontsize=10); axes[0].set_xlabel(target_name, fontproperties=FONT_PROP, fontsize=9); axes[0].set_ylabel("频率", fontproperties=FONT_PROP, fontsize=9)
            sns.boxplot(y=y_clean, ax=axes[1], color="lightblue"); apply_plot_style(axes[1])
            axes[1].set_title(f"'{target_name}' 箱线图", fontproperties=FONT_PROP, fontsize=10); axes[1].set_ylabel(target_name, fontproperties=FONT_PROP, fontsize=9); axes[1].tick_params(axis='x', bottom=False, labelbottom=False)
            plt.tight_layout(); visualizations.append(fig_target_reg)
            # 偏度检查
            skewness = y_clean.skew()
            if pd.notna(skewness) and abs(skewness) > HIGH_SKEWNESS_THRESHOLD: findings.append({'type':'warning', 'message': f"回归目标变量 '{target_name}' 存在明显偏态 (Skewness: {skewness:.2f})。考虑变换。"})

        elif task_type == 'Classification':
            if pd.api.types.is_numeric_dtype(y): findings.append({'type':'info', 'message': f"分类目标 '{target_name}' 是数值类型。"})
            elif pd.api.types.is_string_dtype(y) or pd.api.types.is_categorical_dtype(y) or y.dtype == 'object': findings.append({'type':'info', 'message': f"分类目标 '{target_name}' 是文本/类别类型。"})
            else: findings.append({'type':'warning', 'message': f"分类目标 '{target_name}' 类型 ({y.dtype}) 未知。"})
            # 分类分布图 (依赖 check_class_balance 函数中返回的 class_counts)
            # 注意：此处假设 check_class_balance 先被调用且返回了 class_counts
            # 这在 evaluate_data 函数的结构中是保证的
            class_counts_local = y.value_counts().sort_index() # 重新计算以防万一
            if not class_counts_local.empty and len(class_counts_local) > 1:
                 fig_target_cls, ax_target_cls = create_figure_with_safe_dimensions(8, 5, dpi=80)
                 apply_plot_style(ax_target_cls)
                 bars = ax_target_cls.bar(class_counts_local.index.astype(str), class_counts_local.values, color=plt.cm.viridis(np.linspace(0, 1, len(class_counts_local))))
                 ax_target_cls.set_title(f"目标变量 '{target_name}' 分布 (分类)", fontproperties=FONT_PROP if FONT_PROP else None)
                 ax_target_cls.set_xlabel("类别", fontproperties=FONT_PROP if FONT_PROP else None); ax_target_cls.set_ylabel("样本数量", fontproperties=FONT_PROP if FONT_PROP else None)
                 for bar in bars: height = bar.get_height(); ax_target_cls.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
                 plt.xticks(rotation=45, ha='right', fontsize=8, fontproperties=FONT_PROP if FONT_PROP else None)
                 plt.tight_layout(); visualizations.append(fig_target_cls)

    except Exception as e:
        findings.append({'type': 'error', 'message': f"分析目标变量 '{target_name}' 时出错: {e}"})
        if fig_target_reg: plt.close(fig_target_reg)
        if fig_target_cls: plt.close(fig_target_cls)

    return findings, visualizations

def check_data_leakage_heuristic(X, y, task_type):
    """使用启发式方法检查潜在的数据泄露风险"""
    findings = []
    if X is None or X.empty or y is None or y.empty:
        return findings # 不能检查

    numeric_X = X.select_dtypes(include=np.number)
    if numeric_X.empty:
        findings.append({'type':'info', 'message':'未发现数值特征，跳过基于相关性/AUC的数据泄露检查。'})
        return findings

    suspicious_features = []

    if task_type == 'Regression':
        if pd.api.types.is_numeric_dtype(y):
            try:
                y_clean = y.dropna()
                X_clean = numeric_X.loc[y_clean.index] # 对齐 X
                if not X_clean.empty:
                    correlations = X_clean.corrwith(y_clean).abs().dropna()
                    leaky = correlations[correlations > LEAKAGE_CORR_THRESHOLD]
                    if not leaky.empty:
                        suspicious_features.extend([(f, f"与目标相关性 {leaky[f]:.3f}") for f in leaky.index])
            except Exception as e: findings.append({'type':'warning', 'message': f'计算回归泄露相关性时出错: {e}'})

    elif task_type == 'Classification':
        try:
            y_clean = y.dropna()
            X_clean = numeric_X.loc[y_clean.index] # 对齐 X
            if X_clean.empty or y_clean.empty: return findings

            if not pd.api.types.is_numeric_dtype(y_clean):
                 le = LabelEncoder()
                 y_encoded = le.fit_transform(y_clean)
            else:
                 y_encoded = y_clean.astype(int)

            n_classes = len(np.unique(y_encoded))
            if n_classes <= 1: return findings # 无法计算 AUC

            for col in X_clean.columns:
                try:
                    X_single = X_clean[[col]].copy()
                    if X_single[col].isnull().any():
                        X_single[col].fillna(X_single[col].median(), inplace=True)
                    if X_single[col].var() < 1e-8: continue # 跳过低方差

                    if len(X_single) < 10 or n_classes > len(y_encoded) // 2: continue

                    X_train, X_test, y_train, y_test = train_test_split(X_single, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

                    if len(np.unique(y_train)) > 1 and len(np.unique(y_test)) > 1:
                         model = LogisticRegression(random_state=42, class_weight='balanced', solver='liblinear', max_iter=100) # 增加 max_iter
                         model.fit(X_train, y_train)
                         if hasattr(model, "predict_proba"):
                             y_prob = model.predict_proba(X_test)
                             if n_classes == 2: auc_val = roc_auc_score(y_test, y_prob[:, 1])
                             else: auc_val = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
                             if auc_val > LEAKAGE_AUC_THRESHOLD: suspicious_features.append((col, f"单独AUC {auc_val:.3f}"))
                         else:
                             y_pred_leak = model.predict(X_test)
                             acc_leak = accuracy_score(y_test, y_pred_leak)
                             if acc_leak > LEAKAGE_AUC_THRESHOLD: suspicious_features.append((col, f"单独准确率 {acc_leak:.3f}"))
                except ValueError as ve: pass # 忽略分层错误等
                except Exception as e_inner: findings.append({'type':'error', 'message': f"检查特征'{col}'泄露时出错: {e_inner}"})
        except Exception as e_outer: findings.append({'type':'error', 'message': f"检查分类任务泄露时出错: {e_outer}"})

    # 报告结果
    if suspicious_features:
        details_str = "; ".join([f"{f} ({reason})" for f, reason in suspicious_features])
        findings.append({
            'type': 'error',
            'message': f"潜在数据泄露风险！以下特征与目标变量关联极高: {details_str}。"
                       f" **强烈建议在建模前移除这些特征！**"
        })
    else:
        findings.append({'type': 'info', 'message': '未检测到明显的单一特征数据泄露风险。'})

    return findings

def assess_generalization_risk(n_samples, n_features, findings):
    """基于已有发现评估模型的泛化风险"""
    risk_factors = set() # 使用集合去重
    recommendations = set()

    # 维度灾难风险
    if n_samples > 0 and n_features > 0:
        ratio = n_features / n_samples
        if n_features > n_samples: risk_factors.add("高维稀疏(特征>样本)"); recommendations.add("降维/特征选择"); recommendations.add("强正则化"); recommendations.add("获取更多数据")
        elif ratio > 0.1: risk_factors.add("特征相对较多"); recommendations.add("特征选择/正则化"); recommendations.add("交叉验证")

    # 根据发现添加风险/建议
    for finding in findings:
        if finding['type'] in ['warning', 'error']:
            msg = finding['message'].lower()
            if "数据量较小" in msg: risk_factors.add("数据量不足"); recommendations.add("获取更多数据"); recommendations.add("数据增强"); recommendations.add("简单模型"); recommendations.add("交叉验证")
            if "类别不平衡" in msg: risk_factors.add("类别不平衡"); recommendations.add("数据平衡"); recommendations.add("调整类别权重"); recommendations.add("使用不平衡指标")
            if "缺失值" in msg: risk_factors.add("数据缺失"); recommendations.add("缺失值处理")
            if "方差较低" in msg: risk_factors.add("低信息量特征"); recommendations.add("移除低方差特征")
            if "高相关性" in msg: risk_factors.add("特征共线性"); recommendations.add("移除相关特征"); recommendations.add("PCA"); recommendations.add("正则化"); recommendations.add("树模型")
            if "明显偏态" in msg: risk_factors.add("特征分布偏斜"); recommendations.add("数据变换"); recommendations.add("使用对分布不敏感模型")
            if "离群点" in msg: risk_factors.add("潜在离群点"); recommendations.add("离群点处理"); recommendations.add("使用鲁棒模型")
            if "数据泄露" in msg: risk_factors.add("数据泄露"); recommendations.add("移除泄露特征！")
            if "基数过高" in msg: risk_factors.add("高基数分类特征"); recommendations.add("合适编码/降基数")
            if "稀有类别" in msg: risk_factors.add("稀有分类类别"); recommendations.add("合并类别")

    # 总结
    final_findings = []
    if risk_factors:
        final_findings.append({'type': 'warning', 'message': f"潜在泛化风险因素: {'; '.join(sorted(list(risk_factors)))}。"})
        final_findings.append({'type': 'recommendation', 'message': f"建议措施: {'; '.join(sorted(list(recommendations)))}。"})
    else:
        final_findings.append({'type': 'info', 'message': "初步评估未发现明显影响泛化能力的风险。"})
    return final_findings


# --- 主要评估逻辑函数 (调用上面的辅助函数) ---
def evaluate_data(df, target_col, feature_cols, task_type):
    """
    执行各种数据检查并返回结果和可视化图表。
    """
    all_findings = []
    all_visualizations = []

    # --- 1. 输入验证 ---
    if df is None or df.empty: all_findings.append({'type': 'error', 'message': '数据为空。'}); return all_findings, all_visualizations
    if not target_col or target_col not in df.columns: all_findings.append({'type': 'error', 'message': '未选有效目标列。'}); return all_findings, all_visualizations
    if not feature_cols or not all(col in df.columns for col in feature_cols): all_findings.append({'type': 'error', 'message': '未选有效特征列。'}); return all_findings, all_visualizations

    # --- 2. 数据准备 ---
    try:
        y = df[target_col].copy()
        X = df[feature_cols].copy()
        n_samples_orig, n_features = X.shape
        all_findings.append({'type': 'info', 'message': f"原始数据维度: {n_samples_orig} 样本, {n_features} 特征。"})

        # 特征类型转换
        for col in X.columns:
             if X[col].dtype == 'object': X[col] = pd.to_numeric(X[col], errors='ignore')

        # 目标变量处理
        if y.dtype == 'object':
            y_numeric = pd.to_numeric(y, errors='coerce')
            if not y_numeric.isnull().all(): y = y_numeric
            elif task_type == 'Regression': all_findings.append({'type': 'error', 'message': f'回归目标列 {target_col} 含非数值。'}); return all_findings, all_visualizations

        # 移除目标变量为 NaN 的行
        nan_indices = y[y.isnull()].index
        if not nan_indices.empty:
            num_nan_y = len(nan_indices)
            all_findings.append({'type': 'warning', 'message': f'目标列发现 {num_nan_y} 个缺失值，已移除对应样本。'})
            X = X.drop(index=nan_indices)
            y = y.drop(index=nan_indices)
            if X.empty: all_findings.append({'type': 'error', 'message': '移除目标列缺失值后数据为空。'}); return all_findings, all_visualizations

        n_samples_clean, n_features = X.shape # 清理后的样本数

    except KeyError as ke: all_findings.append({'type': 'error', 'message': f'数据准备出错：找不到列 {ke}。'}); return all_findings, all_visualizations
    except Exception as e: all_findings.append({'type': 'error', 'message': f'准备数据时出错: {e}'}); return all_findings, all_visualizations

    if y is None or y.empty: all_findings.append({'type': 'error', 'message': '目标列无效或为空。'}); return all_findings, all_visualizations

    # --- 3. 执行检查 ---
    all_findings.append({'type': 'header', 'message': '数据质量检查'})
    all_findings.extend(check_data_size(df))
    all_findings.extend(check_missing_values(X))

    all_findings.append({'type': 'header', 'message': '特征 (X) 分析'})
    all_findings.extend(check_feature_variance(X))
    dist_findings, dist_visualizations = check_feature_distribution(X); all_findings.extend(dist_findings); all_visualizations.extend(dist_visualizations)
    cat_findings, cat_visualizations = check_categorical_features(X); all_findings.extend(cat_findings); all_visualizations.extend(cat_visualizations)
    all_findings.extend(check_outliers_summary(X))
    corr_findings, corr_visualizations = check_feature_correlation(X); all_findings.extend(corr_findings); all_visualizations.extend(corr_visualizations)

    all_findings.append({'type': 'header', 'message': '目标变量 (Y) 分析'})
    class_counts = None
    if task_type == 'Classification':
        balance_findings, class_counts = check_class_balance(y, task_type); all_findings.extend(balance_findings)
    target_findings, target_visualizations = analyze_target_variable(y, task_type); all_findings.extend(target_findings); all_visualizations.extend(target_visualizations)

    all_findings.append({'type': 'header', 'message': '潜在数据泄露检查'})
    all_findings.extend(check_data_leakage_heuristic(X, y, task_type))

    all_findings.append({'type': 'header', 'message': '泛化能力评估与建模建议'})
    all_findings.extend(assess_generalization_risk(n_samples_clean, n_features, all_findings))

    print(f"评估完成，共 {len(all_findings)} 条发现，{len(all_visualizations)} 个可视化图表。")
    return all_findings, all_visualizations


# ==============================================================
#               Streamlit UI 函数 (由 app.py 调用)
# ==============================================================
def show_data_evaluator_page():
    """显示数据评估页面的 Streamlit UI 元素"""
    st.title("📝 数据评估与建模建议")
    st.markdown("上传数据，选择任务类型和特征，系统将分析数据质量、特征分布、目标变量特性，并评估其对机器学习建模（分类/回归）的适用性及潜在风险。")
    st.markdown("---")

    # 初始化会话状态 (特定于此模块)
    if 'de_data' not in st.session_state: st.session_state.de_data = None
    if 'de_task_type' not in st.session_state: st.session_state.de_task_type = 'Classification'
    if 'de_target_col' not in st.session_state: st.session_state.de_target_col = None
    if 'de_feature_cols' not in st.session_state: st.session_state.de_feature_cols = []
    if 'de_evaluation_results' not in st.session_state: st.session_state.de_evaluation_results = None
    if 'de_visualizations' not in st.session_state: st.session_state.de_visualizations = []

    # --- 1. 数据上传 ---
    st.header("1. 上传数据")
    uploaded_file_eval = st.file_uploader("选择 CSV 或 Excel 文件进行评估", type=["csv", "xlsx", "xls"], key="de_uploader")

    if uploaded_file_eval:
        if st.button("加载评估数据", key="de_load_btn"):
            with st.spinner("加载并预处理数据..."):
                try:
                    data = pd.read_csv(uploaded_file_eval) if uploaded_file_eval.name.lower().endswith('.csv') else pd.read_excel(uploaded_file_eval)
                    data.dropna(axis=1, how='all', inplace=True)
                    if data.empty:
                        st.error("上传的文件为空或处理后为空。")
                        st.session_state.de_data = None
                    else:
                        st.session_state.de_data = data
                        st.session_state.de_target_col = None
                        st.session_state.de_feature_cols = []
                        st.session_state.de_evaluation_results = None
                        st.session_state.de_visualizations = []
                        st.success(f"成功加载文件: {uploaded_file_eval.name} ({data.shape[0]}行, {data.shape[1]}列)")
                except Exception as e:
                    st.error(f"加载数据出错: {e}")
                    st.session_state.de_data = None

    # --- 2. 配置与运行评估 ---
    if st.session_state.de_data is not None:
        df = st.session_state.de_data
        st.dataframe(df.head())
        st.markdown("---")
        st.header("2. 配置评估参数")

        col1, col2 = st.columns(2)
        with col1:
             st.session_state.de_task_type = st.radio("选择目标任务类型", ['Classification', 'Regression'], key='de_task_radio', horizontal=True, help="选择您最终想用这个数据进行的机器学习任务类型。")
             all_cols = df.columns.tolist()
             target_options = [None] + all_cols
             current_target_index = 0
             if st.session_state.de_target_col in all_cols:
                  try: current_target_index = target_options.index(st.session_state.de_target_col)
                  except ValueError: st.session_state.de_target_col = None
             st.session_state.de_target_col = st.selectbox("选择目标变量 (Y)", target_options, index=current_target_index, key='de_target_select', help="您想要预测或分类的列。")

        with col2:
             if st.session_state.de_target_col:
                 feature_options = [col for col in all_cols if col != st.session_state.de_target_col]
                 default_features = [col for col in st.session_state.de_feature_cols if col in feature_options]
                 if not default_features and feature_options:
                      default_features = feature_options
                 st.session_state.de_feature_cols = st.multiselect("选择特征变量 (X)", feature_options, default=default_features, key='de_feature_select', help="用于预测目标变量的列。")
             else:
                 st.multiselect("选择特征变量 (X)", [], disabled=True, key='de_feature_select')
                 st.warning("请先选择目标变量")

        st.markdown("---")
        # 评估按钮
        can_evaluate = st.session_state.de_target_col and st.session_state.de_feature_cols
        if st.button("📈 开始数据评估", key="de_run_btn", disabled=not can_evaluate, type="primary", use_container_width=True):
             if can_evaluate:
                 st.session_state.de_evaluation_results = None
                 st.session_state.de_visualizations = []
                 with st.spinner("正在执行数据评估分析..."):
                    time.sleep(0.5)
                    try:
                        findings, visualizations = evaluate_data(
                             df,
                             st.session_state.de_target_col,
                             st.session_state.de_feature_cols,
                             st.session_state.de_task_type
                        )
                        st.session_state.de_evaluation_results = findings
                        st.session_state.de_visualizations = visualizations
                        st.success("数据评估完成！结果如下所示。")
                    except Exception as eval_e:
                         st.error(f"评估过程中发生错误: {eval_e}")
                         st.code(traceback.format_exc())

    # --- 3. 显示评估结果 ---
    if st.session_state.de_evaluation_results:
        st.markdown("---")
        st.header("3. 评估结果与建议")

        current_expander = None # 用于追踪当前活动的 expander

        # 使用 Expander 分组显示结果
        for finding in st.session_state.de_evaluation_results:
            msg_type = finding.get('type', 'info')
            message = finding.get('message', '')

            if msg_type == 'header':
                # 创建新的 Expander
                current_expander = st.expander(f"**{message}**", expanded=True)
            elif current_expander: # 如果当前有活动的 expander
                 # 在当前 expander 内显示内容
                 with current_expander:
                     # --- 修正 SyntaxError 的地方 ---
                     message_html = message.replace('\n', '<br>')  # 先进行替换操作
                     if msg_type == 'info':
                         st.markdown(f"<div class='eval-finding info'>ℹ️ {message_html}</div>",
                                     unsafe_allow_html=True)  # 使用新变量
                     elif msg_type == 'warning':
                         st.markdown(f"<div class='eval-finding warning'>⚠️ **警告:** {message_html}</div>",
                                     unsafe_allow_html=True)  # 使用新变量
                     elif msg_type == 'error':
                         st.markdown(f"<div class='eval-finding error'>❌ **错误:** {message_html}</div>",
                                     unsafe_allow_html=True)  # 使用新变量
                     elif msg_type == 'recommendation':
                         st.markdown(f"<div class='eval-finding recommendation'>💡 **建议:** {message_html}</div>",
                                     unsafe_allow_html=True)  # 使用新变量
                     else:
                         st.markdown(f"➡️ {message_html}")  # 使用新变量 (如果默认也需要换行)
                     # --- 结束修正 ---
            else: # 备用：如果消息在任何 header 之前出现 (理论上不应发生，因为第一个 finding 应该是 header)
                  if msg_type == 'info': st.info(f"ℹ️ {message}")
                  elif msg_type == 'warning': st.warning(f"⚠️ **警告:** {message}")
                  elif msg_type == 'error': st.error(f"❌ **错误:** {message}")
                  elif msg_type == 'recommendation': st.success(f"💡 **建议:** {message}")
                  else: st.markdown(f"➡️ {message}")


        # 显示所有可视化图表
        if st.session_state.de_visualizations:
             st.markdown("---")
             st.header("相关可视化图表")
             num_viz = len(st.session_state.de_visualizations)
             cols_per_row = 2
             viz_cols = st.columns(cols_per_row)
             for i, fig in enumerate(st.session_state.de_visualizations):
                  col_index = i % cols_per_row
                  with viz_cols[col_index]:
                       try:
                            plot_title = f"图表 {i+1}"
                            if fig.axes and fig.axes[0].get_title(): plot_title = fig.axes[0].get_title()
                            # st.write(plot_title) # Optional: display title above plot
                            st.pyplot(fig)
                            plt.close(fig)
                       except Exception as viz_e:
                            st.warning(f"显示图表 {i+1} ('{plot_title}') 时出错: {viz_e}")
                            try: plt.close(fig)
                            except: pass

# --- 可选: 用于独立测试 ---
# if __name__ == "__main__":
#     st.set_page_config(layout="wide", page_title="数据评估模块测试")
#     show_data_evaluator_page()