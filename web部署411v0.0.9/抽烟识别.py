# -*- coding: utf-8 -*-

"""
使用 CatBoost 基于时间序列特征对Excel文件进行多分类。
将'烟支'、'棉签'、'镊子'分为三个不同的类别。
每个Excel文件被视为一个样本，从'上环电容值'和'下环电容值'
时间序列中提取特征。
"""

import pandas as pd
import numpy as np
import os
import re
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import sys
import time
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib

plt.rcParams['font.sans-serif'] = ['SimHei', 'SimSun', 'Microsoft YaHei', 'Arial Unicode MS',
                                   'DejaVu Sans']  # 优先使用的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.family'] = 'sans-serif'  # 使用无衬线字体

# --- 配置区域 ---
# !!! 修改为你存放 Excel 文件的文件夹路径 !!!
data_folder_path = r'E:\yjsproject\插烟状态识别\插拔电容数据'  # <--- 检查路径是否正确

# 要读取的列名 (确保你的 Excel 文件中有这些列名)
CAP_COL_UPPER = '上环电容值'
CAP_COL_LOWER = '下环电容值'

target_col = 'Category'  # 用于存储从文件名提取的类别
file_extensions = ('.xls', '.xlsx')

# 设置随机种子 (用于模型训练和数据拆分)
RANDOM_SEED = 30

# 模型输出路径
output_folder = 'model_output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# --- 函数定义 ---

def get_category_from_filename(filename):
    """
    从文件名中提取类别。
    将'烟支'、'棉签'、'镊子'分为三个不同的类别。
    """
    filename_lower = filename.lower()
    if '烟支' in filename_lower:
        return '烟支'
    elif '棉签' in filename_lower:
        return '棉签'
    elif '镊子' in filename_lower:
        return '镊子'
    else:
        # print(f" 信息：文件名 '{filename}' 不含已知类别关键词，将忽略。")
        return None


def extract_time_series_features(df_file, col_upper, col_lower):
    """
    从单个文件的数据 (DataFrame) 中提取时间序列特征。
    输入:
        df_file: 包含单次插拔事件数据的 Pandas DataFrame。
        col_upper: 上环电容列名。
        col_lower: 下环电容列名。
    输出:
        一个包含提取特征的字典，如果数据无效则返回 None。
    """
    features = {}
    try:
        # 确保数据是数值类型，无效值转为 NaN
        series_upper = pd.to_numeric(df_file[col_upper], errors='coerce')
        series_lower = pd.to_numeric(df_file[col_lower], errors='coerce')

        # 删除包含 NaN 的行，以防影响计算
        valid_indices = series_upper.notna() & series_lower.notna()
        series_upper = series_upper[valid_indices]
        series_lower = series_lower[valid_indices]

        # 如果清理后数据太少，则跳过
        if len(series_upper) < 5:  # 至少需要几个点来计算有意义的特征
            # print(f" 警告：有效数据点过少 ({len(series_upper)}), 跳过特征提取。")
            return None

        # --- 特征提取 ---
        # 1. 基本统计量
        features[f'{col_upper}_mean'] = series_upper.mean()
        features[f'{col_upper}_std'] = series_upper.std()
        features[f'{col_upper}_max'] = series_upper.max()
        features[f'{col_upper}_min'] = series_upper.min()
        features[f'{col_upper}_range'] = series_upper.max() - series_upper.min()
        features[f'{col_upper}_median'] = series_upper.median()

        features[f'{col_lower}_mean'] = series_lower.mean()
        features[f'{col_lower}_std'] = series_lower.std()
        features[f'{col_lower}_max'] = series_lower.max()
        features[f'{col_lower}_min'] = series_lower.min()
        features[f'{col_lower}_range'] = series_lower.max() - series_lower.min()
        features[f'{col_lower}_median'] = series_lower.median()

        # 2. 变化率特征 (一阶差分)
        diff_upper = series_upper.diff().dropna()
        diff_lower = series_lower.diff().dropna()

        if not diff_upper.empty:
            features[f'{col_upper}_diff_mean'] = diff_upper.mean()
            features[f'{col_upper}_diff_std'] = diff_upper.std()
            features[f'{col_upper}_diff_max'] = diff_upper.max()
            features[f'{col_upper}_diff_min'] = diff_upper.min()
        else:
            features[f'{col_upper}_diff_mean'] = 0
            features[f'{col_upper}_diff_std'] = 0
            features[f'{col_upper}_diff_max'] = 0
            features[f'{col_upper}_diff_min'] = 0

        if not diff_lower.empty:
            features[f'{col_lower}_diff_mean'] = diff_lower.mean()
            features[f'{col_lower}_diff_std'] = diff_lower.std()
            features[f'{col_lower}_diff_max'] = diff_lower.max()
            features[f'{col_lower}_diff_min'] = diff_lower.min()
        else:
            features[f'{col_lower}_diff_mean'] = 0
            features[f'{col_lower}_diff_std'] = 0
            features[f'{col_lower}_diff_max'] = 0
            features[f'{col_lower}_diff_min'] = 0

        # 3. 相关性
        if series_upper.std() > 1e-6 and series_lower.std() > 1e-6:
            correlation = series_upper.corr(series_lower, method='spearman')
            features['correlation_spearman'] = correlation if pd.notna(correlation) else 0
        else:
            features['correlation_spearman'] = 0

        # 确保所有特征都是有限数值
        for key, value in features.items():
            if not np.isfinite(value):
                features[key] = 0

        return features

    except KeyError as e:
        print(f" 错误：读取文件时缺少列 '{e}'。跳过此文件。")
        return None
    except Exception as e:
        print(f" 错误：在提取特征时发生意外错误: {type(e).__name__} - {e}")
        return None


def try_read_excel_robust(file_path, required_cols):
    """
    更健壮地尝试读取 Excel 文件，检查所需列。
    """
    try:
        try:
            df = pd.read_excel(file_path, engine='openpyxl', sheet_name=0, header=0)
        except Exception:
            try:
                df = pd.read_excel(file_path, engine='xlrd', sheet_name=0, header=0)
            except ImportError:
                print("\n错误：读取旧版 .xls 文件需要 'xlrd' 库。请运行: pip install xlrd")
                return None
            except Exception as e_inner:
                raise e_inner

        df.columns = df.columns.str.strip()
        actual_columns = df.columns.tolist()

        missing_cols = [col for col in required_cols if col not in actual_columns]
        if missing_cols:
            print(f"错误：文件 '{os.path.basename(file_path)}' 缺少所需列: {missing_cols}。")
            print(f" (需要: {required_cols})")
            print(f" 实际找到的列名: {actual_columns}")
            return None
        return df

    except FileNotFoundError:
        print(f"错误：文件未找到 '{file_path}'")
        return None
    except ImportError as e:
        if 'openpyxl' in str(e):
            print("\n错误：读取 .xlsx 文件需要 'openpyxl' 库。请运行: pip install openpyxl")
        else:
            print(f"\n错误：导入错误: {e}")
        return None
    except Exception as e:
        print(f"错误：尝试读取 '{os.path.basename(file_path)}' 时失败: {type(e).__name__} - {e}")
        if "No engine for file type" in str(e):
            print(" 提示: 请确保已安装 'openpyxl' (用于 .xlsx) 或 'xlrd' (用于旧版 .xls)。")
        return None


def plot_confusion_matrix(cm, classes, random_seed, accuracy):
    """
    绘制混淆矩阵并显示随机种子参数。
    增大字体大小，确保中文正常显示。
    """
    # 创建更大的图形
    plt.figure(figsize=(12, 10))

    # 设置全局字体大小
    plt.rcParams.update({
        'font.size': 16,  # 基础字体大小
        'axes.titlesize': 20,  # 标题字体大小
        'axes.labelsize': 18,  # 坐标轴标签字体大小
        'xtick.labelsize': 16,  # x轴刻度标签大小
        'ytick.labelsize': 16,  # y轴刻度标签大小
    })

    # 自定义颜色映射，使对比度更高
    cmap = sns.color_palette("Blues", as_cmap=True)

    # 创建混淆矩阵热图，增大注释字体
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap=cmap,
        xticklabels=classes,
        yticklabels=classes,
        annot_kws={"size": 20}  # 增大矩阵中数字的字体大小
    )

    # 设置标题和轴标签
    plt.title(f'混淆矩阵 (Random Seed: {random_seed}, Accuracy: {accuracy:.4f})',
              fontsize=22, pad=20)
    plt.ylabel('真实类别', fontsize=20, labelpad=15)
    plt.xlabel('预测类别', fontsize=20, labelpad=15)

    # 调整刻度标签位置，以便更清晰地显示
    ax.set_xticklabels(classes, rotation=0, ha='center')
    ax.set_yticklabels(classes, rotation=0, va='center')

    # 给图添加边框
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1)

    # 确保图像不会被裁剪
    plt.tight_layout()

    # 使用高DPI保存图像以确保清晰度
    plt.savefig(os.path.join(output_folder, 'confusion_matrix.png'),
                dpi=300, bbox_inches='tight')
    plt.show()


def export_model(model, feature_names):
    """
    将模型导出为pkl, cbm和cpp格式。
    """
    # 保存为pickle格式
    with open(os.path.join(output_folder, r'E:\yjsproject\插烟状态识别\model\插拔电容模型.pkl'), 'wb') as f:
        pickle.dump(model, f)

    # 保存为CatBoost的原生格式
    model.save_model(os.path.join(output_folder, r'E:\yjsproject\插烟状态识别\model\插拔电容模型.cbm'))

    # 导出为C++代码
    model.save_model(
        os.path.join(output_folder, r'E:\yjsproject\插烟状态识别\model\model.cpp'),
        format="cpp",
        export_parameters=True,
        pool=None
    )

    # 保存特征名称列表，便于后续使用
    with open(os.path.join(output_folder, r'E:\yjsproject\插烟状态识别\model\feature_names.txt'), 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")

    print(f"\n模型已导出到 '{output_folder}' 文件夹:")
    print(f" - model.pkl (Python pickle格式)")
    print(f" - model.cbm (CatBoost原生格式)")
    print(f" - model.cpp (C++格式)")
    print(f" - feature_names.txt (特征名称列表)")


# --- 主逻辑 ---

# 1. 扫描文件夹, 提取特征并收集数据
all_features = []
all_labels = []
start_time = time.time()
print(f"开始处理文件夹: '{data_folder_path}'")
print(f"查找文件类型: {file_extensions}")
print(f"需要列: ['{CAP_COL_UPPER}', '{CAP_COL_LOWER}']")
print(f"分类模式: 多分类 ('烟支', '棉签', '镊子')")
print(f"随机种子: {RANDOM_SEED}")

if not os.path.isdir(data_folder_path):
    print(f"错误：文件夹路径 '{data_folder_path}' 不存在或无法访问。")
    sys.exit(1)

processed_files_count = 0
skipped_files_count = 0
required_columns = [CAP_COL_UPPER, CAP_COL_LOWER]

try:
    filenames = os.listdir(data_folder_path)
except OSError as e:
    print(f"错误：无法列出文件夹 '{data_folder_path}' 中的文件。错误: {e}")
    sys.exit(1)

print(f"在文件夹中找到 {len(filenames)} 个项目。开始处理...")

for filename in filenames:
    if not filename.lower().endswith(file_extensions):
        continue

    file_path = os.path.join(data_folder_path, filename)
    if not os.path.isfile(file_path):
        continue

    category = get_category_from_filename(filename)
    if category is None:
        skipped_files_count += 1
        continue

    df_current_file = try_read_excel_robust(file_path, required_columns)
    if df_current_file is None:
        skipped_files_count += 1
        continue

    features = extract_time_series_features(df_current_file, CAP_COL_UPPER, CAP_COL_LOWER)

    if features:
        all_features.append(features)
        all_labels.append(category)
        processed_files_count += 1
    else:
        skipped_files_count += 1

# --- 数据准备与检查 ---
loading_time = time.time() - start_time
print("\n--------------------")
print(f"文件处理完成 (耗时: {loading_time:.2f} 秒)。")
print(f"成功处理并提取特征的文件数: {processed_files_count}")
print(f"跳过的文件数（无类别/读取错误/数据不足）: {skipped_files_count}")

if not all_features:
    print("\n错误：未能从任何文件中成功提取有效特征。请检查相关配置和文件内容。")
    sys.exit(1)

feature_df = pd.DataFrame(all_features)
labels = np.array(all_labels)

print(f"\n总共获得 {len(feature_df)} 个样本（每个文件一个样本）。")
print(f"每个样本包含 {len(feature_df.columns)} 个特征。")

unique_labels, counts = np.unique(labels, return_counts=True)
print("\n类别分布:")
for label, count in zip(unique_labels, counts):
    print(f" - {label}: {count} 个样本")

if len(unique_labels) < 2:
    print("\n错误：数据中只包含一个类别或没有有效类别，无法进行分类训练。")
    sys.exit(1)
if feature_df.empty:
    print("\n错误: 特征 DataFrame 为空。")
    sys.exit(1)

# --- 模型训练与评估 ---

# 2. 准备训练/测试数据
feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
if feature_df.isnull().values.any():
    print("\n警告：特征数据中发现 NaN 值，将使用列均值填充。")
    feature_df.fillna(feature_df.mean(), inplace=True)
    if feature_df.isnull().values.any():
        print("\n错误：填充后仍有 NaN 值（可能某特征列在所有文件中都无效），无法继续。")
        print("请检查原始数据或特征提取逻辑。有问题的列：")
        print(feature_df.isnull().sum()[feature_df.isnull().sum() > 0])
        sys.exit(1)

X = feature_df
y = labels

# CatBoost 通常不需要显式缩放数值特征
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y)

print(f"\n数据分割完成：训练集 {len(X_train)} 样本, 测试集 {len(X_test)} 样本。")
print(f"使用的特征: {list(X.columns)}")

# --- 使用 CatBoostClassifier ---
print("\n开始训练 CatBoostClassifier 模型...")
train_start_time = time.time()

# 配置 CatBoost 模型参数
model = CatBoostClassifier(iterations=300,
                           learning_rate=0.1,
                           loss_function='MultiClass',
                           eval_metric='Accuracy',
                           auto_class_weights='Balanced',  # 处理类别不平衡
                           random_seed=RANDOM_SEED,
                           verbose=100,
                           early_stopping_rounds=50
                           )

# 训练模型，使用测试集作为验证集进行早停
model.fit(X_train, y_train,
          eval_set=(X_test, y_test),
          plot=False
          )

train_time = time.time() - train_start_time
print(f"模型训练完成 (耗时: {train_time:.2f} 秒)。")
print(f"模型实际迭代次数: {model.get_best_iteration()}")

# 4. 评估模型
print("\n开始评估模型...")
y_pred = model.predict(X_test)

# 确保预测结果格式正确
if isinstance(y_pred[0], (list, np.ndarray)) and len(y_pred[0]) == 1:
    y_pred = [item[0] for item in y_pred]

accuracy = accuracy_score(y_test, y_pred)
print(f"\n模型准确率: {accuracy:.4f}")

print("\n分类报告:")
report_labels = sorted(list(set(y_train) | set(y_test)))
print(classification_report(y_test, y_pred, labels=report_labels, zero_division=0))

print("\n混淆矩阵:")
print(f"(行: 真实类别, 列: 预测类别)")
cm = confusion_matrix(y_test, y_pred, labels=report_labels)
cm_df = pd.DataFrame(cm, index=report_labels, columns=report_labels)
print(cm_df)

# 绘制混淆矩阵
plot_confusion_matrix(cm, report_labels, RANDOM_SEED, accuracy)

# 5. 查看特征重要性 (CatBoost 的方法)
try:
    feature_importances = pd.DataFrame({'feature': X.columns,
                                        'importance': model.get_feature_importance()})
    feature_importances = feature_importances.sort_values('importance', ascending=False).reset_index(drop=True)
    print("\n特征重要性 (Top 15):")
    print(feature_importances.head(15))
except Exception as e:
    print(f"\n无法获取特征重要性: {e}")

# 6. 导出模型
export_model(model, X.columns.tolist())

total_time = time.time() - start_time
print(f"\n脚本总执行完毕 (总耗时: {total_time:.2f} 秒)。")