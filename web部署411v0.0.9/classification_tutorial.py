# -*- coding: utf-8 -*-
"""
Classification Tutorial Module for Streamlit App
Provides an interactive interface to demonstrate classification algorithms.
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import traceback

# --- 尝试从主分类模块导入绘图函数 ---
# 确保 classification_training.py 在同一目录或 Python 路径中
try:
    # 导入您在 classification_training.py 中定义的绘图函数
    from classification_training import (
        plot_confusion_matrix,
        plot_roc_curve, # 确保这个函数存在且接受合适的参数
        apply_plot_style,
        create_figure_with_safe_dimensions,
        FONT_PROP # 导入字体设置
    )
    CLASSIFICATION_MODULE_AVAILABLE = True
    print("成功从 classification_training.py 导入绘图函数。")
except ImportError:
    CLASSIFICATION_MODULE_AVAILABLE = False
    # 如果导入失败，定义占位函数
    def plot_confusion_matrix(*args, **kwargs):
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "错误: 绘图函数\nplot_confusion_matrix\n无法从 classification_training.py 导入", ha='center', va='center', color='red', fontsize=9)
        return fig
    def plot_roc_curve(*args, **kwargs):
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "错误: 绘图函数\nplot_roc_curve\n无法从 classification_training.py 导入", ha='center', va='center', color='red', fontsize=9)
        return fig
    def apply_plot_style(ax): return ax
    def create_figure_with_safe_dimensions(w, h, dpi=80): return plt.subplots(figsize=(w,h), dpi=dpi)
    FONT_PROP = None
    # 不再在模块内部显示错误，让主应用处理
    # st.error("错误：无法导入主分类模块 (classification_training.py) 中的绘图函数。教学演示的可视化功能将受限。")
    print("错误：无法导入主分类模块 (classification_training.py) 中的绘图函数。教学演示的可视化功能将受限。")


# --- 教学状态初始化 ---
def initialize_tutorial_state():
    """初始化分类教学模块专用的会话状态变量"""
    defaults = {
        'cls_tut_dataset_name': 'Synthetic', 'cls_tut_n_samples': 200, 'cls_tut_n_features': 2,
        'cls_tut_n_informative': 2, 'cls_tut_n_redundant': 0, 'cls_tut_n_clusters_per_class': 1,
        'cls_tut_class_sep': 1.0, 'cls_tut_flip_y': 0.01,
        'cls_tut_method': 'Logistic Regression',
        'cls_tut_logreg_c': 1.0,
        'cls_tut_svm_c': 1.0, 'cls_tut_svm_kernel': 'rbf',
        'cls_tut_rf_n_estimators': 100, 'cls_tut_rf_max_depth': None,
        'cls_tut_data_X_raw': None, # 原始特征数据
        'cls_tut_data_X': None, # 标准化后的特征数据
        'cls_tut_data_y': None, # 真实标签
        'cls_tut_X_train': None, 'cls_tut_X_test': None,
        'cls_tut_y_train': None, 'cls_tut_y_test': None,
        'cls_tut_model': None, # 训练好的模型
        'cls_tut_y_pred': None, # 测试集预测标签
        'cls_tut_y_proba': None, # 测试集预测概率 (用于ROC)
        'cls_tut_scaler': StandardScaler(), # 标准化器
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- 教学 UI 函数 ---
def show_tutorial_page():
    """创建交互式分类教学演示的用户界面"""
    initialize_tutorial_state()

    st.header("🎓 分类教学演示")
    st.markdown("""
    欢迎来到分类教学模块！在这里，你可以：
    1.  选择不同的 **示例数据集** 或生成 **合成数据**。
    2.  调整生成合成数据的 **参数**。
    3.  选择 **分类算法**（如逻辑回归, 支持向量机, 随机森林）并调整其关键参数。
    4.  **训练模型** 并 **可视化** 结果（如混淆矩阵、ROC曲线）。
    5.  查看 **评估指标**（如准确率）和结果解读。

    通过互动操作，直观理解不同分类算法的决策边界、参数影响以及评估方法。
    """)
    st.markdown("---")

    # --- 1. 选择示例数据集 ---
    st.subheader("1. 选择示例数据集")
    dataset_options = {
        "Synthetic": "生成可控的合成二分类数据",
        "Iris": "经典的鸢尾花数据集 (3类)",
        "Wine": "葡萄酒数据集 (3类)",
        "Breast Cancer": "乳腺癌数据集 (2类)"
    }
    # 修正：移除赋值
    st.selectbox(
        "选择数据集类型:",
        options=list(dataset_options.keys()),
        key="cls_tut_dataset_select",
        help="选择一个内置的数据集或生成器来创建演示数据。"
    )

    # --- 2. 数据集参数 (仅对 Synthetic 数据集) ---
    if st.session_state.cls_tut_dataset_name == "Synthetic":
        st.subheader("2. 调整合成数据集参数")
        col_data1, col_data2 = st.columns(2)
        with col_data1:
            # 修正：移除赋值
            st.slider(
                "样本数量:", min_value=100, max_value=1000,
                value=st.session_state.cls_tut_n_samples, step=50, key="cls_tut_samples",
                help="生成数据点的总数。"
            )
            st.slider(
                "类别区分度 (class_sep):", min_value=0.1, max_value=2.0,
                value=st.session_state.cls_tut_class_sep,  # 读取 session_state 作为默认值
                step=0.1, format="%.1f", key="cls_tut_class_sep",  # 设置 key
                help="控制类别之间的分离程度，值越大越容易区分。"
            )
            st.slider(
                "信息特征数 (n_informative):", min_value=1, max_value=2,
                value=st.session_state.cls_tut_n_informative,  # 读取 session_state 作为默认值
                step=1, key="cls_tut_n_informative",  # 设置 key
                help="真正包含类别信息的特征数量。"
            )

        with col_data2:
            st.text_input("特征数量 (固定为2D):", value="2", key="cls_tut_features_display", disabled=True)
            # st.session_state.cls_tut_n_redundant = 0 # 这些不需要用户输入，直接在生成时使用固定值
            # st.session_state.cls_tut_n_clusters_per_class = 1

            st.slider(
                "标签噪声比例 (flip_y):", min_value=0.00, max_value=0.20,
                value=st.session_state.cls_tut_flip_y,  # 读取 session_state 作为默认值
                step=0.01, format="%.2f", key="cls_tut_flip_y",  # 设置 key
                help="随机翻转部分样本标签的比例，模拟噪声。"
            )
    else:
        st.subheader("2. 数据集信息")
        # ... (数据集信息描述保持不变) ...
        if st.session_state.cls_tut_dataset_name == "Iris":
            st.markdown("- **鸢尾花 (Iris):** 包含3个类别 (Setosa, Versicolour, Virginica)，每个类别50个样本，4个特征 (萼片长度/宽度, 花瓣长度/宽度)。我们将只使用前两个特征进行2D可视化。")
        elif st.session_state.cls_tut_dataset_name == "Wine":
            st.markdown("- **葡萄酒 (Wine):** 包含3个类别，共178个样本，13个特征。我们将只使用前两个特征进行2D可视化。")
        elif st.session_state.cls_tut_dataset_name == "Breast Cancer":
            st.markdown("- **乳腺癌 (Breast Cancer):** 包含2个类别 (Malignant, Benign)，共569个样本，30个特征。我们将只使用前两个特征进行2D可视化。")


    # --- 生成/加载数据集按钮 ---
    if st.button("🔄 加载/生成数据集", key="cls_tut_generate_data"):
        X_raw, y_true = None, None
        with st.spinner("正在准备数据..."):
            try:
                random_state_data = 42
                if st.session_state.cls_tut_dataset_name == "Synthetic":
                    # 在生成时直接读取 session state 中的值
                    X_raw, y_true = make_classification(
                        n_samples=st.session_state.cls_tut_n_samples,
                        n_features=2, # 固定为2
                        n_informative=st.session_state.cls_tut_n_informative,
                        n_redundant=0, # 固定
                        n_clusters_per_class=1, # 固定
                        class_sep=st.session_state.cls_tut_class_sep,
                        flip_y=st.session_state.cls_tut_flip_y,
                        random_state=random_state_data
                    )
                elif st.session_state.cls_tut_dataset_name == "Iris":
                    iris = load_iris()
                    X_raw, y_true = iris.data[:, :2], iris.target
                elif st.session_state.cls_tut_dataset_name == "Wine":
                    wine = load_wine()
                    X_raw, y_true = wine.data[:, :2], wine.target
                elif st.session_state.cls_tut_dataset_name == "Breast Cancer":
                    cancer = load_breast_cancer()
                    X_raw, y_true = cancer.data[:, :2], cancer.target

                if X_raw is not None:
                    st.session_state.cls_tut_data_X_raw = X_raw
                    st.session_state.cls_tut_data_X = st.session_state.cls_tut_scaler.fit_transform(X_raw)
                    st.session_state.cls_tut_data_y = y_true

                    st.session_state.cls_tut_X_train, st.session_state.cls_tut_X_test, \
                    st.session_state.cls_tut_y_train, st.session_state.cls_tut_y_test = train_test_split(
                        st.session_state.cls_tut_data_X, st.session_state.cls_tut_data_y,
                        test_size=0.3, random_state=random_state_data, stratify=y_true
                    )
                    st.session_state.cls_tut_model = None
                    st.session_state.cls_tut_y_pred = None
                    st.session_state.cls_tut_y_proba = None
                    st.success(f"数据集 '{st.session_state.cls_tut_dataset_name}' 已准备就绪并分割为训练/测试集。")
                else:
                     st.error("无法加载或生成所选数据集。")

            except Exception as data_err:
                st.error(f"准备数据集时出错: {data_err}")
                print(traceback.format_exc())
                st.session_state.cls_tut_data_X = None

    # --- 显示准备好的数据集 ---
    if st.session_state.cls_tut_data_X is not None:
        st.write("---")
        st.markdown("#### 数据集预览（已标准化，按真实类别着色）")
        if CLASSIFICATION_MODULE_AVAILABLE:
            try:
                fig_data, ax_data = create_figure_with_safe_dimensions(8, 5)
                n_classes = len(np.unique(st.session_state.cls_tut_data_y))
                 # 确保 n_classes 大于 0
                if n_classes > 0:
                    cmap = plt.cm.get_cmap('viridis', n_classes)
                    scatter = ax_data.scatter(
                        st.session_state.cls_tut_data_X[:, 0],
                        st.session_state.cls_tut_data_X[:, 1],
                        c=st.session_state.cls_tut_data_y,
                        cmap=cmap,
                        s=30, alpha=0.7, edgecolors='k', linewidth=0.5
                    )
                    apply_plot_style(ax_data)
                    title_str = f"数据集: {st.session_state.cls_tut_dataset_name} (真实类别)"
                    ax_data.set_title(title_str, fontproperties=FONT_PROP if FONT_PROP else None)
                    ax_data.set_xlabel("特征 1 (标准化后)", fontproperties=FONT_PROP if FONT_PROP else None)
                    ax_data.set_ylabel("特征 2 (标准化后)", fontproperties=FONT_PROP if FONT_PROP else None)

                    handles, labels = scatter.legend_elements()
                    legend_labels = [f"类别 {i}" for i in range(n_classes)]
                    ax_data.legend(handles, legend_labels, title="类别", prop=FONT_PROP)
                    st.pyplot(fig_data)
                else:
                    st.warning("数据集中未找到有效类别，无法绘图。")

            except Exception as plot_err:
                st.warning(f"绘制数据集图表时出错: {plot_err}")
                print(traceback.format_exc())
        else:
             st.warning("无法显示数据集图表，因为绘图函数导入失败。")
    else:
        st.info("请点击 **“🔄 加载/生成数据集”** 按钮来创建数据。")
        return

    st.markdown("---")

    # --- 3. 选择分类方法与参数 ---
    st.subheader("3. 选择分类方法与参数")
    cls_tut_method_options = ["Logistic Regression", "SVM (支持向量机)", "Random Forest (随机森林)"]
    # 修正：移除赋值
    st.selectbox(
        "选择分类算法:",
        options=cls_tut_method_options,
        key="cls_tut_method_select",
        help="选择要应用于上方数据的分类算法。"
    )

    # 显示参数设置
    if st.session_state.cls_tut_method == "Logistic Regression":
        # 修正：移除赋值
        st.slider(
            "正则化强度 C:", min_value=0.01, max_value=10.0,
            value=st.session_state.cls_tut_logreg_c, step=0.1, format="%.2f", key="cls_tut_logreg_slider",
            help="C 值越小，正则化越强（模型更简单，可能欠拟合）；C 值越大，正则化越弱（模型更复杂，可能过拟合）。"
        )
        st.markdown("**算法说明:** 逻辑回归是一种线性模型，通过 Sigmoid 函数将线性输出映射到 (0, 1) 区间，用于估计概率。常用于二分类问题，也可扩展到多分类。")
        st.markdown(f"**参数影响:** 主要参数是正则化强度 C。调整 C 值观察决策边界的变化。")
    elif st.session_state.cls_tut_method == "SVM (支持向量机)":
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            # 修正：移除赋值
            st.slider(
                "正则化强度 C:", min_value=0.1, max_value=10.0,
                value=st.session_state.cls_tut_svm_c, step=0.1, format="%.1f", key="cls_tut_svm_slider",
                help="控制分类错误的惩罚程度。C 值越大，模型越倾向于正确分类所有点，可能导致过拟合。"
            )
        with col_p2:
            # 修正：移除赋值
            st.selectbox(
                "核函数 (kernel):", options=['rbf', 'linear', 'poly'],
                key="cls_tut_svm_kernel_select",
                help="'rbf' (径向基函数) 和 'poly' (多项式) 可以处理非线性问题，'linear' (线性) 只能处理线性可分问题。"
            )
        st.markdown("**算法说明:** 支持向量机 (SVM) 寻找一个最优超平面来最大化不同类别之间的间隔。通过核函数（如 RBF、多项式），SVM 可以有效地处理非线性可分的数据。")
        st.markdown(f"**参数影响:** 调整 **C** 和 **kernel** 观察决策边界的变化。线性核产生直线边界，RBF 核可以产生复杂的非线性边界。")
    elif st.session_state.cls_tut_method == "Random Forest (随机森林)":
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            # 修正：移除赋值
            st.slider(
                "树的数量 (n_estimators):", min_value=10, max_value=200,
                value=st.session_state.cls_tut_rf_n_estimators, step=10, key="cls_tut_rf_est_slider",
                help="森林中决策树的数量。数量越多，模型通常越稳定，但训练时间越长。"
            )
        with col_p2:
             # 这里不直接赋值给 session_state，而是先赋给临时变量
             rf_max_depth_val = st.slider(
                "树的最大深度 (max_depth, 0表示无限制):", min_value=0, max_value=20,
                value=(st.session_state.cls_tut_rf_max_depth if st.session_state.cls_tut_rf_max_depth is not None else 0),
                step=1, key="cls_tut_rf_depth_slider", # key 仍然更新 session_state
                help="限制单棵决策树的最大深度，有助于防止过拟合。0表示不限制。"
             )
             # 在需要使用时，从 session_state 读取，或者根据临时变量更新逻辑
             # 例如，在训练时：max_depth = None if st.session_state.cls_tut_rf_depth_slider == 0 else st.session_state.cls_tut_rf_depth_slider

        st.markdown("**算法说明:** 随机森林是一种集成学习方法，它构建多棵决策树，并通过投票（分类）或平均（回归）来得出最终预测结果。它通常具有较高的准确性和鲁棒性。")
        st.markdown("**参数影响:** 调整 **n_estimators** 和 **max_depth** 观察对决策边界复杂度和模型性能的影响。")

    # --- 训练模型按钮 ---
    if st.button("🧠 训练并评估模型", key="cls_tut_run_training", help="使用当前选择的算法和参数训练分类模型，并在测试集上评估"):
        if st.session_state.cls_tut_X_train is None:
             st.error("请先准备数据集！")
        else:
            X_train_tut = st.session_state.cls_tut_X_train
            y_train_tut = st.session_state.cls_tut_y_train
            X_test_tut = st.session_state.cls_tut_X_test
            y_test_tut = st.session_state.cls_tut_y_test
            method_tut = st.session_state.cls_tut_method
            model_tut = None
            success_flag = False
            try:
                with st.spinner(f"正在训练 {method_tut}..."):
                    # 在这里读取 session_state 中的参数值
                    if method_tut == "Logistic Regression":
                        model_tut = LogisticRegression(C=st.session_state.cls_tut_logreg_c, random_state=42)
                    elif method_tut == "SVM (支持向量机)":
                        model_tut = SVC(C=st.session_state.cls_tut_svm_c,
                                       kernel=st.session_state.cls_tut_svm_kernel,
                                       probability=True,
                                       random_state=42)
                    elif method_tut == "Random Forest (随机森林)":
                         # 读取 max_depth 并处理 None 的情况
                         rf_depth = st.session_state.get("cls_tut_rf_depth_slider", 0) # 使用 get 获取，提供默认值
                         max_depth_param = None if rf_depth == 0 else rf_depth
                         model_tut = RandomForestClassifier(
                            n_estimators=st.session_state.cls_tut_rf_n_estimators,
                            max_depth=max_depth_param, # 使用处理后的值
                            random_state=42, n_jobs=-1)

                    if model_tut:
                         model_tut.fit(X_train_tut, y_train_tut)
                         st.session_state.cls_tut_model = model_tut
                         st.session_state.cls_tut_y_pred = model_tut.predict(X_test_tut)
                         if hasattr(model_tut, "predict_proba"):
                              st.session_state.cls_tut_y_proba = model_tut.predict_proba(X_test_tut)
                         else:
                              st.session_state.cls_tut_y_proba = None
                         success_flag = True

                if success_flag:
                    st.success(f"{method_tut} 模型训练完成！请查看下方评估结果。")
                else:
                    st.error("未能初始化所选模型。")

            except Exception as train_e:
                st.error(f"训练 {method_tut} 出错: {train_e}")
                print(traceback.format_exc())
                st.session_state.cls_tut_model = None
                st.session_state.cls_tut_y_pred = None
                st.session_state.cls_tut_y_proba = None

    st.markdown("---")

    # --- 4. 显示评估结果 ---
    if st.session_state.cls_tut_model is not None and st.session_state.cls_tut_y_pred is not None:
        st.subheader("4. 模型评估结果（基于测试集）")

        y_test = st.session_state.cls_tut_y_test
        y_pred = st.session_state.cls_tut_y_pred
        y_proba = st.session_state.cls_tut_y_proba
        class_names = None
        try:
            if hasattr(st.session_state.cls_tut_model, 'classes_'):
                class_names = [str(c) for c in st.session_state.cls_tut_model.classes_]
            elif st.session_state.cls_tut_y_train is not None:
                class_names = [str(c) for c in sorted(np.unique(st.session_state.cls_tut_y_train))]
        except: pass

        try:
            accuracy = accuracy_score(y_test, y_pred)
            st.metric("准确率 (Accuracy)", f"{accuracy:.3f}",
                      help="模型正确分类的样本比例。简单直观，但在类别不平衡时可能具有误导性。")
        except Exception as acc_e:
            st.error(f"计算准确率时出错: {acc_e}")

        if CLASSIFICATION_MODULE_AVAILABLE:
            col_viz1, col_viz2 = st.columns(2)
            with col_viz1:
                st.markdown("##### 混淆矩阵")
                st.markdown("显示了模型预测的正确和错误情况。行代表真实类别，列代表预测类别。")
                try:
                    fig_cm = plot_confusion_matrix(y_test, y_pred, class_names=class_names)
                    st.pyplot(fig_cm)
                except Exception as cm_err:
                     st.warning(f"绘制混淆矩阵时出错: {cm_err}")
                     print(traceback.format_exc())
            with col_viz2:
                st.markdown("##### ROC 曲线 (仅适用于二分类或多分类OvR)")
                st.markdown("衡量分类器在不同阈值下的性能。曲线下面积 (AUC) 越大，模型区分能力越好 (接近1为优)。")
                if y_proba is not None:
                    try:
                        fig_roc = plot_roc_curve(y_test, y_proba, class_names=class_names)
                        st.pyplot(fig_roc)
                    except Exception as roc_err:
                        st.warning(f"绘制 ROC 曲线时出错: {roc_err}")
                        print(traceback.format_exc())
                else:
                    st.info("当前模型不支持概率预测，无法绘制 ROC 曲线。")
        else:
             st.warning("无法显示可视化结果，因为绘图函数导入失败。")

        st.markdown("#### 结果解读提示")
        st.markdown("- **准确率:** 越高越好，但要注意类别是否平衡。")
        st.markdown("- **混淆矩阵:**")
        st.markdown("  - **对角线** 上的值表示 **正确分类** 的样本数。")
        st.markdown("  - **非对角线** 上的值表示 **错误分类** 的样本数。例如，第 i 行第 j 列的值表示真实类别为 i 但被错误预测为 j 的样本数。")
        st.markdown("  - 观察哪些类别之间容易混淆。")
        st.markdown("- **ROC 曲线与 AUC:**")
        st.markdown("  - 曲线越 **靠近左上角**，表示模型性能越好。")
        st.markdown("  - **AUC 值** 越接近 1，表示模型区分正负样本（或各个类别）的能力越强。AUC 为 0.5 表示随机猜测。")

# --- 允许直接运行此脚本进行测试 ---
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="分类教学演示（独立运行）")
    st.sidebar.info("这是分类教学模块的独立测试运行。")
    show_tutorial_page()