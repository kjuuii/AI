# test_fonts.py - 字体诊断和测试页面
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os
import glob
import subprocess

st.set_page_config(page_title="字体诊断测试", page_icon="🔍", layout="wide")

st.title("🔍 中文字体诊断测试")

# 导入并运行字体设置
from font_utils import setup_chinese_font, debug_font_info

# 设置字体
with st.spinner("正在设置字体..."):
    font_loaded = setup_chinese_font()

if font_loaded:
    st.success("✅ 字体设置完成")
else:
    st.warning("⚠️ 字体设置可能未完全成功")

# 显示调试信息
with st.expander("📋 查看详细诊断信息", expanded=True):
    st.code(debug_font_info())

    # 显示所有 matplotlib 识别的字体
    st.subheader("所有可用字体")
    all_fonts = sorted(set([f.name for f in fm.fontManager.ttflist]))

    # 筛选可能的中文字体
    chinese_fonts = [f for f in all_fonts if any(k in f.lower() for k in ['noto', 'cjk', 'chinese', 'sc', 'cn'])]

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**可能的中文字体 ({len(chinese_fonts)} 个):**")
        for font in chinese_fonts[:20]:  # 只显示前20个
            st.write(f"• {font}")

    with col2:
        st.write(f"**所有字体 (共 {len(all_fonts)} 个):**")
        for font in all_fonts[:20]:  # 只显示前20个
            st.write(f"• {font}")

# 测试matplotlib图表
st.subheader("📊 Matplotlib 中文显示测试")

col1, col2 = st.columns(2)

with col1:
    # 测试1：简单文本
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    test_texts = [
        "测试中文显示",
        "Test English",
        "中英混合 Mixed",
        "数字: 123.456",
        "符号: ±×÷≈≠",
        "负数: -789"
    ]

    for i, text in enumerate(test_texts):
        ax1.text(0.1, 0.9 - i * 0.15, text, fontsize=12, transform=ax1.transAxes)

    ax1.set_title("文本显示测试")
    ax1.set_xlabel("X轴标签")
    ax1.set_ylabel("Y轴标签")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    st.pyplot(fig1)
    plt.close()

with col2:
    # 测试2：带数据的图表
    fig2, ax2 = plt.subplots(figsize=(6, 4))

    categories = ['类别A', '类别B', '类别C', '类别D']
    values = [25, 40, 30, 35]

    bars = ax2.bar(categories, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax2.set_title("柱状图测试")
    ax2.set_xlabel("分类")
    ax2.set_ylabel("数值")

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height}',
                 ha='center', va='bottom')

    st.pyplot(fig2)
    plt.close()

# 测试3：复杂图表
st.subheader("📈 复杂图表测试")

fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 4))

# 左图：折线图
x = np.linspace(-5, 5, 100)
y1 = np.sin(x)
y2 = np.cos(x)

ax3.plot(x, y1, label='正弦曲线', linewidth=2)
ax3.plot(x, y2, label='余弦曲线', linewidth=2)
ax3.set_title("三角函数图")
ax3.set_xlabel("角度")
ax3.set_ylabel("数值")
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

# 右图：饼图
sizes = [30, 25, 20, 15, 10]
labels = ['数据处理', '模型训练', '特征工程', '模型评估', '部署']
colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']

ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
ax4.set_title("项目时间分配")

plt.tight_layout()
st.pyplot(fig3)
plt.close()

# 检查系统字体文件
st.subheader("🔍 系统字体文件检查")

font_paths = {
    '/usr/share/fonts/': '系统字体目录',
    '/usr/share/fonts/truetype/': 'TrueType字体',
    '/usr/share/fonts/opentype/': 'OpenType字体',
    '/usr/local/share/fonts/': '本地字体目录'
}

for path, desc in font_paths.items():
    if os.path.exists(path):
        # 查找 Noto 字体
        noto_files = glob.glob(os.path.join(path, '**/[Nn]oto*.ttf'), recursive=True)
        noto_files += glob.glob(os.path.join(path, '**/[Nn]oto*.otf'), recursive=True)

        if noto_files:
            st.write(f"**{desc} ({path}):**")
            st.write(f"找到 {len(noto_files)} 个 Noto 字体文件")
            # 显示前5个文件名
            for file in noto_files[:5]:
                st.write(f"  • {os.path.basename(file)}")
            if len(noto_files) > 5:
                st.write(f"  ... 还有 {len(noto_files) - 5} 个文件")

# 检查当前matplotlib配置
st.subheader("⚙️ Matplotlib 配置")

config_info = f"""
**当前配置:**
- font.family: {plt.rcParams['font.family']}
- font.sans-serif: {plt.rcParams['font.sans-serif'][:3]}...
- axes.unicode_minus: {plt.rcParams['axes.unicode_minus']}
- 缓存目录: {fm.get_cachedir()}
"""
st.markdown(config_info)

# 最终建议
st.subheader("💡 诊断结果")

if font_loaded and len(chinese_fonts) > 0:
    st.success("""
    ✅ **字体配置正常**
    - 系统已安装 Noto CJK 字体
    - Matplotlib 可以识别中文字体
    - 图表应该能正常显示中文
    """)
else:
    st.error("""
    ❌ **字体配置有问题**

    可能的解决方案：
    1. 确认 packages.txt 在 GitHub 仓库根目录
    2. 尝试重新部署应用（Reboot app）
    3. 如果问题持续，考虑使用英文标签作为临时方案
    """)

    # 提供快速修复代码
    st.code("""
# 临时解决方案：在你的代码中使用英文标签
plt.xlabel('Time')  # 而不是 '时间'
plt.ylabel('Value')  # 而不是 '数值'
plt.title('Data Analysis')  # 而不是 '数据分析'
    """, language='python')