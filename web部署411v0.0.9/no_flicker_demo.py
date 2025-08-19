import streamlit as st
import time

st.set_page_config(page_title="无闪烁界面演示", layout="wide")

st.title("🚀 无闪烁界面演示")

st.markdown("""
这个演示展示了如何实现真正无闪烁的Streamlit界面。
当你切换模型选择时，其他输入框不会重置或闪烁。
""")

# 模拟可用模型列表
available_models = ['linear_regression', 'random_forest', 'xgboost', 'neural_network', 'svm']
model_display_names = {
    'linear_regression': '线性回归',
    'random_forest': '随机森林', 
    'xgboost': 'XGBoost',
    'neural_network': '神经网络',
    'svm': '支持向量机'
}

st.markdown("---")

# 创建两个对比示例
col1, col2 = st.columns(2)

with col1:
    st.subheader("❌ 传统方式 (会闪烁)")
    
    # 传统的实现方式
    selected_model_old = st.selectbox(
        "选择模型 (传统方式)",
        available_models,
        format_func=lambda x: model_display_names[x],
        key="old_selectbox"
    )
    
    # 这些输入框会在模型切换时重置
    model_name_old = st.text_input(
        "模型名称 (传统方式)", 
        value=selected_model_old,
        key="old_name"
    )
    
    version_old = st.text_input(
        "版本号 (传统方式)", 
        value="1.0",
        key="old_version"
    )
    
    description_old = st.text_area(
        "描述 (传统方式)", 
        value="这是一个测试模型",
        key="old_desc"
    )

with col2:
    st.subheader("✅ 无闪烁方式 (推荐)")
    
    # 无闪烁的实现方式
    # 初始化持久状态
    if 'demo_selected_model' not in st.session_state:
        st.session_state.demo_selected_model = available_models[0]
    if 'demo_model_name' not in st.session_state:
        st.session_state.demo_model_name = available_models[0]
    if 'demo_version' not in st.session_state:
        st.session_state.demo_version = "1.0"
    if 'demo_description' not in st.session_state:
        st.session_state.demo_description = "这是一个测试模型"
    
    # 确保选择的模型在可用列表中
    if st.session_state.demo_selected_model not in available_models:
        st.session_state.demo_selected_model = available_models[0]
    
    # 使用容器避免闪烁
    demo_container = st.container()
    
    with demo_container:
        # 模型选择
        current_index = available_models.index(st.session_state.demo_selected_model)
        
        selected_model_new = st.selectbox(
            "选择模型 (无闪烁方式)",
            available_models,
            index=current_index,
            format_func=lambda x: model_display_names[x],
            key="new_selectbox"
        )
        
        # 只在真正变化时更新状态
        if selected_model_new != st.session_state.demo_selected_model:
            st.session_state.demo_selected_model = selected_model_new
            # 智能更新模型名称：只在用户没有自定义修改时才更新
            if st.session_state.demo_model_name in available_models:
                st.session_state.demo_model_name = selected_model_new
        
        # 使用持久状态的输入框
        model_name_new = st.text_input(
            "模型名称 (无闪烁方式)", 
            value=st.session_state.demo_model_name,
            key="new_name"
        )
        if model_name_new != st.session_state.demo_model_name:
            st.session_state.demo_model_name = model_name_new
        
        version_new = st.text_input(
            "版本号 (无闪烁方式)", 
            value=st.session_state.demo_version,
            key="new_version"
        )
        if version_new != st.session_state.demo_version:
            st.session_state.demo_version = version_new
        
        description_new = st.text_area(
            "描述 (无闪烁方式)", 
            value=st.session_state.demo_description,
            key="new_desc"
        )
        if description_new != st.session_state.demo_description:
            st.session_state.demo_description = description_new

st.markdown("---")

# 显示当前状态
st.subheader("📊 当前状态对比")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**传统方式状态:**")
    st.json({
        "选择的模型": model_display_names[selected_model_old],
        "模型名称": model_name_old,
        "版本号": version_old,
        "描述": description_old[:50] + "..." if len(description_old) > 50 else description_old
    })

with col2:
    st.markdown("**无闪烁方式状态:**")
    st.json({
        "选择的模型": model_display_names[st.session_state.demo_selected_model],
        "模型名称": st.session_state.demo_model_name,
        "版本号": st.session_state.demo_version,
        "描述": st.session_state.demo_description[:50] + "..." if len(st.session_state.demo_description) > 50 else st.session_state.demo_description
    })

st.markdown("---")

# 使用说明
st.subheader("📝 使用说明")

st.markdown("""
### 测试步骤：

1. **在左侧 (传统方式)**：
   - 修改模型名称、版本号或描述
   - 然后切换模型选择
   - 观察：你的输入会被重置！😱

2. **在右侧 (无闪烁方式)**：
   - 修改模型名称、版本号或描述  
   - 然后切换模型选择
   - 观察：你的输入保持不变！🎉

### 技术要点：

- ✅ 使用 `st.session_state` 保持组件状态
- ✅ 使用 `st.container()` 避免重新渲染
- ✅ 智能状态更新：只在真正变化时更新
- ✅ 固定的组件 `key` 避免重复创建
- ✅ 状态同步：手动同步组件值和session state

### 关键代码模式：

```python
# 1. 初始化持久状态
if 'my_value' not in st.session_state:
    st.session_state.my_value = default_value

# 2. 使用容器
with st.container():
    # 3. 创建组件时使用当前状态值
    current_value = st.text_input(
        "标签",
        value=st.session_state.my_value,
        key="unique_key"
    )
    
    # 4. 手动同步状态
    if current_value != st.session_state.my_value:
        st.session_state.my_value = current_value
```

这种方法确保了用户在切换选择时不会丢失任何输入，提供了流畅的用户体验。
""")

# 重置按钮
if st.button("🔄 重置演示", help="重置所有状态到初始值"):
    # 清除session state
    keys_to_clear = [
        'demo_selected_model', 'demo_model_name', 
        'demo_version', 'demo_description'
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.success("演示已重置！")
    time.sleep(0.5)
    st.experimental_rerun()
