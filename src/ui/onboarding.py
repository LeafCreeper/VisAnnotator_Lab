import streamlit as st

@st.dialog("欢迎使用 VisAnnotator Lab! 👋")
def show_onboarding():
    st.markdown("""
    **VisAnnotator Lab** 是专为社会科学研究设计的 LLM 标注实验台。
    
    ### 快速上手指南:
    
    1.  **📂 数据接入 (Data)**
        *   上传您的 CSV 或 Excel 数据文件。
        *   系统会自动预览数据。
        
    2.  **📝 定义与调试 (Define)**
        *   **Schema**: 定义您想让 AI 提取的变量（如情感、分类）。
        *   **Prompt**: 编写提示词，使用 `{{列名}}` 插入数据。
        *   **多配置**: 点击右上角 `+` 号，创建多个 Prompt 版本进行对比。
        
    3.  **🧪 标注执行台 (Runner)**
        *   **调试模式**: 先跑几条试试效果。
        *   **生产模式**: 效果满意后，一键全量运行并下载结果。
        
    4.  **📈 评估与信度 (Evaluation)**
        *   **验证集**: 构建一个小样本（或导入现有标注）。
        *   **对比实验**: 比较不同 Prompt 组合的准确率与一致性 (Kappa)。
    """)
    
    if st.button("我已了解，开始使用 (Get Started)", type="primary"):
        st.session_state.has_seen_tutorial = True
        st.rerun()
