import streamlit as st
import pandas as pd

def render_data_tab():
    st.header("数据上传 (Data Upload)")
    uploaded_file = st.file_uploader("上传 CSV 或 Excel 文件", type=["csv", "xlsx"])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.df = df
            st.success(f"成功加载 {len(df)} 行数据。")
            
            st.subheader("数据预览")
            st.dataframe(df.head(100), use_container_width=True)
            
        except Exception as e:
            st.error(f"加载文件出错: {e}")
    else:
        st.info("请上传文件以开始。")
        if st.button("加载演示数据"):
            data = {
                "text_id": [1, 2, 3, 4, 5],
                "content": [
                    "这个产品太棒了！我很喜欢。",
                    "糟糕的体验，不推荐。",
                    "感觉一般般，没什么特别的。",
                    "物流很快，但是东西坏了。",
                    "客服非常有帮助。"
                ],
                "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
            }
            st.session_state.df = pd.DataFrame(data)
            st.success("演示数据加载成功！请前往“提示词与Schema”标签页查看。")
            st.rerun()
