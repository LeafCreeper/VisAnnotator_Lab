import streamlit as st
import json

def render_sidebar():
    with st.sidebar:
        st.title("🧪 VisAnnotator Lab")
        st.markdown("---")
        
        # --- Config Import/Export ---
        st.header("配置管理")
        
        # Export
        current_config = {
            "schema_fields": st.session_state.get("schema_fields", []),
            "system_prompt": st.session_state.get("system_prompt", ""),
            "user_prompt_template": st.session_state.get("user_prompt_template", ""),
            # We don't save API Key for security, or maybe optional? Better not to.
        }
        
        st.download_button(
            label="📤 导出配置 (JSON)",
            data=json.dumps(current_config, indent=4, ensure_ascii=False),
            file_name="visannotator_config.json",
            mime="application/json"
        )
        
        # Import
        uploaded_config = st.file_uploader("📥 导入配置", type=["json"])
        if uploaded_config is not None:
            try:
                loaded_conf = json.load(uploaded_config)
                # Update Session State
                if "schema_fields" in loaded_conf:
                    st.session_state.schema_fields = loaded_conf["schema_fields"]
                if "system_prompt" in loaded_conf:
                    st.session_state.system_prompt = loaded_conf["system_prompt"]
                if "user_prompt_template" in loaded_conf:
                    st.session_state.user_prompt_template = loaded_conf["user_prompt_template"]
                st.success("配置已加载！")
            except Exception as e:
                st.error(f"配置文件无效: {e}")

        st.markdown("---")
        
        st.header("LLM 配置")
        
        config = {}
        
        config["api_key"] = st.text_input("DeepSeek API Key", type="password", help="在此输入您的 DeepSeek API Key")
        config["base_url"] = st.text_input("Base URL", value="https://api.deepseek.com", help="DeepSeek API 基础地址")
        config["model"] = st.text_input("模型名称", value="deepseek-chat", help="例如: deepseek-chat")
        
        config["temperature"] = st.slider("温度 (Temperature)", 0.0, 2.0, 1.0, 0.1)
        config["max_tokens"] = st.number_input("最大 Token 数", 1, 8192, 4096)
        
        c1, c2 = st.columns(2)
        with c1:
            config["concurrency"] = st.number_input("并发数", 1, 50, 5, help="同时发起的请求数量")
        with c2:
            config["batch_size"] = st.number_input("单次请求条数", 1, 20, 1, help="一次 API 请求处理多少条数据 (Batch API)")
        
        st.markdown("---")
        if not config["api_key"]:
            st.warning("请输入 API Key 以开始使用。")
        
        return config