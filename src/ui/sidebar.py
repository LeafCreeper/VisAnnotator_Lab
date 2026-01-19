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
            "prompt_configs": st.session_state.get("prompt_configs", []),
            "current_config_idx": st.session_state.get("current_config_idx", 0)
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
                if "prompt_configs" in loaded_conf:
                    st.session_state.prompt_configs = loaded_conf["prompt_configs"]
                if "current_config_idx" in loaded_conf:
                    st.session_state.current_config_idx = loaded_conf["current_config_idx"]
                
                # Sync global for compatibility
                curr = st.session_state.prompt_configs[st.session_state.current_config_idx]
                st.session_state.system_prompt = curr["system"]
                st.session_state.user_prompt_template = curr["user"]
                
                st.success("配置已加载！")
            except Exception as e:
                st.error(f"配置文件无效: {e}")

        st.markdown("---")
        
        st.header("LLM 模型配置")
        
        # Provider Selection
        provider = st.selectbox(
            "选择模型厂商 (Provider)", 
            ["DeepSeek", "OpenAI (ChatGPT)", "Zhipu AI (GLM)", "Gemini (Google)", "Claude (Anthropic)"]
        )
        
        config = {"provider": provider}
        
        # Dynamic Defaults
        defaults = {
            "DeepSeek": {
                "base_url": "https://api.deepseek.com",
                "models": ["deepseek-chat", "deepseek-reasoner"],
                "key_label": "DeepSeek API Key",
                "help": "在 platform.deepseek.com 获取"
            },
            "OpenAI (ChatGPT)": {
                "base_url": "https://api.openai.com/v1",
                "models": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
                "key_label": "OpenAI API Key",
                "help": "在 platform.openai.com 获取"
            },
            "Zhipu AI (GLM)": {
                "base_url": "https://open.bigmodel.cn/api/paas/v4",
                "models": ["glm-4-plus", "glm-4-flash", "glm-4-air"],
                "key_label": "Zhipu API Key",
                "help": "在 bigmodel.cn 获取"
            },
            "Gemini (Google)": {
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                "models": ["gemini-1.5-flash", "gemini-1.5-pro"],
                "key_label": "Google AI Studio Key",
                "help": "在 aistudio.google.com 获取"
            },
            "Claude (Anthropic)": {
                "base_url": "https://api.anthropic.com", # Not used directly by AsyncOpenAI but good for ref
                "models": ["claude-3-5-sonnet-20240620", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"],
                "key_label": "Anthropic API Key",
                "help": "在 console.anthropic.com 获取"
            }
        }
        
        curr_defaults = defaults[provider]
        
        # API Key
        config["api_key"] = st.text_input(curr_defaults["key_label"], type="password", help=curr_defaults["help"])
        
        # Base URL (Only show for OpenAI-compatible providers)
        if provider != "Claude (Anthropic)":
            config["base_url"] = st.text_input("Base URL", value=curr_defaults["base_url"], help="API 基础地址")
        else:
            config["base_url"] = "" # Claude SDK manages this
        
        # Model Name (Editable with suggestions)
        # We use a text_input with suggestions via help, or a selectbox that allows custom input?
        # Streamlit selectbox doesn't allow custom input easily unless using a specific component.
        # We'll use a selectbox with an "Other..." option or just a text_input.
        # Let's use text_input but pre-fill with a selectbox helper? No, too complex.
        # Just use selectbox for common models, but allow editing? Streamlit doesn't support ComboBox natively well.
        # Let's use a Selectbox with common models.
        
        selected_model = st.selectbox("选择或输入模型名称", curr_defaults["models"] + ["自定义 (Custom)"])
        if selected_model == "自定义 (Custom)":
            config["model"] = st.text_input("请输入模型名称")
        else:
            config["model"] = selected_model
        
        # Parameters
        c1, c2 = st.columns(2)
        with c1:
            config["temperature"] = st.slider("温度 (Temperature)", 0.0, 2.0, 1.0, 0.1)
        with c2:
            config["max_tokens"] = st.number_input("最大 Token 数", 1, 128000, 4096)
        
        c3, c4 = st.columns(2)
        with c3:
            config["concurrency"] = st.number_input("并发数", 1, 50, 5, help="同时发起的请求数量")
        with c4:
            config["batch_size"] = st.number_input("单次请求条数", 1, 20, 1, help="一次 API 请求处理多少条数据")
        
        st.markdown("---")
        if not config["api_key"]:
            st.warning("请输入 API Key 以开始使用。")
        
        return config
