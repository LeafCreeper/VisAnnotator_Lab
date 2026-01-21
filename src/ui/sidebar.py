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
        
        st.write("") # Spacer
        
        # Import with Help Dialog Trigger
        c_label, c_help = st.columns([0.85, 0.15])
        with c_label:
            st.markdown("**📥 导入配置**")
        
        # State to control dialog visibility
        if 'show_config_help' not in st.session_state:
            st.session_state.show_config_help = False

        def toggle_help():
            st.session_state.show_config_help = True

        with c_help:
            # Simple button, on_click triggers state change
            st.button("❓", on_click=toggle_help, help="标注变量太多，不想手动配置怎么办？")

        # Dialog Implementation (Simulated Modal via st.expander or st.dialog if available in future, 
        # here we use a conditional container or new API if available. 
        # Since we are on Streamlit >= 1.28, `st.dialog` (experimental) or custom modal is needed.
        # But for stability, we will use the `show_onboarding` style approach if we want a true modal, 
        # or `st.popover` (available in newer Streamlit) which we used before but user disliked the style.
        # User requested "Like onboarding dialog". Onboarding uses `st.rerun()` loop or just renders on top.
        # Let's check `src/ui/onboarding.py` to see how it's done.
        
        # Assuming we can use st.dialog (Streamlit 1.34+) which is experimental_dialog.
        # If not, we fallback to session state conditional rendering at top of app?
        # But sidebar renders early.
        # Let's use `st.expander` or just `st.info` if we can't do full modal easily here without complex logic.
        # WAIT: User said "popover" style was "ugly button". But popover IS a modal-like. 
        # User specifically asked for "Circle Exclamation" char.
        # And "Like Newcomer Tutorial".
        
        # Let's try `st.experimental_dialog` if possible, else standard conditional.
        # Since I can't be sure of version, I will stick to the safe `st.popover` logic BUT 
        # change the button appearance as requested to just a char.
        # But wait, I already did popover and user said "Button feels ugly".
        # So I will use a minimal button "❓" and trigger a `st.dialog`.
        
        # Let's define the dialog function
        @st.dialog("🤖 智能配置助手")
        def show_ai_config_help():
            st.markdown("#### 标注变量太多？不想手动配置？")
            st.write("如果您有一份详细的 Codebook (编码手册)，可以让 ChatGPT 或 DeepSeek 帮您直接生成配置文件。")
            st.info("只需将下面的 **提示词** 和 **JSON 模板** 复制给 AI，附上您的编码手册内容即可。")
            
            st.markdown("##### 1. 复制提示词 (Prompt)")
            st.code("请根据我提供的编码手册，生成一个符合以下 JSON 结构的配置文件。Schema 字段类型支持：String, Integer, Boolean, Enum, List。请确保 JSON 格式合法。", language="text")
            
            st.markdown("##### 2. 复制 JSON 模板")
            st.code("""{
  "schema_fields": [
    {
      "name": "sentiment",
      "type": "Enum",
      "options": "Positive, Negative, Neutral",
      "description": "文本的情感倾向"
    },
    {
      "name": "topic",
      "type": "String",
      "options": "",
      "description": "文本的主题"
    }
  ],
  "prompt_configs": [
    {
      "name": "Standard Prompt",
      "system": "You are an expert coder.",
      "user": "Analyze this text: {{content}}"
    }
  ]
}""", language="json")
            st.success("生成的 JSON 保存文件后，在右侧“导入配置”处上传即可一键应用！")

        if st.session_state.get('show_config_help', False):
            show_ai_config_help()
            st.session_state.show_config_help = False # Reset after showing? 
            # Dialogs in Streamlit handle their own closing usually.
            # But the trigger needs to be reset. 
            # Actually st.experimental_dialog needs to be called to open.
        
        # Initialize uploader key if not present
        if "config_uploader_key" not in st.session_state:
            st.session_state.config_uploader_key = 0

        uploaded_config = st.file_uploader(
            "导入配置", 
            type=["json"], 
            label_visibility="collapsed",
            key=f"uploader_{st.session_state.config_uploader_key}"
        )

        # 显示导入成功提示
        if st.session_state.get('import_success_msg'):
            st.success("✅ 配置已成功加载！")
            # 标记为已显示，下次交互时将消失
            st.session_state.import_success_msg = False
        
        if uploaded_config is not None:
            try:
                loaded_conf = json.load(uploaded_config)
                # Update Session State
                if "schema_fields" in loaded_conf:
                    st.session_state.schema_fields = loaded_conf["schema_fields"]
                    
                    # --- FIX: Sync Widget State ---
                    for i, field in enumerate(st.session_state.schema_fields):
                        st.session_state[f"field_name_{i}"] = field.get("name", "")
                        st.session_state[f"field_type_{i}"] = field.get("type", "String")
                        st.session_state[f"field_opts_{i}"] = field.get("options", "")
                        st.session_state[f"field_desc_{i}"] = field.get("description", "")
                
                if "prompt_configs" in loaded_conf:
                    st.session_state.prompt_configs = loaded_conf["prompt_configs"]
                if "current_config_idx" in loaded_conf:
                    st.session_state.current_config_idx = loaded_conf["current_config_idx"]
                
                # Sync global for compatibility
                if st.session_state.prompt_configs:
                    curr = st.session_state.prompt_configs[st.session_state.current_config_idx]
                    st.session_state.system_prompt = curr["system"]
                    st.session_state.user_prompt_template = curr["user"]
                    
                    # --- FIX: Sync Prompt Widgets ---
                    if "sys_prompt_area" in st.session_state:
                         st.session_state["sys_prompt_area"] = curr["system"]
                    if "user_prompt_area" in st.session_state:
                         st.session_state["user_prompt_area"] = curr["user"]
                    if "cfg_name_input" in st.session_state:
                         st.session_state["cfg_name_input"] = curr["name"]
                    if "config_selector" in st.session_state:
                        st.session_state["config_selector"] = f"{st.session_state.current_config_idx}: {curr['name']}"
                
                # 设置成功标记
                st.session_state.import_success_msg = True
                
                # Increment key to reset uploader on next rerun
                st.session_state.config_uploader_key += 1
                st.toast("✅ 配置已成功加载！", icon="🎉")
                st.rerun()
                
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
            config["max_tokens"] = st.number_input("最大 Token 数", 1, 128000, 8192)
        
        c3, c4 = st.columns(2)
        with c3:
            config["concurrency"] = st.number_input("并发数", 1, 50, 5, help="同时发起的请求数量")
        with c4:
            config["batch_size"] = st.number_input("单次请求条数", 1, 20, 1, help="一次 API 请求处理多少条数据")
        
        st.markdown("---")
        
        # --- Advanced Settings ---
        @st.dialog("⚙️ 高级设置 (Advanced Settings)")
        def show_advanced_settings():
            st.subheader("标注模式选择 (Annotation Mode)")
            
            mode_options = ["标准模式 (Standard)", "长文档分块模式 (Chunking)", "TrueSkill 比较模式 (TrueSkill)"]
            
            # Map current state to index
            current_mode = st.session_state.annotation_mode
            if current_mode == "Chunking": idx = 1
            elif current_mode == "TrueSkill": idx = 2
            else: idx = 0
            
            selected_mode_label = st.radio("选择模式", mode_options, index=idx)
            
            # Update State based on selection
            if "Standard" in selected_mode_label:
                st.session_state.annotation_mode = "Standard"
            elif "Chunking" in selected_mode_label:
                st.session_state.annotation_mode = "Chunking"
            elif "TrueSkill" in selected_mode_label:
                st.session_state.annotation_mode = "TrueSkill"
            
            st.divider()
            
            # Contextual Settings
            if st.session_state.annotation_mode == "Chunking":
                st.subheader("分块设置")
                st.session_state.max_chunk_len = st.number_input(
                    "分块长度上限", 
                    100, 5000, 
                    st.session_state.max_chunk_len,
                    help="系统会自动寻找最接近此长度的句号、问号或感叹号进行分割。"
                )
                st.info("💡 请在“提示词与Schema”页面选择具体要分块的变量。")
                
            elif st.session_state.annotation_mode == "TrueSkill":
                st.subheader("TrueSkill 设置")
                st.session_state.num_comparisons_per_item = st.number_input(
                    "每条数据参与比较次数", 
                    1, 20, 
                    st.session_state.num_comparisons_per_item,
                    help="增加比较次数会提高评分精度，但会消耗更多 Token。"
                )
                st.warning("⚠️ 此模式将改变提示词工程界面的逻辑：您只需定义单条数据的展示格式，系统会自动构建比较 Prompt。")
            
            if st.button("确定", use_container_width=True):
                st.rerun()

        if st.button("🛠️ 高级设置", use_container_width=True):
            show_advanced_settings()

        st.markdown("---")
        if not config["api_key"]:
            st.warning("请输入 API Key 以开始使用。")
        
        return config
