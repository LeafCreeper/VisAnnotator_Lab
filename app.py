import streamlit as st
from src.ui.sidebar import render_sidebar
from src.ui.data_tab import render_data_tab
from src.ui.schema_tab import render_schema_tab
from src.ui.playground_tab import render_playground_tab
from src.ui.eval_tab import render_eval_tab
from src.ui.onboarding import show_onboarding

# Set page config
st.set_page_config(
    page_title="VisAnnotator Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
# Trigger redeploy
if 'df' not in st.session_state:
    st.session_state.df = None
if 'schema_fields' not in st.session_state:
    st.session_state.schema_fields = [{"name": "sentiment", "type": "Enum", "options": "Positive, Negative, Neutral", "description": "The sentiment of the text."}]
if 'results_df' not in st.session_state:
    st.session_state.results_df = None

# Validation State
if 'validation_indices' not in st.session_state:
    st.session_state.validation_indices = []
if 'human_annotations' not in st.session_state:
    st.session_state.human_annotations = {}

# Tutorial State
if 'has_seen_tutorial' not in st.session_state:
    st.session_state.has_seen_tutorial = False

# --- Advanced Settings State ---
if 'annotation_mode' not in st.session_state:
    st.session_state.annotation_mode = "Standard" # Options: "Standard", "Chunking", "TrueSkill"
if 'max_chunk_len' not in st.session_state:
    st.session_state.max_chunk_len = 600
if 'num_comparisons_per_item' not in st.session_state:
    st.session_state.num_comparisons_per_item = 3
if 'chunk_target_var' not in st.session_state:
    st.session_state.chunk_target_var = None

# --- New: Multi-Prompt Configuration State ---
if 'prompt_configs' not in st.session_state:
    # Migrate old state if exists, else default
    initial_sys = st.session_state.get('system_prompt', "You are a helpful assistant that analyzes text and outputs JSON.")
    initial_user = st.session_state.get('user_prompt_template', "Analyze the following text: {{content}}")
    
    st.session_state.prompt_configs = [
        {"name": "默认配置 (Default)", "system": initial_sys, "user": initial_user}
    ]

if 'current_config_idx' not in st.session_state:
    st.session_state.current_config_idx = 0

# Helper to sync legacy keys for backward compatibility
current = st.session_state.prompt_configs[st.session_state.current_config_idx]
st.session_state.system_prompt = current["system"]
st.session_state.user_prompt_template = current["user"]

def main():
    # Show onboarding dialog if needed
    if not st.session_state.has_seen_tutorial:
        show_onboarding()

    # Render Sidebar and get Config
    config = render_sidebar()

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "1. 数据接入 (Data)", 
        "2. 提示词与Schema (Define)", 
        "3. 标注执行台 (Runner)",
        "4. 评估与信度 (Evaluation)"
    ])

    with tab1:
        render_data_tab()

    with tab2:
        render_schema_tab(config)

    with tab3:
        render_playground_tab(config)
        
    with tab4:
        render_eval_tab(config)

if __name__ == "__main__":
    main()