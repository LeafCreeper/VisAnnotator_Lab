import streamlit as st
import json
from src.logic.schema import convert_ui_fields_to_schema

def render_schema_tab():
    st.header("定义输出结构 (Schema)")
    st.markdown("定义您希望 LLM 提取的字段。这将构建 JSON Schema。")
    
    col_edit, col_preview = st.columns([1, 1])
    
    with col_edit:
        st.subheader("字段编辑器")
        
        def add_field():
            st.session_state.schema_fields.append({"name": "", "type": "String", "options": "", "description": ""})
        
        def remove_field(index):
            st.session_state.schema_fields.pop(index)

        for i, field in enumerate(st.session_state.schema_fields):
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.text_input(f"字段名 #{i+1}", value=field["name"], key=f"field_name_{i}", 
                                  on_change=lambda i=i: st.session_state.schema_fields[i].update({"name": st.session_state[f"field_name_{i}"]}))
                with c2:
                    if st.button("🗑️", key=f"del_{i}"):
                        remove_field(i)
                        st.rerun()
                
                c3, c4 = st.columns([1, 2])
                with c3:
                    type_val = st.selectbox(f"类型 #{i+1}", ["String", "Integer", "Boolean", "Enum", "List"], 
                                            index=["String", "Integer", "Boolean", "Enum", "List"].index(field["type"]), key=f"field_type_{i}",
                                            on_change=lambda i=i: st.session_state.schema_fields[i].update({"type": st.session_state[f"field_type_{i}"]}))
                with c4:
                    if type_val == "Enum":
                        st.text_input(f"选项 (逗号分隔) #{i+1}", value=field["options"], key=f"field_opts_{i}",
                                      on_change=lambda i=i: st.session_state.schema_fields[i].update({"options": st.session_state[f"field_opts_{i}"]}))
                
                st.text_input(f"描述 (Description) #{i+1}", value=field["description"], key=f"field_desc_{i}",
                              on_change=lambda i=i: st.session_state.schema_fields[i].update({"description": st.session_state[f"field_desc_{i}"]}))

        st.button("添加字段", on_click=add_field)

    with col_preview:
        st.subheader("Schema 预览 (JSON)")
        schema_structure = convert_ui_fields_to_schema(st.session_state.schema_fields)
        st.json(schema_structure)

    st.markdown("---")
    
    # --- Prompt Engineering Section ---
    st.header("提示词工程 (Prompt Engineering)")
    
    # Configuration Manager
    st.subheader("配置管理")
    
    # Layout: Dropdown | Name Input | Add | Delete
    c_sel, c_name, c_add, c_del = st.columns([2, 2, 0.5, 0.5])
    
    with c_sel:
        # Create options list
        options = [f"{i}: {cfg['name']}" for i, cfg in enumerate(st.session_state.prompt_configs)]
        selected_option = st.selectbox(
            "选择当前配置", 
            options, 
            index=st.session_state.current_config_idx,
            key="config_selector"
        )
        # Update index
        new_idx = int(selected_option.split(":")[0])
        if new_idx != st.session_state.current_config_idx:
            st.session_state.current_config_idx = new_idx
            st.rerun()

    current_idx = st.session_state.current_config_idx
    current_config = st.session_state.prompt_configs[current_idx]

    with c_name:
        new_name = st.text_input("配置名称", value=current_config["name"], key=f"cfg_name_{current_idx}")
        if new_name != current_config["name"]:
            st.session_state.prompt_configs[current_idx]["name"] = new_name
            # No rerun needed strictly, but good for UI sync if we wanted
            
    with c_add:
        if st.button("➕", help="添加新配置"):
            # Clone current or create new
            new_cfg = {
                "name": f"New Config {len(st.session_state.prompt_configs) + 1}",
                "system": current_config["system"], # Clone
                "user": current_config["user"]       # Clone
            }
            st.session_state.prompt_configs.append(new_cfg)
            st.session_state.current_config_idx = len(st.session_state.prompt_configs) - 1
            st.rerun()

    with c_del:
        if st.button("🗑️", help="删除当前配置"):
            if len(st.session_state.prompt_configs) > 1:
                st.session_state.prompt_configs.pop(current_idx)
                st.session_state.current_config_idx = max(0, current_idx - 1)
                st.rerun()
            else:
                st.toast("至少保留一个配置！", icon="⚠️")

    # Prompt Editors (Bound to Current Config)
    st.markdown(f"#### 正在编辑: **{current_config['name']}**")
    
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        st.subheader("系统提示词 (System Prompt)")
        
        def update_sys():
            st.session_state.prompt_configs[current_idx]["system"] = st.session_state.sys_prompt_area
            # Sync global for compatibility
            st.session_state.system_prompt = st.session_state.sys_prompt_area

        st.text_area(
            "System Prompt", 
            key="sys_prompt_area", 
            value=current_config["system"], 
            height=200,
            on_change=update_sys
        )
        
    with col_p2:
        st.subheader("用户提示词模板 (User Prompt)")
        st.markdown("使用 Jinja2 语法插入变量，例如 `{{column_name}}`。")
        
        if st.session_state.df is not None:
            cols = st.session_state.df.columns.tolist()
            st.info(f"可用列名: {', '.join(cols)}")
        
        def update_user():
            st.session_state.prompt_configs[current_idx]["user"] = st.session_state.user_prompt_area
            # Sync global for compatibility
            st.session_state.user_prompt_template = st.session_state.user_prompt_area

        st.text_area(
            "User Prompt", 
            key="user_prompt_area", 
            value=current_config["user"], 
            height=200,
            on_change=update_user
        )
