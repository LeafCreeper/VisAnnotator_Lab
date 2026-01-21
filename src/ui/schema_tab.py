import streamlit as st
import json
import asyncio
import pandas as pd
import re
from src.logic.schema import convert_ui_fields_to_schema
from src.logic.llm import call_llm_batch

def render_schema_tab(config):
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
                                            index=["String", "Integer", "Boolean", "Enum", "List"].index(field["type"])
                                            , key=f"field_type_{i}",
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

    left_conf, right_test = st.columns([1, 1])

    # === LEFT COLUMN: CONFIGURATION ===
    with left_conf:
        # Configuration Manager
        st.subheader("配置管理")
        
        # Helper: Construct option list strings
        def get_options():
            return [f"{i}: {cfg['name']}" for i, cfg in enumerate(st.session_state.prompt_configs)]
        
        options = get_options()

        # Callbacks for robust state management
        def on_config_add():
            current_config = st.session_state.prompt_configs[st.session_state.current_config_idx]
            new_cfg = {
                "name": f"New Config {len(st.session_state.prompt_configs) + 1}",
                "system": current_config["system"],
                "user": current_config["user"]
            }
            st.session_state.prompt_configs.append(new_cfg)
            new_idx = len(st.session_state.prompt_configs) - 1
            st.session_state.current_config_idx = new_idx
            # Manually update the selectbox state to match the new item
            st.session_state.config_selector = f"{new_idx}: {new_cfg['name']}"

        def on_config_del():
            idx = st.session_state.current_config_idx
            if len(st.session_state.prompt_configs) > 1:
                st.session_state.prompt_configs.pop(idx)
                new_idx = max(0, idx - 1)
                st.session_state.current_config_idx = new_idx
                # Manually update selectbox state
                new_name = st.session_state.prompt_configs[new_idx]["name"]
                st.session_state.config_selector = f"{new_idx}: {new_name}"
            else:
                st.toast("至少保留一个配置！", icon="⚠️")
        
        def on_config_select():
            # Parse "Index: Name" to get Index
            val = st.session_state.config_selector
            idx = int(val.split(":")[0])
            st.session_state.current_config_idx = idx

        # Layout: Dropdown | Name Input | Add | Delete
        c_sel, c_name, c_add, c_del = st.columns([2, 2, 0.5, 0.5])
        
        with c_sel:
            # Sync Index safety check
            if st.session_state.current_config_idx >= len(options):
                st.session_state.current_config_idx = 0

            # Selectbox with bidirectional binding via key and callbacks
            st.selectbox(
                "选择当前配置", 
                options,
                index=st.session_state.current_config_idx,
                key="config_selector",
                on_change=on_config_select
            )

        current_idx = st.session_state.current_config_idx
        current_config = st.session_state.prompt_configs[current_idx]

        with c_name:
            def on_name_change():
                new_name = st.session_state[f"cfg_name_input"]
                st.session_state.prompt_configs[current_idx]["name"] = new_name
                # Update selectbox state to reflect name change immediately?
                # This is tricky because key 'config_selector' holds the old string. 
                # But on rerun, options list regenerates. 
                # Ideally we update the selector string too to avoid "value not in options"
                st.session_state.config_selector = f"{current_idx}: {new_name}"

            st.text_input(
                "配置名称", 
                value=current_config["name"], 
                key="cfg_name_input",
                on_change=on_name_change
            )
                
        with c_add:
            st.button("➕", help="添加新配置", on_click=on_config_add)

        with c_del:
            st.button("🗑️", help="删除当前配置", on_click=on_config_del)
        
        # System Prompt
        st.subheader("系统提示词 (System Prompt)")
        st.caption("建议将详细标注规则指导放在系统提示词中。")
        
        def update_sys():
            st.session_state.prompt_configs[current_idx]["system"] = st.session_state.sys_prompt_area
            st.session_state.system_prompt = st.session_state.sys_prompt_area

        st.text_area(
            "System Prompt", 
            key="sys_prompt_area", 
            value=current_config["system"], 
            height=150,
            on_change=update_sys
        )
        
        # User Prompt & Variable Helpers
        st.subheader("用户提示词模板 (User Prompt)")
        
        # --- Mode Specific UI ---
        mode = st.session_state.get("annotation_mode", "Standard")
        
        if mode == "Chunking":
            st.info("ℹ️ **分块模式**：请指定一个变量作为“长文本来源”。系统将对其进行切割，其他变量将在每个分块请求中保持不变。")
            
            # Extract variables from current user prompt
            current_user_p = current_config["user"]
            vars_in_prompt = re.findall(r"\{\{(.*?)\}\}", current_user_p)
            
            if not vars_in_prompt:
                st.warning("⚠️ 请在下方 User Prompt 中至少插入一个变量 (如 {{content}})。")
            else:
                # Select Chunk Target
                # Default to previously saved or first one
                default_idx = 0
                if st.session_state.chunk_target_var in vars_in_prompt:
                    default_idx = vars_in_prompt.index(st.session_state.chunk_target_var)
                
                selected_var = st.selectbox(
                    "✂️ 选择要分块的变量 (Chunking Target)",
                    vars_in_prompt,
                    index=default_idx,
                    help="此变量的内容若超过长度限制，将被切分。"
                )
                st.session_state.chunk_target_var = selected_var

        elif mode == "TrueSkill":
            st.info("ℹ️ **TrueSkill 模式**：User Prompt 将用于**渲染单条数据**。系统会自动构建 A/B 比较的 Prompt。")
            st.caption("例如：User Prompt 写为 `评论内容: {{text}}`。系统会自动生成 `Compare Item A (评论内容: xxx) vs Item B (评论内容: yyy)`。")
            
            # Check Schema compliance
            has_int = any(f["type"] == "Integer" for f in st.session_state.schema_fields)
            if not has_int:
                st.error("❌ TrueSkill 模式需要至少定义一个 Integer 类型的字段用于存储最终评分。")

        # Standard Hints
        if mode == "Standard":
            st.caption("在单次请求条数大于1时，每条待标注文本都会使得用户提示词被复制一次。")

        st.caption("点击下方变量名即可插入到提示词末尾(jinja2语法)：")
        
        if st.session_state.df is not None:
            cols = st.session_state.df.columns.tolist()
            
            # Helper function to append text
            def append_var(col_name):
                # Always read from current state OR config if state is empty/desync
                current_text = st.session_state.get("user_prompt_area", current_config["user"])
                if current_text is None: current_text = ""
                
                # Add space if needed
                if current_text and not current_text.endswith(" "):
                    current_text += " "
                new_text = current_text + f"{{{{{col_name}}}}}"
                
                # Update both Config and Session State
                st.session_state.prompt_configs[current_idx]["user"] = new_text
                st.session_state.user_prompt_template = new_text
                st.session_state.user_prompt_area = new_text # Direct set for widget
            
            # Use columns to mimic "horizontal list" or "pills"
            if cols:
                # Wrap in a container
                with st.container(border=True):
                     # Simple flex-like wrapping isn't native, so we just list them in rows of 4
                    n_cols = 4
                    rows = [cols[i:i + n_cols] for i in range(0, len(cols), n_cols)]
                    for row_cols in rows:
                        c_list = st.columns(n_cols)
                        for idx, col in enumerate(row_cols):
                            with c_list[idx]:
                                # Use compact button without full container width
                                if st.button(f"{col}", key=f"btn_insert_{col}"):
                                    append_var(col)
                                    # No explicit rerun needed if button callback updates state? 
                                    # Actually yes, to refresh the text_area. 
                                    # But we are not using callback=append_var, we are inline. 
                                    # So we manually rerun? Or let streamlit handle it.
                                    # Streamlit reruns on button click automatically.
                                    pass

        def update_user():
            st.session_state.prompt_configs[current_idx]["user"] = st.session_state.user_prompt_area
            st.session_state.user_prompt_template = st.session_state.user_prompt_area

        st.text_area(
            "User Prompt", 
            key="user_prompt_area", 
            value=current_config["user"], 
            height=250,
            on_change=update_user
        )

    # === RIGHT COLUMN: INSTANT TEST ===
    with right_test:
        st.subheader("⚡ 即时测试 (Instant Test)")
        
        if st.session_state.df is None or len(st.session_state.df) == 0:
            st.warning("请先上传数据。" )
        else:
            # === Preview Column Logic (New) ===
            # Identify a good preview column
            candidates = ['content', 'text', 'body', 'review', 'comment']
            default_prev = st.session_state.df.columns[0]
            for cand in candidates:
                # Case insensitive match
                match = next((c for c in st.session_state.df.columns if c.lower() == cand), None)
                if match:
                    default_prev = match
                    break
            
            # Allow user to change it
            preview_col = st.selectbox(
                "预览列 (Preview Column)", 
                st.session_state.df.columns,
                index=st.session_state.df.columns.get_loc(default_prev)
            )

            # === Test Set Selector (Refactored) ===
            c1, c2 = st.columns([1, 2])
            with c1:
                sample_method = st.selectbox("采样方式", ["手动选择", "前 N 行", "随机采样"], key="test_sample_method")
            
            target_indices = []
            
            with c2:
                if sample_method == "手动选择":
                    # Helper for formatting
                    def format_option(idx):
                        try:
                            # Use SELECTED preview column for snippet
                            val = str(st.session_state.df.loc[idx, preview_col])
                            # Strip newlines for cleaner dropdown
                            val = val.replace('\n', ' ').replace('\r', '') 
                            snippet = val[:30] + "..." if len(val) > 30 else val
                            return f"{idx}: {snippet}"
                        except:
                            return f"{idx}"

                    # Multi-Select
                    if 'test_indices_manual' not in st.session_state:
                        st.session_state.test_indices_manual = []
                        
                    target_indices = st.multiselect(
                        "选择样本",
                        options=st.session_state.df.index.tolist(),
                        default=st.session_state.test_indices_manual,
                        format_func=format_option,
                        key="test_ms",
                        placeholder="输入行号或关键词..."
                    )
                
                elif sample_method == "前 N 行":
                    n = st.number_input("行数", 1, 20, 3, key="test_n_head")
                    target_indices = st.session_state.df.head(n).index.tolist()
                    
                elif sample_method == "随机采样":
                    n = st.number_input("行数", 1, 20, 3, key="test_n_rand")
                    if len(st.session_state.df) > 0:
                        target_indices = st.session_state.df.sample(min(n, len(st.session_state.df))).index.tolist()
            
            # Preview Button & Table Logic
            if target_indices:
                st.caption(f"已选择 {len(target_indices)} 条样本。" )
            
            # Run Button
            if st.button("▶️ 运行测试 (Run Batch Test)", type="primary", disabled=len(target_indices)==0):
                if not config.get("api_key"):
                    st.error("请先配置 API Key")
                else:
                    with st.spinner(f"正在处理 {len(target_indices)} 条样本..."):
                        # Prepare Schema
                        schema = convert_ui_fields_to_schema(st.session_state.schema_fields)
                        
                        # Prepare data
                        rows_data = [st.session_state.df.loc[i].to_dict() for i in target_indices]
                        
                        sys_p = current_config["system"]
                        user_p = current_config["user"]
                        
                        # Run Async
                        try:
                            results = asyncio.run(call_llm_batch(
                                target_indices, rows_data, sys_p, user_p, schema, config
                            ))
                            
                            st.session_state.last_test_results = results
                            st.session_state.last_test_indices = target_indices # Store indices to check if we are stale
                            
                        except Exception as e:
                            st.error(f"System Error: {e}")

            # === Always Display Table if indices selected (New Logic) ===
            if target_indices:
                st.subheader("测试结果 (Results)")
                
                # Check if we have valid cached results for these indices
                # We need to map index -> result if available
                
                # Build base table from input data
                user_tmpl = current_config["user"]
                used_vars = set(re.findall(r"\{\{(.*?)\}\}", user_tmpl))
                valid_used_cols = [c for c in used_vars if c in st.session_state.df.columns]
                
                # If no used vars, default to preview col
                if not valid_used_cols:
                    valid_used_cols = [preview_col]

                # Map results if they exist and match current selection
                results_map = {}
                if 'last_test_results' in st.session_state:
                    # Only use if last_test_indices matches current target_indices roughly?
                    # Or better: just look up by index.
                    # Warning: If user changes Prompt, results might be stale.
                    # Ideally we should clear results if Prompt changes, but that's complex to track.
                    # We will show what we have.
                    for res in st.session_state.last_test_results:
                        results_map[res["index"]] = res
                
                display_rows = []
                for idx in target_indices:
                    original_row = st.session_state.df.loc[idx].to_dict()
                    flat_row = {}
                    
                    # 1. Inputs
                    for col in valid_used_cols:
                        flat_row[f"Input: {col}"] = original_row.get(col, "")
                    
                    # 2. Outputs
                    if idx in results_map:
                        res = results_map[idx]
                        if res["status"] == "success":
                            parsed = res["parsed"]
                            for k, v in parsed.items():
                                flat_row[f"Output: {k}"] = v
                            flat_row["_Status"] = "✅"
                        else:
                            flat_row["_Status"] = "❌ Error"
                            flat_row["_Error"] = res.get("error", "Unknown")
                    else:
                        # Pending
                        flat_row["_Status"] = "⏳ Pending"
                    
                    display_rows.append(flat_row)
                
                res_df = pd.DataFrame(display_rows)
                
                # Sort Columns
                cols = res_df.columns.tolist()
                in_cols = [c for c in cols if c.startswith("Input:")]
                out_cols = [c for c in cols if c.startswith("Output:")]
                status_cols = [c for c in cols if c.startswith("_")]
                
                final_order = in_cols + out_cols + status_cols
                res_df = res_df[final_order]

                st.dataframe(
                    res_df, 
                    width="stretch"
                )
