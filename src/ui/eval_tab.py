import streamlit as st
import pandas as pd
import asyncio
from src.logic.llm import run_batch_annotation
from src.logic.schema import convert_ui_fields_to_schema
from src.logic.metrics import calculate_metrics

def render_eval_tab(config):
    st.header("评估与信度 (Evaluation & Reliability)")
    
    if st.session_state.df is None:
        st.warning("请先上传数据。")
        return

    # Initialize Validation State
    if 'validation_indices' not in st.session_state:
        st.session_state.validation_indices = []
    if 'human_annotations' not in st.session_state:
        st.session_state.human_annotations = {} # {index: {field: value}}
    
    # Store experiment results in session state to persist between reruns
    if 'experiment_results' not in st.session_state:
        st.session_state.experiment_results = {} # {config_name: {index: result_dict}}

    tab_set, tab_run = st.tabs(["1. 构建验证集 (Human Label)", "2. 提示词对比实验 (Experiments)"])
    
    # --- Tab 1: Build Validation Set ---
    with tab_set:
        st.subheader("人工标注验证集")
        
        # Select Sample
        c1, c2 = st.columns([1, 1])
        with c1:
            n_val = st.number_input("添加随机样本数量", 1, 100, 10, key="n_val_input")
        with c2:
            st.write("") # Spacer
            st.write("")
            if st.button("➕ 添加样本到验证集"):
                current_indices = set(st.session_state.validation_indices)
                available_indices = [i for i in st.session_state.df.index if i not in current_indices]
                
                if len(available_indices) < n_val:
                    st.warning("剩余可用数据不足。")
                    to_add = available_indices
                else:
                    import random
                    to_add = random.sample(available_indices, n_val)
                
                st.session_state.validation_indices.extend(to_add)
                st.success(f"已添加 {len(to_add)} 条数据。")
                st.rerun()
            
        if not st.session_state.validation_indices:
            st.info("验证集为空。请先添加样本。")
        else:
            st.markdown(f"**验证集大小:** {len(st.session_state.validation_indices)}")
            
            if st.button("🗑️ 清空验证集", type="secondary"):
                st.session_state.validation_indices = []
                st.session_state.human_annotations = {}
                st.rerun()
            
            st.divider()
            
            # --- New Feature: Import Ground Truth ---
            with st.expander("📂 从数据列导入正确答案 (Import Ground Truth)", expanded=False):
                st.info("如果您上传的数据中已经包含了某些变量的正确标注（Ground Truth），可以在此将其批量映射到验证集，无需手动重新标注。")
                
                mapping = {}
                cols = st.session_state.df.columns.tolist()
                cols_options = ["(不导入)"] + cols
                
                # Grid layout for mapping
                m_cols = st.columns(3)
                
                for i, field in enumerate(st.session_state.schema_fields):
                    fname = field["name"]
                    if not fname: continue
                    
                    # Auto-match if column name matches field name
                    default_idx = 0
                    if fname in cols:
                        default_idx = cols_options.index(fname)
                    
                    with m_cols[i % 3]:
                        mapping[fname] = st.selectbox(
                            f"Field `{fname}` 对应列:", 
                            cols_options, 
                            index=default_idx,
                            key=f"map_{fname}"
                        )
                
                if st.button("📥 开始导入 (Import)", type="primary"):
                    imported_count = 0
                    for idx in st.session_state.validation_indices:
                        if idx not in st.session_state.human_annotations:
                            st.session_state.human_annotations[idx] = {}
                            
                        for fname, col_name in mapping.items():
                            if col_name != "(不导入)":
                                val = st.session_state.df.at[idx, col_name]
                                st.session_state.human_annotations[idx][fname] = str(val)
                    
                    st.success(f"成功为验证集导入了标注数据！")
                    st.rerun()

            st.divider()

            # --- Annotation Interface ---
            
            # Select Text Column to Display
            cols = st.session_state.df.columns.tolist()
            # Try to guess text column (contains 'text', 'content', 'body' or is object type)
            default_idx = 0
            for i, col in enumerate(cols):
                if any(x in col.lower() for x in ['text', 'content', 'body', 'comment', 'review']):
                    default_idx = i
                    break
            
            display_col = st.selectbox("选择用于标注参考的文本列:", cols, index=default_idx)
            
            st.divider()

            val_df = st.session_state.df.loc[st.session_state.validation_indices]
            schema_fields = st.session_state.schema_fields
            
            for idx, row in val_df.iterrows():
                # Card-like container
                with st.container(border=True):
                    # Display Text Content
                    text_content = row[display_col]
                    st.markdown(f"**{display_col}:**")
                    st.info(f"{text_content}") # Use st.info box for better readability of text
                    st.caption(f"Row Index: {idx}")
                    
                    # Ensure storage exists
                    if idx not in st.session_state.human_annotations:
                        st.session_state.human_annotations[idx] = {}
                    
                    # Input fields grid
                    input_cols = st.columns(len(schema_fields))
                    
                    for i, field in enumerate(schema_fields):
                        field_name = field["name"]
                        if not field_name: continue
                        
                        current_val = st.session_state.human_annotations[idx].get(field_name, None)
                        
                        with input_cols[i]:
                            if field["type"] == "Enum":
                                options = [opt.strip() for opt in field["options"].split(",") if opt.strip()]
                                index = options.index(current_val) if current_val in options else 0
                                
                                new_val = st.selectbox(
                                    f"{field_name}", 
                                    options, 
                                    index=index,
                                    key=f"human_{idx}_{field_name}",
                                    label_visibility="visible"
                                )
                                st.session_state.human_annotations[idx][field_name] = new_val
                            else:
                                new_val = st.text_input(
                                    f"{field_name}",
                                    value=str(current_val) if current_val else "",
                                    key=f"human_{idx}_{field_name}"
                                )
                                st.session_state.human_annotations[idx][field_name] = new_val

    # --- Tab 2: Run Experiments ---
    with tab_run:
        st.subheader("提示词组合对比实验")
        
        if not st.session_state.validation_indices:
            st.warning("⚠️ 请先在 Tab 1 构建验证集并进行人工标注。")
            return
        elif not config["api_key"]:
            st.error("⚠️ 请配置 API Key。")
            return
            
        # 1. Select Configs to Run
        all_configs = st.session_state.prompt_configs
        config_names = [c["name"] for c in all_configs]
        
        selected_configs = st.multiselect(
            "选择要对比的提示词配置 (Select Configs)", 
            config_names, 
            default=[all_configs[st.session_state.current_config_idx]["name"]]
        )
        
        if st.button("🚀 运行实验 (Run Experiments)", type="primary"):
            val_df = st.session_state.df.loc[st.session_state.validation_indices].copy()
            schema = convert_ui_fields_to_schema(st.session_state.schema_fields)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_steps = len(selected_configs)
            
            for i, cfg_name in enumerate(selected_configs):
                status_text.text(f"正在运行配置: {cfg_name} ({i+1}/{total_steps})...")
                
                # Find config object
                cfg_obj = next(c for c in all_configs if c["name"] == cfg_name)
                
                try:
                    # Run Batch
                    results = asyncio.run(run_batch_annotation(
                        val_df,
                        cfg_obj["system"],
                        cfg_obj["user"],
                        schema,
                        config
                    ))
                    
                    # Store processed results
                    results_map = {res["index"]: res for res in results}
                    st.session_state.experiment_results[cfg_name] = results_map
                    
                except Exception as e:
                    st.error(f"配置 {cfg_name} 运行失败: {e}")
                
                progress_bar.progress((i + 1) / total_steps)
            
            status_text.success("所有配置运行完成！")

        st.divider()

        # 2. Display Results & Analysis
        if selected_configs and any(name in st.session_state.experiment_results for name in selected_configs):
            st.subheader("📊 实验结果分析")
            
            # Prepare Analysis Data
            # We want a DataFrame where:
            # Index: Row ID
            # Columns: Human_{Field}, ConfigA_{Field}, ConfigB_{Field}...
            
            analysis_rows = []
            
            for idx in st.session_state.validation_indices:
                row_data = {"index": idx}
                
                # Human Labels
                h_labels = st.session_state.human_annotations.get(idx, {})
                for k, v in h_labels.items():
                    row_data[f"Human_{k}"] = v
                
                # AI Labels for each Config
                for cfg_name in selected_configs:
                    if cfg_name in st.session_state.experiment_results:
                        res_map = st.session_state.experiment_results[cfg_name]
                        ai_res = res_map.get(idx, {}).get("parsed", {})
                        for k, v in ai_res.items():
                            row_data[f"{cfg_name}_{k}"] = v
                
                analysis_rows.append(row_data)
            
            df_analysis = pd.DataFrame(analysis_rows)
            st.dataframe(df_analysis, use_container_width=True)
            
            # 3. Metrics Table
            st.subheader("📈 信效度指标 (Metrics)")
            
            # For each field, compare Configs vs Human
            for field in st.session_state.schema_fields:
                fname = field["name"]
                if not fname: continue
                
                with st.expander(f"字段: {fname}", expanded=True):
                    metrics_data = []
                    
                    # Human Column
                    h_col = f"Human_{fname}"
                    if h_col not in df_analysis.columns:
                        st.warning(f"字段 {fname} 缺少人工标注。")
                        continue
                        
                    # Calculate metrics for each config
                    for cfg_name in selected_configs:
                        a_col = f"{cfg_name}_{fname}"
                        if a_col in df_analysis.columns:
                            m = calculate_metrics(df_analysis, h_col, a_col)
                            metrics_data.append({
                                "Configuration": cfg_name,
                                "Accuracy": f"{m['accuracy']:.2%}",
                                "Kappa": f"{m['kappa']:.4f}",
                                "N": m["n"]
                            })
                    
                    if metrics_data:
                        st.table(pd.DataFrame(metrics_data).set_index("Configuration"))
                    else:
                        st.info("无有效数据计算指标。")

            # 4. Inter-Config Comparison (Optional)
            if len(selected_configs) > 1:
                st.subheader("🤝 配置间一致性 (Inter-Config Agreement)")
                st.caption("比较不同提示词配置之间的输出一致性 (Cohen's Kappa)")
                
                # Matrix
                for field in st.session_state.schema_fields:
                    fname = field["name"]
                    if not fname: continue
                    
                    st.markdown(f"**字段: {fname}**")
                    matrix = pd.DataFrame(index=selected_configs, columns=selected_configs)
                    
                    for c1 in selected_configs:
                        for c2 in selected_configs:
                            col1 = f"{c1}_{fname}"
                            col2 = f"{c2}_{fname}"
                            if col1 in df_analysis.columns and col2 in df_analysis.columns:
                                m = calculate_metrics(df_analysis, col1, col2)
                                matrix.loc[c1, c2] = f"{m['kappa']:.4f}"
                    
                    st.table(matrix)