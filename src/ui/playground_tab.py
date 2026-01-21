import streamlit as st
import pandas as pd
import asyncio
import math
import time
from src.logic.llm import run_batch_annotation, run_trueskill_annotation
from src.logic.schema import convert_ui_fields_to_schema
from src.logic.generator import generate_python_script
from src.logic.cost import calculate_total_tokens
from src.logic.chunking import is_chunkable_schema
from src.logic.trueskill_logic import is_trueskill_applicable

def render_playground_tab(config):
    st.header("标注执行台 (Annotation Runner)")
    
    # --- Check Advanced Settings Conflicts ---
    mode = st.session_state.get("annotation_mode", "Standard")
    
    if mode == "Chunking" and not is_chunkable_schema(st.session_state.schema_fields):
        st.error("❌ 当前 Schema 不支持分块处理。分块处理仅对‘仅有一个 List 类型变量’的 Schema 开放。")
    
    if mode == "TrueSkill" and not is_trueskill_applicable(st.session_state.schema_fields):
        st.error("❌ 当前 Schema 不支持 TrueSkill 比较。TrueSkill 仅对‘仅有 Integer 类型变量’的 Schema 开放。")

    if st.session_state.df is None:
        st.warning("请先在“数据上传”标签页上传数据。")
        return
    
    # --- 1. Mode Selection ---
    st.subheader("1. 选择运行模式")
    
    mode_selection = st.radio("模式", ["调试模式 (Debug / Sample)", "生产模式 (Full Batch)"], horizontal=True)
    
    target_df = None
    
    if mode_selection == "调试模式 (Debug / Sample)":
        st.info("在此模式下，仅抽取少量数据进行测试，用于验证 Prompt 和 Schema 是否符合预期。")
        
        c1, c2 = st.columns([1, 2])
        with c1:
            sample_method = st.selectbox("采样方式", ["前 N 行", "随机采样", "关键词过滤"])
        
        with c2:
            if sample_method == "前 N 行":
                n = st.number_input("行数", 1, 100, 5)
                target_df = st.session_state.df.head(n).copy()
                
            elif sample_method == "随机采样":
                n = st.number_input("行数", 1, 100, 5)
                if len(st.session_state.df) > 0:
                    target_df = st.session_state.df.sample(min(n, len(st.session_state.df))).copy()
                else:
                    target_df = pd.DataFrame()
                    
            elif sample_method == "关键词过滤":
                col = st.selectbox("筛选列", st.session_state.df.columns)
                keyword = st.text_input("包含关键词")
                n = st.number_input("最大返回行数", 1, 100, 5)
                
                if keyword:
                    mask = st.session_state.df[col].astype(str).str.contains(keyword, case=False, na=False)
                    filtered = st.session_state.df[mask]
                    target_df = filtered.head(n).copy()
                else:
                    target_df = st.session_state.df.head(n).copy()
        
        st.markdown(f"**当前预览 (Preview): {len(target_df)} rows**")
        st.dataframe(target_df.head(), width="stretch")

    else: # Production Mode
        st.warning("⚠️ 生产模式将对**所有**上传数据进行标注。请确保您的 API Key 余额充足，并且已在调试模式下验证过效果。")
        target_df = st.session_state.df.copy()
        st.markdown(f"**待处理数据总量: {len(target_df)} rows**")
        
        # Cost Estimation (New Logic)
        with st.spinner("正在估算 Token 消耗..."):
            cost_config = config.copy()
            cost_config.update({
                "annotation_mode": mode,
                "chunk_target_var": st.session_state.chunk_target_var,
                "max_chunk_len": st.session_state.max_chunk_len,
                "num_comparisons_per_item": st.session_state.num_comparisons_per_item,
                "schema_fields": st.session_state.schema_fields
            })
            
            est_tokens = calculate_total_tokens(
                target_df, 
                st.session_state.system_prompt, 
                st.session_state.user_prompt_template, 
                cost_config
            )
            
        st.info(f"💰 预计消耗 Token: **~{est_tokens:,}**")

    # --- 2. Run Annotation ---
    st.markdown("---")
    st.subheader("2. 执行标注")
    
    if not config["api_key"]:
        st.error("请在左侧栏输入 API Key。")
        return

    # Add Info for Advanced Modes
    if mode == "Chunking" and is_chunkable_schema(st.session_state.schema_fields):
        st.info(f"ℹ️ **已开启长文档分块模式**。文档将按 {st.session_state.max_chunk_len} 长度进行拆分标注。")
        if st.session_state.chunk_target_var:
            st.caption(f"分块目标变量: `{st.session_state.chunk_target_var}`")
    
    if mode == "TrueSkill" and is_trueskill_applicable(st.session_state.schema_fields):
        st.info(f"ℹ️ **已开启 TrueSkill 比较模式**。系统将进行两两比较。")

    run_btn = st.button("🚀 开始运行任务", type="primary")
    
    if run_btn:
        # --- Pre-run Checks ---
        if mode == "Chunking" and not is_chunkable_schema(st.session_state.schema_fields):
             st.error("❌ 当前 Schema 不支持分块处理。请修改 Schema 或关闭分块模式。")
             st.stop()

        if mode == "TrueSkill" and not is_trueskill_applicable(st.session_state.schema_fields):
             st.error("❌ 当前 Schema 不支持 TrueSkill 比较。请确保 Schema 仅包含 Integer 类型变量。")
             st.stop()

        if mode == "TrueSkill" and len(target_df) < 2:
            st.error("❌ TrueSkill 模式至少需要 2 条数据进行比较。当前数据量不足。")
            st.stop()
            
        schema = convert_ui_fields_to_schema(st.session_state.schema_fields)
        
        # Progress UI
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Calculate Progress Steps
        total_rows = len(target_df)
        if mode == "TrueSkill" and is_trueskill_applicable(st.session_state.schema_fields):
            total_steps = (st.session_state.num_comparisons_per_item * total_rows) // 2
        else:
            batch_size = config.get("batch_size", 1)
            total_steps = math.ceil(total_rows / batch_size)
        
        completed_steps = 0
        start_time = time.time()
        
        def update_progress():
            nonlocal completed_steps
            completed_steps += 1
            progress = min(completed_steps / total_steps, 1.0)
            progress_bar.progress(progress)
            
            elapsed = time.time() - start_time
            avg_time_per_step = elapsed / completed_steps if completed_steps > 0 else 0
            remaining_steps = total_steps - completed_steps
            est_remaining = remaining_steps * avg_time_per_step
            
            status_text.markdown(f"**进度:** {completed_steps}/{total_steps} | **预计剩余时间:** {est_remaining:.1f}s")

        try:
            with st.spinner("正在调用 LLM 进行标注..."):
                # Pass advanced settings into config for logic layer
                config["annotation_mode"] = mode
                config["max_chunk_len"] = st.session_state.max_chunk_len
                config["num_comparisons_per_item"] = st.session_state.num_comparisons_per_item
                config["chunk_target_var"] = st.session_state.chunk_target_var
                
                if mode == "TrueSkill" and is_trueskill_applicable(st.session_state.schema_fields):
                    results = asyncio.run(run_trueskill_annotation(
                        target_df,
                        st.session_state.system_prompt,
                        st.session_state.user_prompt_template,
                        st.session_state.schema_fields,
                        config,
                        progress_callback=update_progress
                    ))
                else:
                    results = asyncio.run(run_batch_annotation(
                        target_df, 
                        st.session_state.system_prompt, 
                        st.session_state.user_prompt_template, 
                        schema, 
                        st.session_state.schema_fields,
                        config,
                        progress_callback=update_progress
                    ))
            
            status_text.success("✅ 标注任务完成！")
            
            # Process Results
            results_map = {res["index"]: res for res in results}
            
            final_data = []
            for index, row in target_df.iterrows():
                row_data = row.to_dict()
                if index in results_map:
                    res = results_map[index]
                    if res["status"] == "success":
                        parsed = res["parsed"]
                        # Robustly update row_data with all schema fields
                        for field in st.session_state.schema_fields:
                            fname = field["name"]
                            val = parsed.get(fname, None)
                            if isinstance(val, list):
                                val = str(val)
                            row_data[fname] = val if val is not None else None # None for empty
                            
                        row_data["_raw_response"] = res["raw"]
                        row_data["_status"] = "success"
                    else:
                        row_data["_error"] = res.get("error", "Unknown Error")
                        row_data["_raw_response"] = res.get("raw", "")
                        row_data["_status"] = "error"
                else:
                    row_data["_status"] = "skipped"
                
                final_data.append(row_data)
            
            st.session_state.results_df = pd.DataFrame(final_data)
            
        except Exception as e:
            st.error(f"运行过程中发生错误: {e}")

    # --- 3. Results & Export ---
    if st.session_state.results_df is not None:
        st.markdown("---")
        st.subheader("3. 结果与导出")
        
        # Metrics
        df_res = st.session_state.results_df
        if "_status" in df_res.columns:
            success_count = len(df_res[df_res["_status"] == "success"])
            error_count = len(df_res[df_res["_status"] == "error"])
            rate = success_count / len(df_res) * 100
            
            c1, c2, c3 = st.columns(3)
            c1.metric("成功条数", success_count)
            c2.metric("失败条数", error_count)
            c3.metric("成功率", f"{rate:.1f}%")
        
        st.dataframe(df_res.head(100), width="stretch")
        if len(df_res) > 100:
            st.caption(f"仅展示前 100 行，共 {len(df_res)} 行。请下载完整文件查看。")

        # Download CSV
        csv = df_res.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 下载标注结果 (CSV)",
            data=csv,
            file_name="annotated_results.csv",
            mime="text/csv",
            type="primary"
        )
        
        # --- Backup: Python Script ---
        with st.expander("🛠️ 附加功能：导出离线运行脚本 (Python Script)"):
            st.info("如果您需要在服务器后台运行或处理超大数据集，可以导出此脚本。")
            
            schema = convert_ui_fields_to_schema(st.session_state.schema_fields)
            script_content = generate_python_script(
                st.session_state.system_prompt, 
                st.session_state.user_prompt_template, 
                schema, 
                config
            )
            
            st.download_button(
                label="下载 batch_label.py",
                data=script_content,
                file_name="batch_label.py",
                mime="text/x-python"
            )