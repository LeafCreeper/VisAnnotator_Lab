import json
import asyncio
from openai import AsyncOpenAI
from src.logic.chunking import split_text_by_length, is_chunkable_schema
import random


# Import Anthropic conditionally or globally (since we added it to requirements)
try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None

async def call_llm_batch(indices, rows_data, system_prompt, user_tmpl, schema, config):
    """
    Async call to LLM API (OpenAI-compatible or Claude) for a batch of rows.
    """
    api_key = config["api_key"]
    base_url = config.get("base_url")
    model = config["model"]
    temperature = config["temperature"]
    max_tokens = config["max_tokens"]
    provider = config.get("provider", "DeepSeek")

    # Construct Batch User Prompt
    try:
        user_content_lines = []
        for i, row in enumerate(rows_data):
            single_prompt = user_tmpl
            for col, val in row.items():
                single_prompt = single_prompt.replace(f"{{{{{col}}}}}", str(val))
            user_content_lines.append(f"Item {i+1}:\n{single_prompt}")
        
        combined_user_content = "Please analyze the following items:\n\n" + "\n\n".join(user_content_lines)
    except Exception as e:
        return [{"index": idx, "status": "error", "error": f"Prompt Construction Error: {e}"} for idx in indices]

    # Construct Schema-wrapped System Prompt
    batch_schema = {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": schema
            }
        },
        "required": ["results"]
    }
    
    # ---------------------------------------------------------
    # ROUTING LOGIC
    # ---------------------------------------------------------

    if "Claude" in provider:
        # --- CLAUDE (ANTHROPIC) LOGIC ---
        if not AsyncAnthropic:
            return [{"index": idx, "status": "error", "error": "Anthropic library not installed."} for idx in indices]
        
        client = AsyncAnthropic(api_key=api_key)
        
        system_msg = f"{system_prompt}\n\nYou MUST output valid JSON."
        
        # Claude uses tools or prefill for JSON. 
        # But per official docs, the cleanest way for "JSON Mode" without tools 
        # is often asking for it and prefilling "{" or using new beta features.
        # However, for simplicity and robustness across models, we will use the standard prompt engineering 
        # approach combined with a very strong instruction.
        # UPDATE: We will use the standard prompt approach first. If strictly needed, we can use `tool_use`.
        # But given the user wants "Simple configuration", we stick to Prompting + JSON Output.
        
        # Wait, the user specifically mentioned "official docs". 
        # Claude 3.5 Sonnet is very good at JSON.
        # We will append the schema to the User Prompt or System Prompt.
        
        full_system = f"{system_msg}\n\nOutput strictly following this JSON schema:\n{json.dumps(batch_schema)}"
        
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=full_system,
                messages=[
                    {"role": "user", "content": combined_user_content},
                    {"role": "assistant", "content": "{"} # Prefill to force JSON
                ]
            )
            
            # Reconstruct JSON (Prefill means we need to add '{' back)
            content = "{" + response.content[0].text
            
            # Parsing logic is same as below
            try:
                parsed_root = json.loads(content)
                results_list = parsed_root.get("results", [])
                
                output_rows = []
                for i, idx in enumerate(indices):
                    if i < len(results_list):
                        output_rows.append({"index": idx, "status": "success", "raw": content, "parsed": results_list[i]})
                    else:
                        output_rows.append({"index": idx, "status": "error", "raw": content, "error": "Missing result for item"})
                return output_rows

            except json.JSONDecodeError:
                return [{"index": idx, "status": "error", "raw": content, "error": "JSON Decode Error"} for idx in indices]
                
        except Exception as e:
            return [{"index": idx, "status": "error", "error": str(e)} for idx in indices]

    else:
        # --- OPENAI COMPATIBLE LOGIC (DeepSeek, OpenAI, Zhipu, Gemini) ---
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        
        system_with_schema = f"{system_prompt}\n\nYou MUST output a valid JSON object strictly following this schema:\n{json.dumps(batch_schema)}"
        
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_with_schema},
                    {"role": "user", "content": combined_user_content}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            
            try:
                parsed_root = json.loads(content)
                results_list = parsed_root.get("results", [])
                
                output_rows = []
                for i, idx in enumerate(indices):
                    if i < len(results_list):
                        output_rows.append({"index": idx, "status": "success", "raw": content, "parsed": results_list[i]})
                    else:
                        output_rows.append({"index": idx, "status": "error", "raw": content, "error": "Missing result for item"})
                return output_rows

            except json.JSONDecodeError:
                return [{"index": idx, "status": "error", "raw": content, "error": "JSON Decode Error"} for idx in indices]
                
        except Exception as e:
            return [{"index": idx, "status": "error", "error": str(e)} for idx in indices]

async def run_batch_annotation(df, system_prompt, user_tmpl, schema, schema_fields, config, progress_callback=None):
    """
    Orchestrates the batch processing with semaphore for concurrency.
    """
    tasks = []
    sem = asyncio.Semaphore(config["concurrency"])
    batch_size = config.get("batch_size", 1)
    
    indices = df.index.tolist()
    
    def chunker(seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))

    async def sem_task(chunk_indices):
        async with sem:
            chunk_rows = []
            
            # --- Check if Chunking is applicable and enabled ---
            # Check mode from config
            mode = config.get("annotation_mode", "Standard")
            is_chunking = (mode == "Chunking") and is_chunkable_schema(schema_fields)
            
            if is_chunking:
                # If chunking, we process one document at a time for simplicity in merging
                all_results = []
                target_var = config.get("chunk_target_var")
                
                for idx in chunk_indices:
                    row_dict = df.loc[idx].to_dict()
                    
                    # Determine text to split
                    doc_text = ""
                    if target_var and target_var in row_dict:
                         doc_text = str(row_dict[target_var])
                    else:
                        # Fallback: Try to find first used col in prompt
                        import re
                        used_cols = re.findall(r"\{\{(.*?)\}\}", user_tmpl)
                        if used_cols:
                            target_var = used_cols[0] # Auto-select first if not specified
                            doc_text = str(row_dict.get(target_var, ""))
                    
                    if not doc_text:
                        # No text to chunk, treat as empty or error?
                        # Treat as single empty chunk
                        text_chunks = [""]
                    else:
                        text_chunks = split_text_by_length(doc_text, config.get("max_chunk_len", 600))
                    
                    chunk_sub_results = []
                    for sub_idx, text_chunk in enumerate(text_chunks):
                        sub_row = row_dict.copy()
                        # Replace the target variable with the chunk
                        if target_var:
                            sub_row[target_var] = text_chunk
                        
                        sub_res = await call_llm_batch(
                            [idx], [sub_row],
                            system_prompt, user_tmpl, schema,
                            config
                        )
                        chunk_sub_results.append(sub_res[0])
                    
                    # Merge results
                    list_key = schema_fields[0]["name"]
                    merged_list = []
                    raw_combined = ""
                    for sr in chunk_sub_results:
                        if sr["status"] == "success":
                            merged_list.extend(sr["parsed"].get(list_key, []))
                            raw_combined += "\n---\n" + sr["raw"]
                    
                    all_results.append({
                        "index": idx,
                        "status": "success" if chunk_sub_results else "error",
                        "raw": raw_combined,
                        "parsed": {list_key: merged_list}
                    })
                
                if progress_callback:
                    progress_callback()
                return all_results
            else:
                chunk_rows = [df.loc[i].to_dict() for i in chunk_indices]
                
                res = await call_llm_batch(
                    chunk_indices, chunk_rows, 
                    system_prompt, user_tmpl, schema,
                    config
                )
                
                if progress_callback:
                    progress_callback()
                return res

    for chunk_indices in chunker(indices, batch_size):
        tasks.append(sem_task(chunk_indices))
    
    results_nested = await asyncio.gather(*tasks)
    flat_results = [item for sublist in results_nested for item in sublist]
    
    return flat_results


async def run_trueskill_annotation(df, system_prompt, user_tmpl, schema_fields, config, progress_callback=None):
    """
    Orchestrates TrueSkill comparisons for MULTIPLE variables using Batch Active Learning.
    """
    # 保持原有的引用
    from src.logic.trueskill_logic import init_ratings, update_comparison_multi
    
    indices = df.index.tolist()
    ratings = init_ratings(indices, schema_fields)
    
    # 1. 计算总预算 (Total Budget)
    num_items = len(indices)
    # 默认每个 item 比较 3 次，总比较次数 = (3 * N) / 2
    comparisons_per_item = config.get("num_comparisons_per_item", 3)
    total_budget = (comparisons_per_item * num_items) // 2
    
    # 2. 配置并发相关
    concurrency = config["concurrency"]
    # 批次大小设为并发数，保证满载运行的同时，能尽可能快地进行下一轮分数更新
    batch_size = concurrency 
    
    sem = asyncio.Semaphore(concurrency)
    api_key = config["api_key"]
    base_url = config.get("base_url")
    model = config["model"]
    temperature = config["temperature"]
    max_tokens = config["max_tokens"]
    
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    # Get all variable names
    var_names = [f["name"] for f in schema_fields]
    
    # 记录已比较过的配对 (防止重复)
    # 格式: tuple(sorted((idx_a, idx_b)))
    played_pairs = set()

    async def compare_task(pair):
        idx_a, idx_b = pair
        row_a = df.loc[idx_a].to_dict()
        row_b = df.loc[idx_b].to_dict()
        
        # --- [改进 1] 消除位置偏见 (Position Bias Mitigation) ---
        # 随机决定是否交换展示顺序
        is_swapped = random.random() < 0.5
        
        if is_swapped:
            # 视觉上交换，但数据源 row_a/row_b 变量本身不换
            content_first = row_b
            content_second = row_a
        else:
            content_first = row_a
            content_second = row_b

        # Construct comparison prompt inner function
        def get_content(row):
            p = user_tmpl
            for col, val in row.items():
                p = p.replace(f"{{{{{col}}}}}", str(val))
            return p
        
        str_content_1 = get_content(content_first)
        str_content_2 = get_content(content_second)
        
        vars_str = ", ".join([f"'{v}'" for v in var_names])
        
        comparison_prompt = f"Please compare the following two items and decide which one has a higher score for the following criteria: {vars_str}.\n\n"
        comparison_prompt += f"ITEM A:\n{str_content_1}\n\n"
        comparison_prompt += f"ITEM B:\n{str_content_2}\n\n"
        
        # Construct Expected JSON format example
        ex_json = {}
        for v in var_names:
            ex_json[v] = "A (or B or Draw)"
        ex_str = json.dumps(ex_json, indent=2)
        
        comparison_prompt += f"Which one is higher for each? Respond with a VALID JSON object mapping variable names to 'A', 'B', or 'Draw'.\nExample:\n{ex_str}"

        async with sem:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": comparison_prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"}
                )
                content = response.choices[0].message.content
                parsed = json.loads(content)
                
                # Extract winners for each var
                winners = {}
                for v in var_names:
                    # Robust extraction
                    val = parsed.get(v, "Draw")
                    if isinstance(val, dict): # Handle nested {"winner": "A"} case
                        val = val.get("winner", "Draw")
                    
                    # --- [改进 2] 更严格的解析逻辑 ---
                    val_str = str(val).upper().strip().strip('."\'')
                    
                    raw_winner = 'draw'
                    if val_str == 'A':
                        raw_winner = 'a'
                    elif val_str == 'B':
                        raw_winner = 'b'
                    
                    # --- [改进 3] 映射回真实的索引 (Un-swap) ---
                    # 如果 raw_winner 是 'a' (视觉上的第一个)，且 is_swapped=True，那其实选的是 idx_b
                    if raw_winner == 'a':
                        final_winner = 'b' if is_swapped else 'a'
                    elif raw_winner == 'b':
                        final_winner = 'a' if is_swapped else 'b'
                    else:
                        final_winner = 'draw'
                        
                    winners[v] = final_winner
                
                if progress_callback:
                    progress_callback()
                
                return (idx_a, idx_b, winners)
            
            except Exception as e:
                # Log error in production
                if progress_callback:
                    progress_callback() # 依然推进进度条防止卡死
                # Return draws for all on error
                return (idx_a, idx_b, {v: 'draw' for v in var_names})

    # --- [改进 4] 主动学习循环 (Active Learning Loop) ---
    comparisons_done = 0
    
    while comparisons_done < total_budget:
        # A. 动态生成最佳配对 (Swiss System logic)
        # 计算当前所有 item 的综合得分 (所有字段 mu 之和)
        item_scores = []
        for idx in indices:
            # 加微小扰动防止初始 25.0 全部相同导致死板排序
            total_mu = sum([ratings[idx][v].mu for v in var_names])
            score = total_mu + random.uniform(-0.01, 0.01)
            item_scores.append((idx, score))
        
        # 按分数排序：让分数接近的人相邻
        item_scores.sort(key=lambda x: x[1])
        sorted_indices = [x[0] for x in item_scores]
        
        current_batch_pairs = []
        used_in_batch = set()
        
        # 贪心匹配相邻者
        i = 0
        while i < len(sorted_indices) - 1:
            if len(current_batch_pairs) >= batch_size:
                break
            
            # 如果预算快不够了，就不生成满批次
            if (comparisons_done + len(current_batch_pairs)) >= total_budget:
                break

            p1 = sorted_indices[i]
            
            # 寻找 p1 的最佳对手 (优先 p1+1, 其次 p1+2...)
            found_opponent = False
            for offset in range(1, 4): # 向后看 3 个邻居
                if i + offset >= len(sorted_indices): 
                    break
                
                p2 = sorted_indices[i + offset]
                if p2 in used_in_batch: 
                    continue
                
                pair_key = tuple(sorted((p1, p2)))
                
                if pair_key not in played_pairs:
                    current_batch_pairs.append([p1, p2])
                    played_pairs.add(pair_key)
                    used_in_batch.add(p1)
                    used_in_batch.add(p2)
                    found_opponent = True
                    break
            
            # 无论是否找到对手，主指针都向前移
            # (严格来说如果没找到，p1这次轮空，继续看下一个人)
            if found_opponent:
                i += 1 # 下次大循环还需要 i+1，配合循环底部的逻辑
            
            i += 1

        # 如果实在找不到配对了 (例如全比完了)，强制退出防止死循环
        if not current_batch_pairs:
            break

        # B. 并发执行本批次
        tasks = [compare_task(p) for p in current_batch_pairs]
        results = await asyncio.gather(*tasks)
        
        # C. 立即更新分数 (Online Update)
        # 这一步更新后，下一轮 while 循环的 item_scores 就会变化，
        # 从而实现“根据最新战况安排下一场比赛”
        for idx_a, idx_b, winners_map in results:
            update_comparison_multi(ratings, idx_a, idx_b, winners_map)
        
        comparisons_done += len(results)

    # Convert ratings to a list of results
    final_results = []
    for idx in indices:
        res_parsed = {}
        for v in var_names:
            r = ratings[idx][v]
            res_parsed[f"{v}_trueskill_mu"] = r.mu
            res_parsed[f"{v}_trueskill_sigma"] = r.sigma
            res_parsed[v] = round(r.mu, 2)
            
        final_results.append({
            "index": idx,
            "status": "success",
            "parsed": res_parsed,
            "raw": "TrueSkill Multi-Variable Active Learning"
        })
        
    return final_results