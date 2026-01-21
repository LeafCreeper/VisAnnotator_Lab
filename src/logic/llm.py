import json
import asyncio
from openai import AsyncOpenAI
from src.logic.chunking import split_text_by_length, is_chunkable_schema

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
    Orchestrates TrueSkill comparisons.
    """
    from src.logic.trueskill_logic import init_ratings, generate_pairs, update_comparison
    
    indices = df.index.tolist()
    ratings = init_ratings(indices)
    
    # Generate pairs: num_comparisons_per_item * total_items / 2
    num_items = len(indices)
    total_comparisons = (config.get("num_comparisons_per_item", 3) * num_items) // 2
    pairs = generate_pairs(indices, total_comparisons)
    
    sem = asyncio.Semaphore(config["concurrency"])
    api_key = config["api_key"]
    base_url = config.get("base_url")
    model = config["model"]
    temperature = config["temperature"]
    max_tokens = config["max_tokens"]
    
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def compare_task(pair):
        idx_a, idx_b = pair
        row_a = df.loc[idx_a].to_dict()
        row_b = df.loc[idx_b].to_dict()
        
        # Construct comparison prompt
        # We use the user_tmpl as a base for each item
        def get_content(row):
            p = user_tmpl
            for col, val in row.items():
                p = p.replace(f"{{{{{col}}}}}", str(val))
            return p
        
        content_a = get_content(row_a)
        content_b = get_content(row_b)
        
        # Variable to compare
        var_name = schema_fields[0]["name"]
        
        comparison_prompt = f"Please compare the following two items and decide which one has a higher score for '{var_name}'.\n\n"
        comparison_prompt += f"ITEM A:\n{content_a}\n\n"
        comparison_prompt += f"ITEM B:\n{content_b}\n\n"
        comparison_prompt += f"Which one is higher? Respond with a JSON object: {{\"winner\": \"A\"}}, {{\"winner\": \"B\"}}, or {{\"winner\": \"Draw\"}}."

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
                winner_val = parsed.get("winner", "Draw").upper()
                
                winner = 'draw'
                if winner_val == 'A': winner = 'a'
                elif winner_val == 'B': winner = 'b'
                
                if progress_callback:
                    progress_callback()
                
                return (idx_a, idx_b, winner)
            except Exception as e:
                if progress_callback:
                    progress_callback()
                return (idx_a, idx_b, 'draw')

    tasks = [compare_task(p) for p in pairs]
    results = await asyncio.gather(*tasks)
    
    # Update ratings
    for idx_a, idx_b, winner in results:
        update_comparison(ratings, idx_a, idx_b, winner)
        
    # Convert ratings to a list of results
    final_results = []
    for idx in indices:
        final_results.append({
            "index": idx,
            "status": "success",
            "parsed": {
                f"{var_name}_trueskill_mu": ratings[idx].mu,
                f"{var_name}_trueskill_sigma": ratings[idx].sigma,
                # Optionally map to some score, but mu is the standard mean rating
                var_name: round(ratings[idx].mu, 2)
            },
            "raw": "TrueSkill Rating"
        })
        
    return final_results
