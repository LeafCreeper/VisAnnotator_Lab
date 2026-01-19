import json
import asyncio
from openai import AsyncOpenAI
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

async def run_batch_annotation(df, system_prompt, user_tmpl, schema, config, progress_callback=None):
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
