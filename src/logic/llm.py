import json
import asyncio
from openai import AsyncOpenAI

async def call_deepseek_batch(indices, rows_data, system_prompt, user_tmpl, schema, api_key, base_url, model, temperature, max_tokens):
    """
    Async call to DeepSeek API for a batch of rows.
    """
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    
    # Construct Batch User Prompt
    # We will format it as a numbered list or JSON list context
    try:
        user_content_lines = []
        for i, row in enumerate(rows_data):
            # Resolve template for each row
            single_prompt = user_tmpl
            for col, val in row.items():
                single_prompt = single_prompt.replace(f"{{{{{col}}}}}", str(val))
            user_content_lines.append(f"Item {i+1}:\n{single_prompt}")
        
        combined_user_content = "Please analyze the following items:\n\n" + "\n\n".join(user_content_lines)
    except Exception as e:
        return [{"index": idx, "status": "error", "error": f"Prompt Construction Error: {e}"} for idx in indices]

    # Construct System Prompt
    # We need to wrap the single item schema into a list schema
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
            
            # Map back to indices
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
    
    # Chunk the dataframe
    indices = df.index.tolist()
    
    # Helper for chunking
    def chunker(seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))

    async def sem_task(chunk_indices):
        async with sem:
            # Get rows for these indices
            chunk_rows = [df.loc[i].to_dict() for i in chunk_indices]
            
            # Determine if we use single mode (backward compat) or batch mode
            # Actually, let's just use the batch function for everything, even size 1.
            # It simplifies logic.
            
            res = await call_deepseek_batch(
                chunk_indices, chunk_rows, 
                system_prompt, user_tmpl, schema,
                config["api_key"], config["base_url"], config["model"], 
                config["temperature"], config["max_tokens"]
            )
            
            if progress_callback:
                progress_callback() # Callback per request (batch), not per row
            return res

    for chunk_indices in chunker(indices, batch_size):
        tasks.append(sem_task(chunk_indices))
    
    # Flatten results
    results_nested = await asyncio.gather(*tasks)
    flat_results = [item for sublist in results_nested for item in sublist]
    
    return flat_results