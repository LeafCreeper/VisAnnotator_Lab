import json

def generate_python_script(system_prompt, user_tmpl, schema, config):
    """
    Generates a standalone Python script based on current configuration.
    Supports both OpenAI-compatible providers and Anthropic (Claude).
    """
    
    schema_str = json.dumps(schema, indent=4)
    batch_size = config.get("batch_size", 1)
    provider = config.get("provider", "DeepSeek")
    
    # Robust escaping using chr() to avoid source code syntax errors during file generation
    BS = chr(92) # Backslash
    Q = chr(34)  # Double Quote
    
    # 1. Escape backslashes first: \ -> \\
    # 2. Escape double quotes: " -> \"
    sys_prompt_safe = system_prompt.replace(BS, BS+BS).replace(Q, BS+Q)
    user_tmpl_safe = user_tmpl.replace(BS, BS+BS).replace(Q, BS+Q)
    
    # Common imports and Config
    common_imports = """import asyncio
import json
import pandas as pd
import os
import math
"""
    
    # We use TRIPLE SINGLE QUOTES (''') for the outer string 
    # so we can safely contain TRIPLE DOUBLE QUOTES (""") inside.
    config_section = '''# --- 配置 (Configuration) ---
API_KEY = "{api_key}"
MODEL_NAME = "{model_name}"
TEMPERATURE = {temperature}
MAX_TOKENS = {max_tokens}
CONCURRENCY = {concurrency}
BATCH_SIZE = {batch_size}

INPUT_FILE = "your_data.csv" # 请替换为您的文件名
OUTPUT_FILE = "annotated_results.csv"

# --- 提示词与 Schema (Prompts & Schema) ---
SYSTEM_PROMPT = """{sys_prompt}"""

USER_PROMPT_TEMPLATE = """{user_tmpl}"""

OUTPUT_SCHEMA = {output_schema}
'''.format(
        api_key=config.get('api_key', 'YOUR_API_KEY_HERE'),
        model_name=config.get('model', 'deepseek-chat'),
        temperature=config.get('temperature', 1.0),
        max_tokens=config.get('max_tokens', 4096),
        concurrency=config.get('concurrency', 5),
        batch_size=batch_size,
        sys_prompt=sys_prompt_safe,
        user_tmpl=user_tmpl_safe,
        output_schema=schema_str
    )

    # -------------------------------------------------------------------------
    # CLAUDE (ANTHROPIC) TEMPLATE
    # -------------------------------------------------------------------------
    if "Claude" in provider:
        imports = common_imports + "from anthropic import AsyncAnthropic\n"
        
        core_logic = r'''
# --- 核心逻辑 (Core Logic - Anthropic) ---

async def analyze_batch(sem, client, chunk_indices, chunk_rows):
    async with sem:
        # 构建 Batch Prompt
        try:
            user_content_lines = []
            for i, row in enumerate(chunk_rows):
                single_prompt = USER_PROMPT_TEMPLATE
                for col, val in row.items():
                    placeholder = "{{" + str(col) + "}}"
                    single_prompt = single_prompt.replace(placeholder, str(val))
                user_content_lines.append(f"Item {i+1}:\n{single_prompt}")
            
            combined_user_content = "Please analyze the following items:\n\n" + "\n\n".join(user_content_lines)
            
        except Exception as e:
            print(f"Batch构建错误: {e}")
            return [{**row, "_status": "error", "_error": str(e)} for row in chunk_rows]

        # 构建包含列表的 Schema
        batch_schema = {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": OUTPUT_SCHEMA
                }
            },
            "required": ["results"]
        }

        # System Prompt construction
        system_with_schema = f"{SYSTEM_PROMPT}\n\nOutput strictly following this JSON schema:\n{json.dumps(batch_schema)}"

        try:
            # Prefill assistant message with "{" to enforce JSON
            response = await client.messages.create(
                model=MODEL_NAME,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                system=system_with_schema,
                messages=[
                    {"role": "user", "content": combined_user_content},
                    {"role": "assistant", "content": "{"}
                ]
            )
            
            # Reconstruct JSON (Prefill means we need to add '{' back)
            content = "{" + response.content[0].text
            
            try:
                parsed_root = json.loads(content)
                results_list = parsed_root.get("results", [])
                
                output_rows = []
                for i, row in enumerate(chunk_rows):
                    if i < len(results_list):
                        output_rows.append({**row, **results_list[i], "_status": "success", "_raw_response": content})
                    else:
                        output_rows.append({**row, "_status": "error", "_error": "Missing result for item", "_raw_response": content})
                return output_rows

            except json.JSONDecodeError:
                return [{**row, "_status": "error", "_error": "JSON Decode Error", "_raw_response": content} for row in chunk_rows]
                
        except Exception as e:
            return [{**row, "_status": "error", "_error": str(e)} for row in chunk_rows]

async def main():
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 未找到文件 {INPUT_FILE}。 সন")
        return

    # 加载数据
    if INPUT_FILE.endswith('.csv'):
        df = pd.read_csv(INPUT_FILE)
    else:
        df = pd.read_excel(INPUT_FILE)
    
    print(f"加载了 {len(df)} 行数据，来自 {INPUT_FILE}。 সন")
    
    client = AsyncAnthropic(api_key=API_KEY)
    sem = asyncio.Semaphore(CONCURRENCY)
    
    indices = df.index.tolist()
    
    def chunker(seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))
    
    tasks = []
    print(f"开始批量标注 (Anthropic)，并发数: {CONCURRENCY}，Batch Size: {BATCH_SIZE}...")
    
    for chunk_idx in chunker(indices, BATCH_SIZE):
        chunk_rows = [df.loc[i].to_dict() for i in chunk_idx]
        tasks.append(analyze_batch(sem, client, chunk_idx, chunk_rows))
    
    results_nested = await asyncio.gather(*tasks)
    flat_results = [item for sublist in results_nested for item in sublist]
    
    result_df = pd.DataFrame(flat_results)
    result_df.to_csv(OUTPUT_FILE, index=False)
    print(f"完成！结果已保存至 {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())
'''
        return imports + config_section + core_logic

    # -------------------------------------------------------------------------
    # OPENAI COMPATIBLE TEMPLATE (DeepSeek, Zhipu, Gemini, OpenAI)
    # -------------------------------------------------------------------------
    else:
        base_url_line = f'BASE_URL = "{config.get("base_url", "https://api.deepseek.com")}"\n'
        imports = common_imports + "from openai import AsyncOpenAI\n"
        
        core_logic = r'''
# --- 核心逻辑 (Core Logic - OpenAI Compatible) ---

async def analyze_batch(sem, client, chunk_indices, chunk_rows):
    async with sem:
        try:
            user_content_lines = []
            for i, row in enumerate(chunk_rows):
                single_prompt = USER_PROMPT_TEMPLATE
                for col, val in row.items():
                    placeholder = "{{" + str(col) + "}}"
                    single_prompt = single_prompt.replace(placeholder, str(val))
                user_content_lines.append(f"Item {i+1}:\n{single_prompt}")
            
            combined_user_content = "Please analyze the following items:\n\n" + "\n\n".join(user_content_lines)
            
        except Exception as e:
            print(f"Batch构建错误: {e}")
            return [{**row, "_status": "error", "_error": str(e)} for row in chunk_rows]

        batch_schema = {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": OUTPUT_SCHEMA
                }
            },
            "required": ["results"]
        }

        system_with_schema = f"{SYSTEM_PROMPT}\n\nYou MUST output a valid JSON object strictly following this schema:\n{json.dumps(batch_schema)}"

        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_with_schema},
                    {"role": "user", "content": combined_user_content}
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            
            try:
                parsed_root = json.loads(content)
                results_list = parsed_root.get("results", [])
                
                output_rows = []
                for i, row in enumerate(chunk_rows):
                    if i < len(results_list):
                        output_rows.append({**row, **results_list[i], "_status": "success", "_raw_response": content})
                    else:
                        output_rows.append({**row, "_status": "error", "_error": "Missing result for item", "_raw_response": content})
                return output_rows

            except json.JSONDecodeError:
                return [{**row, "_status": "error", "_error": "JSON Decode Error", "_raw_response": content} for row in chunk_rows]
                
        except Exception as e:
            return [{**row, "_status": "error", "_error": str(e)} for row in chunk_rows]

async def main():
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 未找到文件 {INPUT_FILE}。 সন")
        return

    if INPUT_FILE.endswith('.csv'):
        df = pd.read_csv(INPUT_FILE)
    else:
        df = pd.read_excel(INPUT_FILE)
    
    print(f"加载了 {len(df)} 行数据，来自 {INPUT_FILE}。 সন")
    
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    sem = asyncio.Semaphore(CONCURRENCY)
    
    indices = df.index.tolist()
    
    def chunker(seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))
    
    tasks = []
    print(f"开始批量标注，并发数: {CONCURRENCY}，Batch Size: {BATCH_SIZE}...")
    
    for chunk_idx in chunker(indices, BATCH_SIZE):
        chunk_rows = [df.loc[i].to_dict() for i in chunk_idx]
        tasks.append(analyze_batch(sem, client, chunk_idx, chunk_rows))
    
    results_nested = await asyncio.gather(*tasks)
    flat_results = [item for sublist in results_nested for item in sublist]
    
    result_df = pd.DataFrame(flat_results)
    result_df.to_csv(OUTPUT_FILE, index=False)
    print(f"完成！结果已保存至 {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())
'''
        return imports + config_section.replace('OUTPUT_SCHEMA = ', base_url_line + 'OUTPUT_SCHEMA = ') + core_logic