import json

def generate_python_script(system_prompt, user_tmpl, schema, config):
    """
    Generates a standalone Python script based on current configuration.
    Using raw string replacement to avoid f-string escaping issues.
    """
    
    schema_str = json.dumps(schema, indent=4)
    batch_size = config.get("batch_size", 1)
    
    # We use a raw string for the template. 
    # NOTE: We still need to double braces for the f-strings *inside* the generated code 
    # if we were using f-string here, but since we are not using f-string for the whole block,
    # we can just write the code exactly as it should appear, except for our specific placeholders.
    
    # Actually, simpler: Use f-string but ONLY for the config injection part?
    # No, mixing is confusing.
    
    # Let's use a standard string and .replace().
    
    template = r'''import asyncio
import json
import pandas as pd
from openai import AsyncOpenAI
import os
import math

# --- 配置 (Configuration) ---
API_KEY = "__API_KEY__"
BASE_URL = "__BASE_URL__"
MODEL_NAME = "__MODEL_NAME__"
TEMPERATURE = __TEMPERATURE__
MAX_TOKENS = __MAX_TOKENS__
CONCURRENCY = __CONCURRENCY__
BATCH_SIZE = __BATCH_SIZE__

INPUT_FILE = "your_data.csv" # 请替换为您的文件名
OUTPUT_FILE = "annotated_results.csv"

# --- 提示词与 Schema (Prompts & Schema) ---
SYSTEM_PROMPT = """__SYSTEM_PROMPT__"""

USER_PROMPT_TEMPLATE = """__USER_PROMPT_TEMPLATE__"""

OUTPUT_SCHEMA = __OUTPUT_SCHEMA__

# --- 核心逻辑 (Core Logic) ---

async def analyze_batch(sem, client, chunk_indices, chunk_rows):
    async with sem:
        # 构建 Batch Prompt
        try:
            user_content_lines = []
            for i, row in enumerate(chunk_rows):
                # 替换变量
                single_prompt = USER_PROMPT_TEMPLATE
                for col, val in row.items():
                    placeholder = "{{" + str(col) + "}}"
                    single_prompt = single_prompt.replace(placeholder, str(val))
                # 注意：这里是生成的代码中的 f-string
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

        # 这里使用生成的 SYSTEM_PROMPT 变量
        system_with_schema = f"{SYSTEM_PROMPT}\n\nYou MUST output a valid JSON object strictly following this schema:\n{json.dumps(batch_schema)}"

        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {{"role": "system", "content": system_with_schema}},
                    {{"role": "user", "content": combined_user_content}}
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
                
                # 将结果映射回行
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
        print(f"错误: 未找到文件 {INPUT_FILE}。")
        return

    # 加载数据
    if INPUT_FILE.endswith('.csv'):
        df = pd.read_csv(INPUT_FILE)
    else:
        df = pd.read_excel(INPUT_FILE)
    
    print(f"加载了 {len(df)} 行数据，来自 {INPUT_FILE}。")
    
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    sem = asyncio.Semaphore(CONCURRENCY)
    
    # 分块处理
    indices = df.index.tolist()
    
    def chunker(seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))
    
    tasks = []
    print(f"开始批量标注，并发数: {CONCURRENCY}，Batch Size: {BATCH_SIZE}...")
    
    for chunk_idx in chunker(indices, BATCH_SIZE):
        chunk_rows = [df.loc[i].to_dict() for i in chunk_idx]
        tasks.append(analyze_batch(sem, client, chunk_idx, chunk_rows))
    
    # 收集结果
    results_nested = await asyncio.gather(*tasks)
    flat_results = [item for sublist in results_nested for item in sublist]
    
    # 保存结果
    result_df = pd.DataFrame(flat_results)
    result_df.to_csv(OUTPUT_FILE, index=False)
    print(f"完成！结果已保存至 {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())
'''

    # Replacements
    script = template.replace("__API_KEY__", config.get('api_key', 'YOUR_API_KEY_HERE'))
    script = script.replace("__BASE_URL__", config.get('base_url', 'https://api.deepseek.com'))
    script = script.replace("__MODEL_NAME__", config.get('model', 'deepseek-chat'))
    script = script.replace("__TEMPERATURE__", str(config.get('temperature', 1.0)))
    script = script.replace("__MAX_TOKENS__", str(config.get('max_tokens', 4096)))
    script = script.replace("__CONCURRENCY__", str(config.get('concurrency', 5)))
    script = script.replace("__BATCH_SIZE__", str(batch_size))
    
    script = script.replace("__SYSTEM_PROMPT__", system_prompt.replace('"', '\"'))
    script = script.replace("__USER_PROMPT_TEMPLATE__", user_tmpl.replace('"', '\"'))
    script = script.replace("__OUTPUT_SCHEMA__", schema_str)
    
    return script
