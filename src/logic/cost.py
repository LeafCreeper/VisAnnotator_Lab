import re
import math
from src.logic.chunking import split_text_by_length

def estimate_token_count(text):
    """
    Rough estimation of tokens.
    Heuristic:
    - Chinese characters: ~0.6-0.8 tokens per char.
    - English: ~0.25-0.3 tokens per char (4 chars/token).
    We use a safe blended average of 0.5 tokens per character for mixed content.
    """
    if not text:
        return 0
    return int(len(str(text)) * 0.5)

def fill_template(tmpl, row, override_col=None, override_val=None):
    """
    Fills the prompt template with row values.
    """
    text = tmpl
    # Simple regex substitution or string replace
    # We need to handle {{col}}
    # Find all variables
    vars = re.findall(r"\{\{(.*?)\}\}", tmpl)
    for v in vars:
        val = str(row.get(v, ""))
        if override_col and v == override_col:
            val = str(override_val)
        text = text.replace(f"{{{{{v}}}}}", val)
    return text

def calculate_total_tokens(df, system_prompt, user_tmpl, config):
    """
    Calculates total estimated tokens based on mode.
    Returns: (total_tokens, details_dict)
    """
    mode = config.get("annotation_mode", "Standard")
    sys_tokens = estimate_token_count(system_prompt)
    total_tokens = 0
    
    # 1. Standard Mode
    if mode == "Standard":
        for _, row in df.iterrows():
            u_text = fill_template(user_tmpl, row)
            u_tokens = estimate_token_count(u_text)
            # Output Schema injection approximation (~50 tokens)
            total_tokens += sys_tokens + u_tokens + 50
            
    # 2. Chunking Mode
    elif mode == "Chunking":
        target_var = config.get("chunk_target_var")
        max_len = config.get("max_chunk_len", 600)
        
        # If target var not set, auto-detect (logic mirrored from llm.py)
        if not target_var:
             used_cols = re.findall(r"\{\{(.*?)\}\}", user_tmpl)
             if used_cols:
                 target_var = used_cols[0]

        for _, row in df.iterrows():
            row_dict = row.to_dict()
            doc_text = str(row_dict.get(target_var, "")) if target_var else ""
            
            if not doc_text:
                chunks = [""]
            else:
                chunks = split_text_by_length(doc_text, max_len)
            
            for chunk in chunks:
                u_text = fill_template(user_tmpl, row_dict, override_col=target_var, override_val=chunk)
                u_tokens = estimate_token_count(u_text)
                total_tokens += sys_tokens + u_tokens + 50

    # 3. TrueSkill Mode
    elif mode == "TrueSkill":
        # Calculate Average Item Length
        total_item_tokens = 0
        valid_rows = 0
        for _, row in df.iterrows():
            u_text = fill_template(user_tmpl, row)
            total_item_tokens += estimate_token_count(u_text)
            valid_rows += 1
            
        avg_item_tokens = total_item_tokens / valid_rows if valid_rows > 0 else 0
        
        num_items = len(df)
        comparisons_per_item = config.get("num_comparisons_per_item", 3)
        total_comparisons = (num_items * comparisons_per_item) // 2
        
        # Estimate prompt overhead based on number of variables
        # Base overhead ~100 tokens. Each var adds ~20 tokens (name + json structure)
        fields = config.get("schema_fields", [])
        num_vars = len([f for f in fields if f.get("type") == "Integer"])
        if num_vars == 0: num_vars = 1 # Fallback
        
        overhead = 100 + (num_vars * 20)
        
        # Prompt structure: System + "Compare... Item A: ... Item B: ... Response format..."
        single_comparison_cost = sys_tokens + (2 * avg_item_tokens) + overhead
        
        total_tokens = int(total_comparisons * single_comparison_cost)

    return total_tokens
