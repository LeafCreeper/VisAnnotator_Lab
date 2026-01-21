import re

def split_text_by_length(text, max_len=600):
    """
    Splits text into chunks, trying to break at sentence-ending punctuation.
    """
    if len(text) <= max_len:
        return [text]
    
    chunks = []
    current_pos = 0
    
    # Punctuations to look for (Chinese and English)
    punctuations = r'[。！？\.!?\n]'
    
    while current_pos < len(text):
        if current_pos + max_len >= len(text):
            chunks.append(text[current_pos:])
            break
        
        # Look for the last punctuation within the max_len window
        window = text[current_pos : current_pos + max_len]
        matches = list(re.finditer(punctuations, window))
        
        if matches:
            # Split at the last found punctuation in this window
            split_at = matches[-1].end()
            chunks.append(text[current_pos : current_pos + split_at])
            current_pos += split_at
        else:
            # No punctuation found, just split at max_len
            chunks.append(text[current_pos : current_pos + max_len])
            current_pos += max_len
            
    return chunks

def is_chunkable_schema(fields):
    """
    Checks if the schema consists of exactly one List type field.
    """
    if len(fields) != 1:
        return False
    return fields[0]["type"] == "List"
