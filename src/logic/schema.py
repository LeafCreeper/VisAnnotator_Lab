import json

def convert_ui_fields_to_schema(fields):
    """
    Converts the UI field list to a JSON Schema dictionary.
    """
    schema_structure = {
        "type": "object",
        "properties": {},
        "required": []
    }
    
    for f in fields:
        if not f["name"]: continue
        
        field_def = {"description": f["description"]}
        
        if f["type"] == "String":
            field_def["type"] = "string"
        elif f["type"] == "Integer":
            field_def["type"] = "integer"
        elif f["type"] == "Boolean":
            field_def["type"] = "boolean"
        elif f["type"] == "Enum":
            field_def["type"] = "string"
            # Handle potential empty options safely
            options = [opt.strip() for opt in f["options"].split(",") if opt.strip()]
            if options:
                field_def["enum"] = options
        elif f["type"] == "List":
            field_def["type"] = "array"
            field_def["items"] = {"type": "string"}
        
        schema_structure["properties"][f["name"]] = field_def
        schema_structure["required"].append(f["name"])
        
    return schema_structure

def validate_prompt_template(template, columns):
    """
    Checks if all variables in {{variable}} format exist in columns.
    """
    import re
    variables = re.findall(r"\{\{(.*?)\}\}", template)
    missing = [v for v in variables if v not in columns]
    return missing
