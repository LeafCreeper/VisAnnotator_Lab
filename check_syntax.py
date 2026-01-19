import sys
import os

print("Checking syntax for src/logic/llm.py...")
try:
    import src.logic.llm
    print("✅ src/logic/llm.py passed syntax check.")
except Exception as e:
    print(f"❌ src/logic/llm.py FAILED: {e}")

print("Checking syntax for src/logic/generator.py...")
try:
    import src.logic.generator
    print("✅ src/logic/generator.py passed syntax check.")
except Exception as e:
    print(f"❌ src/logic/generator.py FAILED: {e}")
