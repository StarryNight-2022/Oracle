# 来自/home/ouyk/project/ICDCS/Oracle/profile/scripts/run/qwen3-0.6B-temp-0-en-think.py的load_jsonl()方法
import re
import ast

INVALID = -9999999

# Copy from GSM8K
# def parse_answer_MMLU(answer_str):
#     answer_str = answer_str.replace(",", "")
#     numbers = re.findall(r"\d+", answer_str)
#     if len(numbers) < 1:
#         return INVALID
#     try:
#         return ast.literal_eval(numbers[-1])
#     except SyntaxError:
#         return INVALID

# Set Max_tokens = 1
def parse_answer_MMLU(answer_str):
    return answer_str
