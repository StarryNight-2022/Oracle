import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import traceback
from openai import OpenAI, AsyncOpenAI
from dataclasses import asdict
import random
from dotenv import load_dotenv

THINKING_DIR = "think"  # or "think"

load_dotenv()
DEEPSEEK_API_KEY=os.getenv(f"DEEPSEEK_API_KEY")

# 复用 client，避免每次请求创建
_api_kwargs = dict(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")
GLOBAL_CLIENT = OpenAI(**_api_kwargs)

def generate_response(prompt: str):
    """给定用户问题，生成模型回答，返回outputs"""
    api_kwargs: Dict[str, Any]
    # 格式化提示词
    messages = [{"role": "user", "content": prompt}]
    # # 生成响应
    # api_kwargs = dict(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")  # 这里的v1与模型版本无关
    # client = OpenAI(**api_kwargs)
    # formated_messages = [asdict(message) for message in messages]
    t0 = time.time()
    try:
        response = GLOBAL_CLIENT.chat.completions.create(
            # model="deepseek-chat",        # "deepseek-chat" -> non-thinking mode of DeepSeek-V3.2-Exp; "deepseek-reasoner" -> thinking mode of DeepSeek-V3.2-Exp
            model="deepseek-reasoner",
            messages=messages,
            max_tokens=65536,          # refer the deepseek doc: https://api-docs.deepseek.com/zh-cn/quick_start/pricing; no-think:8K, en-think:64k
            temperature=0.0,                # refer the deepseek doc: https://api-docs.deepseek.com/zh-cn/quick_start/parameter_settings
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            n=1)
    except Exception:
        print(traceback.format_exc())
        return None
    t1 = time.time()
    duration = t1 - t0

    return response, duration


def load_jsonl(path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """逐行读取 JSONL 文件。若提供 max_samples 则截断。"""
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
                if max_samples is not None and len(data) >= max_samples:
                    break
            except json.JSONDecodeError:
                # 跳过坏行
                continue
    return data


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


if __name__ == "__main__":
    # 目录与数据路径
    # runtime_dir = os.path.dirname(os.path.dirname(__file__))  # Runtime
    runtime_dir = "/home/ouyk/project/ICDCS/Oracle/profile"  # Runtime
    data_path = os.path.join(
        runtime_dir,
        "GSM8k",
        "grade-school-math",
        "grade_school_math",
        "data",
        # "train_0.2.jsonl",  # subsample 20%
        # "train_0.8.jsonl",  # subsample 80%
        "train.jsonl",   # full
    )
    outputs_dir = os.path.join(runtime_dir, "GSM8k", "outputs", f"outputs-deepseek-v3.2-exp", f"train_temp_0_{THINKING_DIR}")
    ensure_dir(outputs_dir)

    # 可选：限制样本数（例如快速调试）
    max_samples_env = os.getenv("MAX_SAMPLES")
    max_samples = int(max_samples_env) if max_samples_env and max_samples_env.isdigit() else None

    # 加载数据集（仅使用 question 作为 prompt，不输入 answer）
    dataset = load_jsonl(data_path, max_samples=max_samples)
    if not dataset:
        raise FileNotFoundError(f"数据集为空或不可读取：{data_path}")

    total = len(dataset)
    print(f"加载到 {total} 条样本；输出目录：{outputs_dir}")

    outputs_single_file = os.path.join(outputs_dir, "train_outputs.jsonl")
    for idx, sample in enumerate(dataset, start=1):
        question = sample.get("question", "").strip()
        # reference_answer = sample.get("answer", "").strip()

        if not question:
            # 跳过空问题
            continue

        start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        err = None
        response = []
        try:
            response, duration = generate_response(question)
        except Exception:
            # 记录完整堆栈，便于定位例如 ZeroDivisionError 的真实来源
            err = traceback.format_exc()
        end_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        prompt_token_ids = None   # API不提供token_ids，如果后续需要，可以使用tokenizer自行生成
        output_token_ids = None   # API不提供token_ids，如果后续需要，可以使用tokenizer自行生成

        full_response = response.choices[0].message.content

        record: Dict[str, Any] = {
            "index": idx,
            # "start_time": start_ts,
            # "end_time": end_ts,
            "runtime": round(float(duration), 3),
            "prompt": question,
            "length_of_question": len(question),
            "length_of_prompt_token_ids": response.usage.prompt_tokens,
            "length_of_output_token_ids": response.usage.completion_tokens,
            "full_response": full_response,
            "length_of_full_response": len(full_response),
        }
        
        index_str = str(idx)
        out_path = os.path.join(outputs_dir, f"train_{index_str}.jsonl")
        record["source_file"] = out_path
        with open(out_path, "w", encoding="utf-8") as jf:
            jf.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[{idx}/{total}] 已写入 JSONL: {out_path} | index={index_str} | 耗时: {duration:.2f}s")
