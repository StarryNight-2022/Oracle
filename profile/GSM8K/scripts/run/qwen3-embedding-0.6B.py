"""
CUDA_VISIBLE_DEVICES=0 nohup python Runtime/scripts/disable_thinking/qwen3-0.6B-temp-0-no-think.py  > Runtime/logs/disable_thinking/qwen3-0.6B-temp-0-no-think-full.log 2>&1 &
"""

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import traceback

os.environ['VLLM_USE_MODELSCOPE'] = 'True'
# os.environ['MAX_SAMPLES'] = '1' # for debug

def initialize_model():
    """
    初始化分词器与LLM。
    """
    # 修改为你的实际模型路径
    model_path = "/home/ouyk/project/Runtime/Model/Qwen3-Embedding-0.6B"

    model = LLM(model=model_path, task="embed")

    return model

def generate_response(model, prompt: str):
    """给定用户问题，生成模型回答，返回outputs"""

    # 格式化提示词
    messages = [{"role": "user", "content": prompt}]

    outputs = model.embed([prompt])

    return outputs


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
    runtime_dir = "/home/ouyk/project/ICDCS/Oracle"  # Runtime
    data_path = "/home/ouyk/project/ICDCS/Oracle/profile/GSM8K/data/train.jsonl"
    outputs_dir = os.path.join("/home/ouyk/project/ICDCS/Oracle/input/A100/Raw/GSM8K", "Qwen3-Embedding-0.6B")
    ensure_dir(outputs_dir)

    # 可选：限制样本数（例如快速调试）
    max_samples_env = os.getenv("MAX_SAMPLES")
    max_samples = int(max_samples_env) if max_samples_env and max_samples_env.isdigit() else None

    # 加载数据集（仅使用 question 作为 prompt，不输入 answer）
    dataset = load_jsonl(data_path, max_samples=max_samples)
    if not dataset:
        raise FileNotFoundError(f"数据集为空或不可读取：{data_path}")

    # 初始化模型
    model = initialize_model()

    total = len(dataset)
    print(f"加载到 {total} 条样本；输出目录：{outputs_dir}")


    for idx, sample in enumerate(dataset, start=1):
        question = sample.get("question", "").strip()
        # reference_answer = sample.get("answer", "").strip()
        # index = sample.get("index", "")
        index = idx

        if not question:
            # 跳过空问题
            continue

        start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        t0 = time.time()
        err = None
        outputs = []
        try:
            outputs = generate_response(model, question)
        except Exception:
            # 记录完整堆栈，便于定位例如 ZeroDivisionError 的真实来源
            err = traceback.format_exc()
        t1 = time.time()
        end_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        duration = t1 - t0

        # 新实现：写入 JSONL 一行记录（每条样本一个文件）
        record: Dict[str, Any] = {
            "index": index,
            "runtime": round(float(duration), 3),
            "prompt": question,
        }

        # 记录来源文件（当前样本 JSONL 文件路径）
        index_str = str(index)
        out_path = os.path.join(outputs_dir, f"train_{index_str}.jsonl")
        record["source_file"] = out_path

        with open(out_path, "w", encoding="utf-8") as jf:
            jf.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"[{idx}/{total}] 已写入 JSONL: {out_path} | index={index_str} | 耗时: {duration:.2f}s")
