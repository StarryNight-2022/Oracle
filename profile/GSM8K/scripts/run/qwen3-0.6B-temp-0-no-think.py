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
THINKING_FLAG = False
TEMPERATURE = 0
PARAMS = "0.6B"
THINKING_DIR = "no_think_full"  # or "think"

def initialize_model():
    """
    初始化分词器与LLM。
    """
    # 修改为你的实际模型路径
    model_path = f"Model/Qwen3-{PARAMS}"

    # 初始化分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        # tokenizer_mode: TokenizerMode = "auto",
        # skip_tokenizer_init: bool = False,
        trust_remote_code=True,
        # allowed_local_media_path: str = "",
        # tensor_parallel_size: int = 1,
        # dtype: ModelDType = "auto",
        # quantization: QuantizationMethods | None = None,
        # revision: str | None = None,
        # tokenizer_revision: str | None = None,
        seed = 2025,
        # gpu_memory_utilization: float = 0.9,
        # kv_cache_memory_bytes: int | None = None,  # it (when not-None) ignores gpu_memory_utilization
        # swap_space: float = 4,
        # cpu_offload_gb: float = 0,
        # enforce_eager: bool = False,
        # max_seq_len_to_capture: int = 8192,
        # disable_custom_all_reduce: bool = False,
        # disable_async_output_proc: bool = False,
        # hf_token: bool | str | None = None,
        # hf_overrides: HfOverrides | None = None,
        # mm_processor_kwargs: dict[str, Any] | None = None,
        # override_pooler_config: PoolerConfig | None = None,
        # compilation_config: int | dict[str, Any] | CompilationConfig | None = None,
        # logits_processors: list[str | type[LogitsProcessor]] | None = None,
        # **kwargs: Any
        )
    sampling_params = SamplingParams(
        n=1,
        # best_of: int | None,
        # _real_n: int | None,
        # presence_penalty: float,
        # frequency_penalty: float,
        # repetition_penalty: float,
        temperature=TEMPERATURE, # controls the randomness of the sampling
        top_p=0.95,
        # top_k: int,
        # min_p: float,
        seed=2025,  # 随机种子
        # stop: str | list[str] | None,
        # stop_token_ids: list[int] | None,
        # ignore_eos: bool,
        max_tokens=32768, # 每个输出序列生成的最大 token 数量
        # min_tokens: int,
        # logprobs: int | None,
        # prompt_logprobs: int | None,
        # detokenize: bool,
        # skip_special_tokens: bool,
        # spaces_between_special_tokens: bool,
        # logits_processors: Any | None,
        # include_stop_str_in_output: bool,
        # truncate_prompt_tokens: int | None,
        # output_kind: RequestOutputKind,
        # output_text_buffer_length: int,
        # _all_stop_token_ids: set[int],
        # guided_decoding: GuidedDecodingParams | None,
        # logit_bias: dict[int, float] | None,
        # allowed_token_ids: list[int] | None,
        # extra_args: dict[str, Any] | None,
        # bad_words: list[str] | None,
        # _bad_words_token_ids: list[list[int]] | None
    )

    return llm, tokenizer,sampling_params

def generate_response(llm, tokenizer,sampling_params, prompt: str):
    """给定用户问题，生成模型回答，返回outputs"""

    # 格式化提示词
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=THINKING_FLAG,
        )
    # 生成响应
    # 传入列表形式，规避部分 vLLM 版本在字符串输入时的边界问题

    outputs = llm.generate([formatted_prompt], sampling_params, use_tqdm=False)

    # outputs: a list of 'RequestOutput', len() is 1

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
    runtime_dir = "/home/ouyk/project/EdgeRoute/Runtime"  # Runtime
    data_path = os.path.join(
        runtime_dir,
        "GSM8k",
        "grade-school-math",
        "grade_school_math",
        "data",
        "train_0.8.jsonl",  # subsample 20%
    )
    outputs_dir = os.path.join(runtime_dir, "outputs", f"outputs-qwen3-{PARAMS}", f"train_temp_0_{THINKING_DIR}")
    ensure_dir(outputs_dir)

    # 可选：限制样本数（例如快速调试）
    max_samples_env = os.getenv("MAX_SAMPLES")
    max_samples = int(max_samples_env) if max_samples_env and max_samples_env.isdigit() else None

    # 加载数据集（仅使用 question 作为 prompt，不输入 answer）
    dataset = load_jsonl(data_path, max_samples=max_samples)
    if not dataset:
        raise FileNotFoundError(f"数据集为空或不可读取：{data_path}")

    # 初始化模型
    llm, tokenizer,sampling_params = initialize_model()

    total = len(dataset)
    print(f"加载到 {total} 条样本；输出目录：{outputs_dir}")


    for idx, sample in enumerate(dataset, start=1):
        question = sample.get("question", "").strip()
        # reference_answer = sample.get("answer", "").strip()
        index = sample.get("index", "")

        if not question:
            # 跳过空问题
            continue

        start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        t0 = time.time()
        err = None
        outputs = []
        try:
            outputs = generate_response(llm, tokenizer,sampling_params, question)
        except Exception:
            # 记录完整堆栈，便于定位例如 ZeroDivisionError 的真实来源
            err = traceback.format_exc()
        t1 = time.time()
        end_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        duration = t1 - t0

        # 新实现：写入 JSONL 一行记录（每条样本一个文件）
        record: Dict[str, Any] = {
            "index": index,
            # "start_time": start_ts,
            # "end_time": end_ts,
            "runtime": round(float(duration), 3),
            "prompt": question,
            "length_of_question": len(question),
        }

        prompt_token_ids = outputs[0].prompt_token_ids

        output_token_ids = outputs[0].outputs[0].token_ids

        full_response = outputs[0].outputs[0].text

        record.update({
            "prompt_token_ids": prompt_token_ids,
            "length_of_prompt_token_ids": len(prompt_token_ids),
            "output_token_ids": output_token_ids,
            "length_of_output_token_ids": len(output_token_ids),
            "num_cached_tokens": outputs[0].num_cached_tokens,
            "full_response": full_response,
            "length_of_full_response": len(full_response) ,
        })

        # 记录来源文件（当前样本 JSONL 文件路径）
        index_str = str(index)
        out_path = os.path.join(outputs_dir, f"train_{index_str}.jsonl")
        record["source_file"] = out_path

        with open(out_path, "w", encoding="utf-8") as jf:
            jf.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"[{idx}/{total}] 已写入 JSONL: {out_path} | index={index_str} | 耗时: {duration:.2f}s")
