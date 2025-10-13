
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import traceback
from tqdm import tqdm
from pathlib import Path
import argparse
import yaml
import traceback

# 自定义的包
from utils.Benchmarks.benchmarks import load_dataset
from utils.three_modles_router import Simple3Router


def parse_args():
    parser = argparse.ArgumentParser(description="Process some parameters.")

    # parser.add_argument('--config', type=str, default="./config/Qwen3-0.6B-en-think_AND_GPT-4o-mini.yaml",
    #                     help="Specify the config file")

    # parser.add_argument('--config', type=str, default="./config/Qwen3-0.6B-no-think_AND_Deepseek-v3.2-Exp-chat.yaml",
    #                     help="Specify the config file")
    parser.add_argument('--config', type=str, default="./config/Qwen3-0.6B-no-think_AND_Qwen3-14B-no-think_AND_Deepseek-v3.2-Exp-reasoner.yaml",
                        help="Specify the config file")

    parser.add_argument('--latency_constraint', type=float, default=-1,
                        help="Specify the latency_constraint, default -1 means no latency constraint, Unit is seconds.")

    args = parser.parse_args()
    return args


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def print_sign(benchmark: str):
    width = os.get_terminal_size().columns
    print('=' * width)
    print(benchmark.center(width, '*'))


# 每次装载一个结果
def read_jsonl(config_data: Any, benchmark: str, model: str, idx: int):
    filepath = os.path.join(
        config_data["Models"][model]["profile_result"][benchmark],
        f"train_{idx}.jsonl")
    try:
        with open(filepath, 'r') as file:
            line = file.readline()
            return json.loads(line)
    except Exception:
        print(traceback.format_exc())
        return None


if __name__ == "__main__":
    runtime_dir = os.path.dirname(os.path.abspath(__file__))

    args = parse_args()

    # --------------------------------- Load the YAML file -------------------------------
    config_path = Path(args.config)
    if config_path.exists():
        config_file = config_path
    else:
        raise FileNotFoundError(f"配置文件不存在或不可读取：{config_path}")
    with open(config_file, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)  # config_data: <class 'dict'>
    Benchmarks: List[str] = list(config_data["Benchmarks"].keys())
    Models: List[str] = [model_info['name'] for model_info in config_data['Models'].values()]

    # ---------------------------- Initialize Random router class ------------------------
    Simple3Router = Simple3Router()

    latency_constraint = args.latency_constraint
    if latency_constraint == -1:
        latency_constraint = None

    # ------------------------------------ Main Iteration ---------------------------------
    results: Dict[str, Any] = {}
    # Iteration for benchmarks
    for benchmark in Benchmarks:
        dataset = load_dataset(benchmark, config_data["Benchmarks"][benchmark])
        print_sign(benchmark)

        outputs_dir = os.path.join(runtime_dir, "outputs", f"{benchmark}")
        ensure_dir(outputs_dir)

        output_file = (((str(args.config)).split("/")[-1]).split(".yaml")[0]) + f"_3models_no-latency" + ".jsonl"


        times = {model: 0 for model in Models}
        accuracy = 0
        total_latency_each = {model: 0 for model in Models}
        total_tokens_each = {model: 0 for model in Models}

        with open(os.path.join(outputs_dir, output_file), 'w') as fout:
            # Iteration for every query in the benchmark
            for idx, sample in tqdm(enumerate(dataset, start=1)):
                # Iteration for every LLMs's result
                for model in config_data["Models"].keys():
                    results[config_data['Models'][model]["name"]] = read_jsonl(config_data, benchmark, model, idx)
                # Call random router, return oracle-like dict: Dict[str, Any]
                oracle_like = Simple3Router.oracle3LLMs(results)
                results = {}  # 清空

                # Collect Metrics
                times[oracle_like["model"]] += 1
                accuracy += 1 if oracle_like["correctness"] == True else 0
                total_latency_each[oracle_like["model"]] += float(oracle_like["latency"])
                total_tokens_each[oracle_like["model"]] += int(oracle_like["output_tokens"])
                oracle_like["index"] = idx

                fout.write(json.dumps(oracle_like) + '\n')
        total = len(dataset)
        accuracy /= total
        for model in Models:
            print(f"{model}: {(times[model] / total) * 100:.2f}%")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Total Latency: {sum(total_latency_each.values())}s")
        print(f"Average Latency: {sum(total_latency_each.values()) / total:.2f}s")
        for model in Models:
            if times[model] > 0:
                print(f"{model} Total Latency: {total_latency_each[model]}")
                print(f"{model} Average Latency: {total_latency_each[model] / times[model]:.2f}")
            else:
                print(f"{model} Total Latency: 0")
                print(f"{model} Average Latency: 0.00")
        print(f"Total Output Tokens: {sum(total_tokens_each.values())}")
        print(f"Average Output Tokens: {sum(total_tokens_each.values()) / total:.2f}")
        for model in Models:
            if times[model] > 0:
                print(f"{model} Total Output Tokens: {total_tokens_each[model]}")
                print(f"{model} Average Output Tokens: {total_tokens_each[model] / times[model]:.2f}")
            else:
                print(f"{model} Total Output Tokens: 0")
                print(f"{model} Average Output Tokens: 0.00")

        print(f"已完成 {benchmark} 的 3Models Router 结果，保存在：{os.path.join(outputs_dir, output_file)}")

