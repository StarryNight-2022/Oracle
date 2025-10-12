# 该python脚本负责基于已有的模型输出结果，构建出Oracle路由的决策结果。
# 提前准备内容：
# a.在各个BenchMarks上运行各个LLMs(本地/API)，记录每个prompt的输出结果为一个jsonl文件
# b.将上述jsonl文件进行整理，确保可以使用BenchMarks中的索引直接索引的每个输出结果的jsonl文件(保序+快速检索)。最终将数据按照标准且统一的形式存储到同一个位置，便于后续使用时进行索引。
# c.判断输出结果的正确性(evaluation)，并将结果整理后写入到jsonl文件中。
# 本代码构建计划：
# step1.使用配置文件管理所支持的LLMs(本地/API)的各个数据集测试结果，这将是一个二维的表项(LLMs \times BenchMarks)。
# step2.在代码中可以指定选择哪些LLMs:List[str]与哪些BenchMarks:List[str]
# step3.制定一个Oracle的判定指标，并将其编写成一个可调用的python方法。(见飞书)
# step4.遍历BenchMarks，对于每一个BenchMarks：顺序读取输入的idx，根据该idx找到所有LLMs的输出结果jsonl文件，读取文件中的关键信息，交予Oracle判定方法进行判定并得到结果
# step5.绘制图像来呈现Oracle。(见飞书：https://tcnk04l8pdtf.feishu.cn/wiki/X6yKw6cYZiuxdpk6Z1ccutJrn3b?renamingWikiNode=true#share-PrvMdlUrBoP1Wnxo5r4cm9Rhnsh)

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
from utils.oracle_router import Oracle

def parse_args():
    parser = argparse.ArgumentParser(description="Process some parameters.")
    
    parser.add_argument('--config', type=str, default="./config/Qwen3-0.6B-en-think_AND_Deepseek-v3.2-Exp-chat.yaml",
                        help="Specify the config file")
    
    # parser.add_argument('--config', type=str, default="./config/development.yaml",
    #                 help="Specify the config file")
    
    parser.add_argument('--latency_constraint', type=float, default=-1,
                    help="Specify the latency_constraint, default -1 means no latency constraint, Unit is seconds.")

    args = parser.parse_args()
    return args

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        
def print_sign(benchmark: str):
    width = os.get_terminal_size().columns
    print('='*width)
    print(benchmark.center(width, '*'))
    
# 每次装载一个结果
def read_jsonl(config_data: Any, benchmark:str, model: str, idx: int):
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
    
    #--------------------------------- Load the YAML file -------------------------------
    config_path = Path(args.config)
    if config_path.exists():
        config_file = config_path
    else:
        raise FileNotFoundError(f"配置文件不存在或不可读取：{config_path}")
    with open(config_file, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)         # config_data: <class 'dict'>
    Benchmarks: List[str] = list(config_data["Benchmarks"].keys())
    Models: List[str] = [model_info['name'] for model_info in config_data['Models'].values()]
    
    #---------------------------- Initialize Oracle judging class ------------------------
    Oracle_Judge = Oracle(config_data)
    
    latency_constraint = args.latency_constraint
    if latency_constraint == -1:
        latency_constraint = None
    
    #------------------------------------ Main Iteration ---------------------------------
    results: Dict[str, Any] = {}
    # Iteration for benchmarks
    for benchmark in Benchmarks:
        dataset = load_dataset(benchmark, config_data["Benchmarks"][benchmark])
        print_sign(benchmark)
        
        outputs_dir = os.path.join(runtime_dir, "outputs", f"{benchmark}")
        ensure_dir(outputs_dir)
        if latency_constraint == None:
            output_file = (((str(args.config)).split("/")[-1]).split(".yaml")[0]) + "_no-latency-constraint" + ".jsonl"
        elif type(latency_constraint) == float:
            output_file = (((str(args.config)).split("/")[-1]).split(".yaml")[0]) + f"_{latency_constraint}s-latency-constraint" + ".jsonl"
        else:
            raise ValueError("latency_constraint must be None or float.")
        
        times = {model:0 for model in Models}
        accuracy = 0
        total_latency_each = {model:0 for model in Models}
        total_tokens_each = {model:0 for model in Models}
        
        with open(os.path.join(outputs_dir, output_file), 'w') as fout:
            # Iteration for every query in the benchmark
            for idx, sample in tqdm(enumerate(dataset, start=1)):
                # Iteration for every LLMs's result
                for model in config_data["Models"].keys():
                    results[config_data['Models'][model]["name"]] = read_jsonl(config_data, benchmark, model, idx)
                # Call oracle judging func, return oracle: Dict[str, Any]
                oracle = Oracle_Judge.get_oracle(results, latency_constraint)       # here the oracle choice
                results = {}    # 清空
                
                # Collect Metrics
                times[oracle["model"]] += 1
                accuracy += 1 if oracle["correctness"] == True else 0
                total_latency_each[oracle["model"]] += float(oracle["latency"])
                total_tokens_each[oracle["model"]] += int(oracle["output_tokens"])
                oracle["index"] = idx

                fout.write(json.dumps(oracle) + '\n')
        total = len(dataset)
        accuracy /= total
        for model in Models:
            print(f"{model}: {(times[model]/total)*100:.2f}%")
        print(f"Accuracy: {accuracy*100:.2f}%")
        print(f"Total Latency: {sum(total_latency_each.values())}s")
        print(f"Average Latency: {sum(total_latency_each.values())/total:.2f}s")
        for model in Models:
            print(f"{model} Total Latency: {total_latency_each[model]}")
            print(f"{model} Average Latency: {total_latency_each[model]/times[model]:.2f}")
        print(f"Total Output Tokens: {sum(total_tokens_each.values())}")
        print(f"Average Output Tokens: {sum(total_tokens_each.values())/total:.2f}")
        for model in Models:
            print(f"{model} Total Output Tokens: {total_tokens_each[model]}")
            print(f"{model} Average Output Tokens: {total_tokens_each[model]/times[model]:.2f}")

        print(f"已完成 {benchmark} 的 Oracle 结果，保存在：{os.path.join(outputs_dir, output_file)}")
        
