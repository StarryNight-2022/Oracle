# 背景：在运行了evaluation_all之后，已经为每个模型的回答进行了正确性评价/打分。
# 用处：现在，编写该文件用于绘制对于每个LLM在指定Benchmark上的一种柱状图
# 目的：用于参考判断Oracle Router的决策是否正确。
# 细节：横轴为Latency，纵轴为每个Latency区间的querys数量。
#      每个柱子表示一个时间区间的queries，并在每个柱子上使用颜色差异区分回答正确与不正确的比例关系。
# 可选项：可以在横轴上标记出Latency_constraint的位置

import os
from typing import List, Dict, Any, Tuple, Optional, Union
import argparse
import yaml
from pathlib import Path
import json
import yaml
import traceback
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# 自定义的包
from utils.Benchmarks.benchmarks import load_dataset, parse_answer

def parse_args():
    parser = argparse.ArgumentParser(description="Process some parameters.")
    
    parser.add_argument('--benchmark', 
                        type=str, 
                        default="GSM8K",
                        choices=["GSM8K","MMLU"],
                        help="Specify the benchmark")
    
    parser.add_argument('--input_dir', 
                        type=str, 
                        default="/home/ouyk/project/ICDCS/Oracle/input",
                        help="Specify the input file storage path")
    
    parser.add_argument('--num_interval', type=int, default=50,
                    help="Specify the num_interval.")

    parser.add_argument('--latency_constraint', type=float, default=-1,
                    help="Specify the latency_constraint, default -1 means no latency constraint, Unit is seconds.")

    args = parser.parse_args()
    return args

def print_sign(benchmark: str):
    width = os.get_terminal_size().columns
    print('='*width)
    print(benchmark.center(width, '*'))

def is_folder_empty(path: Path) -> bool:
    # 使用scandir，遇到第一个条目就返回False
    with os.scandir(path) as it:
        for entry in it:
            return False  # 有至少一个条目，不为空
    return True  # 没有条目，为空

# 指定读取每一条prompt对应回复的jsonl文件
def read_profile_result(file_path: Path) -> Any:
    try:
        with open(file_path, 'r') as file:
            line = file.readline()
            return json.loads(line)
    except Exception:
        print(traceback.format_exc())
        return None
    
def add_correctness(file_path: Path, predict: str, lable: str):
    # 修改该文档中的"index"字段
    with open(file_path, 'r') as f:
        line = f.readline()
    data = json.loads(line)
    # 添加correctness字段
    data['correctness'] = (predict == lable)
    # 写回文件  
    with open(file_path, 'w') as f:
        f.write(json.dumps(data) + '\n')  # 写回文件，保持jsonl格式
        
def ensure_out_dir(path: str):
    os.makedirs(path, exist_ok=True)
    
def remove_outliers(data: List[Tuple[float, bool]], m: float = 2.0) -> List[Tuple[float, bool]]:
    latencies = [item[0] for item in data]
    mean = np.mean(latencies)
    std = np.std(latencies)
    filtered_data = [item for item in data if abs(item[0] - mean) <= m * std]
    return filtered_data
        
# 绘制柱状图
def plot_chart(model: str, data: List[Tuple[float, bool]], num_interval: int, latency_constraint: Optional[Union[float, int]], out_path: str):
    latencies = [item[0] for item in data]
    correctness = [item[1] for item in data]

    # 计算区间
    min_latency = min(latencies)
    max_latency = max(latencies)
    intervals = np.linspace(min_latency, max_latency, num_interval + 1)

    # 统计每个区间的正确和错误数量
    correct_counts = np.zeros(num_interval)
    incorrect_counts = np.zeros(num_interval)

    for latency, correct in data:
        for i in range(num_interval):
            if intervals[i] <= latency < intervals[i + 1]:
                if correct:
                    correct_counts[i] += 1
                else:
                    incorrect_counts[i] += 1
                break

    # 绘制柱状图
    bar_width = (intervals[1] - intervals[0]) * 0.4
    x = (intervals[:-1] + intervals[1:]) / 2

    plt.bar(x, correct_counts, width=bar_width, color='g', label='Correct')
    plt.bar(x, incorrect_counts, width=bar_width, bottom=correct_counts, color='r', label='Incorrect')

    # 标记Latency Constraint
    if latency_constraint is not None:
        plt.axvline(x=latency_constraint, color='b', linestyle='--', label='Latency Constraint')

    plt.xlabel('Latency (s)')
    plt.ylabel('Number of Queries')
    plt.title(f'{model}\nLatency Distribution with Correctness')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    

if __name__ == "__main__":
    runtime_dir = os.path.dirname(os.path.abspath(__file__))
    args = parse_args()
    benchmark = str(args.benchmark)
    workspace = Path(os.path.join(str(args.input_dir), benchmark))
    models:List[str] = [item for item in os.listdir(workspace) if os.path.isdir(os.path.join(workspace, item))]
    
    #--------------------------------- Load the YAML file -------------------------------
    record_path = Path(os.path.join(workspace, "evaluate_record.yaml"))
    if record_path.exists():
        record_file = record_path
    else:
        raise FileNotFoundError(f"配置文件不存在或不可读取：{record_path}")
    with open(record_file, 'r', encoding='utf-8') as f:
        record = yaml.safe_load(f)         # record: <class 'dict'>
        
    latency_constraint = args.latency_constraint
    if latency_constraint == -1:
        latency_constraint = None
        
    out_dir = os.path.join(runtime_dir, "outputs", f"{benchmark}", "plots", "bar")
    ensure_out_dir(out_dir)
    
    for model in models:
        model_path = os.path.join(workspace, model)
        if is_folder_empty(model_path):    # 目录为空
            pass
        else:                              # 目录不为空
            data: List[Tuple[float, bool]] = []
            print_sign(model)
            # 遍历*.jsonl文件
            for idx in tqdm(range(1, int(record["Models"][model]["start_idx"])+1)):
                file_path = Path(os.path.join(model_path, f"train_{idx}.jsonl"))
                profile_result = read_profile_result(file_path)
                latency = profile_result["runtime"]
                correctness = profile_result["correctness"]
                data.append((latency, correctness))

            out0 = os.path.join(out_dir, f"{model}_correctness_latency_bar_chart.png")
            # 移除异常值
            data = remove_outliers(data, m=2.0)
            # 绘制柱状图
            plot_chart(model=model,
                       data=data, 
                       num_interval=args.num_interval,
                       latency_constraint=latency_constraint,
                       out_path=out0)



