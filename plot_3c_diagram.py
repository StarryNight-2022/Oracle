# 对应飞书中的3C部分
# step1. 需要移除Latency分布上的outlieres(将大于均值中心3倍以上的视为异常值)
# step2. 获取Latency_max作为100% Latency-Constraint数值。
# Question: 由于当前需要处理的是多个LLMs，如何定义Latency_constraint并进行异常值判定。
# Strategy: 首先对于每个模型进行异常值的判断与移除，而后再选择多个模型中的最大值即可，视为Latency_max
# step3. 划分为Latency-Constraint: [10%, 20%, ...100%] * Latency_max
# step4. 调用gen_oracle.py进行Oracle决策，基于plot_2c_diagrame.py进行Modification

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import yaml
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from utils.config import model_size
from tqdm import tqdm

from utils.tools import ensure_dir, is_folder_empty, print_sign, read_profile_result, remove_outliers

def load_summary(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def collect_random_summaries(path_or_dir: str) -> List[Dict[str, Any]]:
    p = Path(path_or_dir)
    summaries = []
    if p.is_dir():
        for file in sorted(p.glob('*.summary.json')):
            try:
                summaries.append(load_summary(str(file)))
            except Exception:
                continue
    elif p.is_file():
        # assume it's a single file containing a list or a single summary
        try:
            data = load_summary(str(p))
            if isinstance(data, list):
                summaries.extend(data)
            else:
                summaries.append(data)
        except Exception:
            pass
    return summaries


def ensure_out_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_metric_for_model(summary: Dict[str, Any], model: str):
    # Returns a tuple: (selection_pct, accuracy, avg_latency, avg_tokens)
    sel = None
    # model_selection_percent may use keys as model names
    if 'model_selection_percent' in summary:
        sel = summary['model_selection_percent'].get(model)
    # Fallback: if summary contains 'model_selection_percent' as list/dict with other keys, attempt direct access
    accuracy = summary.get('accuracy')
    avg_latency = summary.get('average_latency_per_query')
    avg_tokens = summary.get('average_output_tokens_per_query')
    # Some summaries may put model-specific averages under model_latency/model_tokens
    if (avg_latency is None or avg_tokens is None) and 'model_latency' in summary:
        ml = summary.get('model_latency', {}).get(model)
        if ml:
            if avg_latency is None:
                avg_latency = ml.get('average')
    if (avg_tokens is None) and 'model_tokens' in summary:
        mt = summary.get('model_tokens', {}).get(model)
        if mt:
            avg_tokens = mt.get('average')

    return sel, accuracy, avg_latency, avg_tokens

# TODO: Modify that method
def plot_metric(vals: List[Tuple[float, float]], xlabel: str, ylabel: str, title: str, out_path: str):
    plt.figure(figsize=(8, 6))
    x_vals = [item[0] for item in vals]
    y_vals = [item[1] for item in vals]
    # plot random points
    plt.scatter(x_vals, y_vals, color='tab:orange', s=120, marker='*', alpha=0.6, label='oracle')
    # if many points, draw a faint line connecting them (sorted by x)
    try:
        pairs = sorted([(x, y) for x, y in zip(x_vals, y_vals) if x is not None and y is not None])
        if len(pairs) > 1:
            xs, ys = zip(*pairs)
            plt.plot(xs, ys, color='tab:blue', alpha=0.3)
    except Exception:
        pass

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./config/plot/oracle/Qwen3-0.6B-no-think_AND_Deepseek-v3.2-Exp-reasoner.yaml",
                        help="Specify the config file")
    parser.add_argument('--benchmark', type=str, default="GSM8K", 
                        choices=["GSM8K","MMLU"], help='Benchmark name to load summaries for')
    parser.add_argument('--choice', type=int, default=0,
                    help="Specify the oracle strategy with latency constraint.")
    parser.add_argument('--input_dir', 
                    type=str, 
                    default="/home/ouyk/project/ICDCS/Oracle/input",
                    help="Specify the input file storage path")
    args = parser.parse_args()

    runtime_dir = os.path.dirname(os.path.abspath(__file__))
    benchmark = str(args.benchmark)
    workspace = Path(os.path.join(str(args.input_dir), benchmark))

    #--------------------------------- Load the YAML file -------------------------------
    # load config yaml to derive output paths and available models
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在或不可读取：{config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
        
    record_path = Path(os.path.join(workspace, "evaluate_record.yaml"))
    if record_path.exists():
        record_file = record_path
    else:
        raise FileNotFoundError(f"配置文件不存在或不可读取：{record_path}")
    with open(record_file, 'r', encoding='utf-8') as f:
        record = yaml.safe_load(f)         # record: <class 'dict'>

    Benchmarks: List[str] = list(config_data.get("Benchmarks", {}).keys())
    Models: List[str] = [model_info['name'] for model_info in config_data.get('Models', {}).values()]

    if args.benchmark not in Benchmarks:
        raise ValueError(f"benchmark '{args.benchmark}' 不在配置文件中: {Benchmarks}")

    out_dir = os.path.join(runtime_dir, 'outputs', args.benchmark, 'plots', 'oracle', '3c', f"strategy_{args.choice}", (str(args.config).split("/")[-1]).split(".yaml")[0])
    ensure_out_dir(out_dir)
    print(f"输出目录: {out_dir}")
    
    oracle_summary_outputs_dir = os.path.join(runtime_dir, "outputs", f"{benchmark}", "oracle", f"strategy_{args.choice}", "3c_plot")

    # TODO: Choose the max latency
    max_latency_list: List[float] = []
    for model in Models:
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
                lat = profile_result["runtime"]
                data.append((lat, True))

            # 移除异常值
            data = remove_outliers(data, m=3.0)
            
            # 选出最大Latency作为Latency_max
            latencies = [item[0] for item in data]
            latency_max = max(latencies)
            max_latency_list.append(latency_max)
        max_latency = max(max_latency_list)
    print(f"最大Latency_max: {max_latency}s")

    # 划分出 10个
    Latency_constraints: List[float] = [round((i/10)*max_latency, 2) for i in range(1, 11)]
    
    print(f"Latency_constraints: {Latency_constraints}")
    
    # 调用gen_oracle.py进行Oracle决策
    accuracy_list: List[Tuple[float, float]] = []
    latency_list: List[Tuple[float, float]] = []
    tokens_list: List[Tuple[float, float]] = []
    for idx, latency_constraint in enumerate(Latency_constraints, start=1):
        print_sign(f"Latency Constraint: {latency_constraint}s")
        os.system(f"python {os.path.join(runtime_dir, 'gen_oracle.py')} --config {str(args.config)} --latency_constraint {latency_constraint} --choice {args.choice} --outputs_dir {oracle_summary_outputs_dir}")
        # 读取summary文件，进行指标的收集与绘图
        summary_path = os.path.join(runtime_dir, "outputs", f"{benchmark}", "oracle", f"strategy_{args.choice}", "3c_plot", f"{(str(args.config)).split('/')[-1].split('.yaml')[0]}_{latency_constraint}s-latency-constraint.jsonl.summary.json")
        summary = load_summary(summary_path)
        accuracy = summary["accuracy"]
        latency = summary["average_latency_per_query"]
        tokens = summary["average_output_tokens_per_query"]
        accuracy_list.append((idx/10, accuracy))
        latency_list.append((idx/10, latency))
        tokens_list.append((idx/10, tokens))
        
    # 进行绘图
    # Plot 1: accuracy vs max_latency%
    out1 = os.path.join(out_dir, f"accuracy_vs_latency_constraint.png")
    plot_metric(accuracy_list,
                xlabel='Latency constraint(percentage of max_latency)', 
                ylabel='accuracy',
                title=f'{args.benchmark}\nAccuracy vs Latency constraint(percentage of max_latency:{max_latency}s)', 
                out_path=out1)

    # Plot 2: average latency per query vs max_latency%
    out2 = os.path.join(out_dir, f"latency_vs_latency_constraint.png")
    plot_metric(latency_list,
                xlabel='Latency constraint(percentage of max_latency)', 
                ylabel='avg latency (s)',
                title=f'{args.benchmark}\nAverage Latency vs Latency constraint(percentage of max_latency:{max_latency}s)', 
                out_path=out2)

    # Plot 3: average output tokens per query vs max_latency%
    out3 = os.path.join(out_dir, f"tokens_vs_latency_constraint.png")
    plot_metric(tokens_list,
                xlabel='Latency constraint(percentage of max_latency)', 
                ylabel='avg output tokens',
                title=f'{args.benchmark}\nAverage Output Tokens vs Latency constraint(percentage of max_latency:{max_latency}s)', 
                out_path=out3)

    print('Plots saved:')
    print(out1)
    print(out2)
    print(out3)

if __name__ == '__main__':
    main()
