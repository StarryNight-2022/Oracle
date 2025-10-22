#!/usr/bin/env python3
"""
读取 *.summary.json 文件并绘制三张图：
- accuracy vs model_selection_percentage
- average_latency_per_query vs model_selection_percentage
- average_output_tokens_per_query vs model_selection_percentage

支持两类数据：
- oracle: 单组 summary（单个文件）
- random: 多组 summary（目录或多个文件），将作为多个点绘制

使用示例：
python plot_diagram.py --oracle path/to/oracle.summary.json --random_dir path/to/random_summaries/ --model "Qwen3-0.6B" --out_dir outputs/plots

"""
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import yaml
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

model_size = {
    "Deepseek-v3.2-Exp-temp-0-chat": 685,
    "Deepseek-v3.2-Exp-temp-0-reasoner": 685,
    "GPT-4o-mini-temp-0": 8,
    "o4-mini-temp-1": 0,
    "Qwen3-0.6B-temp-0-en-thinking": 0.6,
    "Qwen3-0.6B-temp-0-no-thinking": 0.6,
    "Qwen3-14B-temp-0-en-thinking": 14,
    "Qwen3-14B-temp-0-no-thinking": 14,
}

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


def plot_metric(x_vals: List[float], y_vals: List[float], oracle_x: float, oracle_y: float, xlabel: str, ylabel: str, title: str, out_path: str):
    plt.figure(figsize=(8, 6))
    # plot random points
    plt.scatter(x_vals, y_vals, color='tab:blue', alpha=0.6, label='random')
    # if many points, draw a faint line connecting them (sorted by x)
    try:
        pairs = sorted([(x, y) for x, y in zip(x_vals, y_vals) if x is not None and y is not None])
        if len(pairs) > 1:
            xs, ys = zip(*pairs)
            plt.plot(xs, ys, color='tab:blue', alpha=0.3)
    except Exception:
        pass

    # plot oracle as a distinct marker
    if oracle_x is not None and oracle_y is not None:
        plt.scatter([oracle_x], [oracle_y], color='tab:orange', s=120, marker='*', label='oracle')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

# TODO: 绘制表格
def plot_chart(data: pd.DataFrame, out_path: str):
    # 使用styler来渲染表格，它会自动处理索引
    fig, ax = plt.subplots(figsize=(8, len(data)*0.5 + 1))
    ax.axis('off')
    
    # 使用pandas的styler来创建表格
    table = ax.table(cellText=np.vstack([data.columns, data.values]),
                     rowLabels=['Models (small->big)'] + data.index.tolist(),
                     cellLoc='center',
                     loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./config/plot/oracle/Qwen3-0.6B-no-think_AND_Deepseek-v3.2-Exp-reasoner.yaml",
                        help="Specify the config file")
    parser.add_argument('--benchmark', type=str, default="GSM8K", 
                        choices=["GSM8K","MMLU"], help='Benchmark name to load summaries for')
    parser.add_argument('--latency_constraint', type=float, default=-1, 
                        help='Latency constraint used when generating summaries (use -1 for none)')
    parser.add_argument('--choice', type=int, default=0,
                    help="Specify the oracle strategy with latency constraint.")
    args = parser.parse_args()

    # load config yaml to derive output paths and available models
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在或不可读取：{config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)

    Benchmarks: List[str] = list(config_data.get("Benchmarks", {}).keys())
    Models: List[str] = [model_info['name'] for model_info in config_data.get('Models', {}).values()]

    if args.benchmark not in Benchmarks:
        raise ValueError(f"benchmark '{args.benchmark}' 不在配置文件中: {Benchmarks}")

    # 挑选两个模型中model_size最大的那个
    model_name = sorted(Models, key=lambda m: model_size.get(m, 0), reverse=True)[0] if Models else None
    print(f"选择模型 '{model_name}' 进行绘图")
    if model_name is None:
        raise ValueError('未指定模型且配置文件中没有模型信息')
    if model_name not in Models:
        raise ValueError(f"指定的模型 '{model_name}' 不在配置文件 Models 列表中: {Models}")

    # determine output directory: prefer explicit arg, otherwise outputs/<benchmark>/plots
    runtime_dir = os.path.dirname(os.path.abspath(__file__))

    out_dir = os.path.join(runtime_dir, 'outputs', args.benchmark, 'plots', 'oracle', f"strategy_{args.choice}", (str(args.config).split("/")[-1]).split(".yaml")[0])
    ensure_out_dir(out_dir)
    print(f"输出目录: {out_dir}")
    latency_constraint = args.latency_constraint
    if latency_constraint == -1:
        latency_constraint = None

    # construct oracle summary path using same naming rule as gen_oracle.py
    oracle_outputs_dir = os.path.join(runtime_dir, "outputs", args.benchmark, "oracle", f"strategy_{args.choice}")
    config_basename = (str(args.config).split("/")[-1]).split(".yaml")[0]
    if latency_constraint is None:
        oracle_output_file = config_basename + "_no-latency-constraint" + ".jsonl"
    else:
        oracle_output_file = config_basename + f"_{latency_constraint}s-latency-constraint" + ".jsonl"
    oracle_summary_path = os.path.join(oracle_outputs_dir, oracle_output_file + ".summary.json")

    # load oracle summary if exists
    if not os.path.exists(oracle_summary_path):
        raise FileNotFoundError(f"找不到 oracle summary 文件: {oracle_summary_path}")
    oracle_summary = load_summary(oracle_summary_path)
    oracle_sel, oracle_acc, oracle_lat, oracle_tokens = get_metric_for_model(oracle_summary, model_name)

    # collect random summaries using same naming rule as gen_random.py
    random_outputs_dir = os.path.join(runtime_dir, "outputs", args.benchmark, "random")
    random_summaries = []
    # percentages 0,10,...,100
    for percentage in range(0, 101, 10):
        random_output_file = config_basename + f"_random_{percentage}percent" + ".jsonl"
        summary_path = os.path.join(random_outputs_dir, random_output_file + ".summary.json")
        if os.path.exists(summary_path):
            try:
                random_summaries.append(load_summary(summary_path))
            except Exception:
                print(f"警告: 无法读取 {summary_path}")
        else:
            print(f"提示: 未找到 {summary_path}, 跳过")
    random_x = []
    random_acc = []
    random_lat = []
    random_tokens = []
    for s in random_summaries:
        sel, acc, lat, tok = get_metric_for_model(s, model_name)
        # only include points that have selection percentage and the specific y metric
        if sel is not None:
            random_x.append(sel)
            random_acc.append(acc if acc is not None else float('nan'))
            random_lat.append(lat if lat is not None else float('nan'))
            random_tokens.append(tok if tok is not None else float('nan'))
            
    # Chart: construct a pd.DataFrame to storage data of a chart
    # 基于model_size对Models排序，由小到大
    out0 = os.path.join(out_dir, f"summary_chart.png")
    models_sort_by_size = sorted(Models, key=lambda m: model_size.get(m, 0))
    models_sort_by_size.append("Oracle")
    chart = pd.DataFrame(columns=['Accuracy', 'Avg. Latency', 'Avg. Tokens', 'Selected'], index=models_sort_by_size)
    # 填充数据，最后一个模型特殊处理
    temp = str(config_data["Models"]["model_0"]["profile_result"][args.benchmark])
    evaluate_record_path = os.path.join(temp, '..', 'evaluate_record.yaml')
    with open(evaluate_record_path, 'r', encoding='utf-8') as f:
        evaluate_record = yaml.safe_load(f)
        for m in Models:
            sel, _, _, _ = get_metric_for_model(oracle_summary, m)
            acc = evaluate_record["Models"][m]["accuracy"]
            lat = evaluate_record["Models"][m]["avg_runtime"]
            tok = evaluate_record["Models"][m]["avg_tokens"]
            chart.at[m, 'Accuracy'] = f"{acc}" if acc is not None else "N/A"
            chart.at[m, 'Avg. Latency'] = f"{lat}" if lat is not None else "N/A"
            chart.at[m, 'Avg. Tokens'] = f"{tok}" if tok is not None else "N/A"
            chart.at[m, 'Selected'] = f"{sel:.2f}%" if sel is not None else "N/A"
    # Oracle行
    chart.at["Oracle", 'Accuracy'] = f"{(float(oracle_acc)*100):.2f}%" if oracle_acc is not None else "N/A"
    chart.at["Oracle", 'Avg. Latency'] = f"{(float(oracle_lat)):.2f} seconds" if oracle_lat is not None else "N/A"
    chart.at["Oracle", 'Avg. Tokens'] = f"{(float(oracle_tokens)):.2f} tokens" if oracle_tokens is not None else "N/A"
    chart.at["Oracle", 'Selected'] = "100%"
    plot_chart(chart, out_path=out0)

    # Plot 1: accuracy vs selection%
    out1 = os.path.join(out_dir, f"accuracy_vs_selection_{model_name}.png")
    plot_metric(random_x, random_acc, oracle_sel, oracle_acc,
                xlabel='model selection percentage (%)', ylabel='accuracy',
                title=f'Accuracy vs {model_name} Selection % ({args.benchmark})', out_path=out1)

    # Plot 2: average latency per query vs selection%
    out2 = os.path.join(out_dir, f"latency_vs_selection_{model_name}.png")
    plot_metric(random_x, random_lat, oracle_sel, oracle_lat,
                xlabel='model selection percentage (%)', ylabel='avg latency (s)',
                title=f'Average Latency vs {model_name} Selection % ({args.benchmark})', out_path=out2)

    # Plot 3: average output tokens per query vs selection%
    out3 = os.path.join(out_dir, f"tokens_vs_selection_{model_name}.png")
    plot_metric(random_x, random_tokens, oracle_sel, oracle_tokens,
                xlabel='model selection percentage (%)', ylabel='avg output tokens',
                title=f'Average Output Tokens vs {model_name} Selection % ({args.benchmark})', out_path=out3)

    print('Plots saved:')
    print(out1)
    print(out2)
    print(out3)
    print('chart saved:')
    print(out0)


if __name__ == '__main__':
    main()
# 该文件负责绘制三个图像