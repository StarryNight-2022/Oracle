# 绘制对于同一个数据集，两个模型对于同一个question的runtime的散点图.
# 散点图上每一个点代表一个question，数值为(model_A的runtime, model_B的runtime)
#!/usr/bin/env python3
"""
使用示例：
"""
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import yaml
import traceback

import matplotlib.pyplot as plt

# 自定义的包
from utils.Benchmarks.benchmarks import load_dataset

def ensure_out_dir(path: str):
    os.makedirs(path, exist_ok=True)
    
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
    
def print_sign(benchmark: str):
    width = os.get_terminal_size().columns
    print('='*width)
    print(benchmark.center(width, '*'))


def save_scatter_plot(x_vals, y_vals, model_names, benchmark, plot_path, dpi=300):
    """Generate and save a scatter plot for two model runtimes.

    Args:
        x_vals (List[float]): runtimes for model A (x-axis).
        y_vals (List[float]): runtimes for model B (y-axis).
        model_names (Tuple[str, str] | List[str]): (nameA, nameB).
        benchmark (str): benchmark name used in the title.
        plot_path (str): full path where the PNG will be saved.
        dpi (int): image DPI for saving (default 300).
    """
    # Basic validation
    if not isinstance(x_vals, (list, tuple)) or not isinstance(y_vals, (list, tuple)):
        raise TypeError("x_vals and y_vals must be lists or tuples of numbers")
    if len(x_vals) == 0 or len(y_vals) == 0:
        raise ValueError("x_vals and y_vals must be non-empty")
    if len(x_vals) != len(y_vals):
        raise ValueError("x_vals and y_vals must have the same length")

    name_x = model_names[0]
    name_y = model_names[1]

    # Create figure with given dpi
    fig, ax = plt.subplots(figsize=(12, 12), dpi=dpi)
    ax.scatter(x_vals, y_vals, alpha=0.6)
    ax.set_xlabel(f"{name_x} Runtime (s)")
    ax.set_ylabel(f"{name_y} Runtime (s)")
    ax.set_title(f"Scatter Plot of Runtimes on {benchmark}")
    ax.grid(True, which="both", ls="--", linewidth=0.5)

    # diagonal y=x line and limits
    combined = list(x_vals) + list(y_vals)
    minv = min(combined)
    maxv = max(combined)
    # if all values identical, expand a little to make the plot visible
    if minv == maxv:
        minv = minv * 0.9 if minv != 0 else -0.1
        maxv = maxv * 1.1 if maxv != 0 else 0.1

    ax.plot([minv, maxv], [minv, maxv], color='red', linestyle='--', label='y=x')
    ax.set_xlim(minv, maxv)
    ax.set_ylim(minv, maxv)
    ax.legend()
    fig.tight_layout()

    # Ensure parent dir exists
    ensure_out_dir(os.path.dirname(plot_path))
    fig.savefig(plot_path, dpi=dpi)
    plt.close(fig)


def remove_outliers(x_vals, y_vals, z_threshold=3.0):
    """Remove outliers by z-score on x and y separately.

    Returns filtered (x_vals, y_vals) lists and number of removed points.
    """
    import statistics

    if len(x_vals) == 0:
        return x_vals, y_vals, 0

    # compute mean and stdev for x and y
    mean_x = statistics.mean(x_vals)
    mean_y = statistics.mean(y_vals)
    stdev_x = statistics.pstdev(x_vals) if len(x_vals) > 1 else 0.0
    stdev_y = statistics.pstdev(y_vals) if len(y_vals) > 1 else 0.0

    if stdev_x == 0 and stdev_y == 0:
        # nothing to remove
        return x_vals, y_vals, 0

    filtered_x = []
    filtered_y = []
    removed = 0
    for xv, yv in zip(x_vals, y_vals):
        zx = abs((xv - mean_x) / stdev_x) if stdev_x > 0 else 0.0
        zy = abs((yv - mean_y) / stdev_y) if stdev_y > 0 else 0.0
        if zx > z_threshold or zy > z_threshold:
            removed += 1
            continue
        filtered_x.append(xv)
        filtered_y.append(yv)

    return filtered_x, filtered_y, removed

# 示例：python plot_scatter.py --benchmark GSM8K --remove-outliers --z-threshold 3
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./config/plot/scatter/Qwen3-0.6B-no-think_AND_Qwen3-14B-no-think.yaml",
                        help="Specify the config file")
    parser.add_argument('--benchmark', type=str, default="GSM8K", 
                        choices=["GSM8K","MMLU","Chatbot-Arena"], help='Benchmark name to load summaries for')
    parser.add_argument('--remove-outliers', action='store_true', help='Also generate plot with outliers removed (z-score)')
    parser.add_argument('--z-threshold', type=float, default=3.0, help='Z-score threshold for outlier removal')

    args = parser.parse_args()
    
    benchmark = args.benchmark

    # load config yaml to derive output paths and available models
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在或不可读取：{config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)

    Benchmarks: List[str] = list(config_data.get("Benchmarks", {}).keys())
    Models: List[str] = [model_info['name'] for model_info in config_data.get('Models', {}).values()]

    if benchmark not in Benchmarks:
        raise ValueError(f"benchmark '{benchmark}' 不在配置文件中: {Benchmarks}")

    # determine output directory: prefer explicit arg, otherwise outputs/<benchmark>/plots
    runtime_dir = os.path.dirname(os.Path.join(os.path.abspath(__file__), ".."))

    out_dir = os.path.join(runtime_dir, 'outputs', benchmark, 'plots', 'scatter', (str(args.config).split("/")[-1]).split(".yaml")[0])
    ensure_out_dir(out_dir)
    print(f"输出目录: {out_dir}")

    # 读取原始数据
    dataset = load_dataset(benchmark, config_data["Benchmarks"][benchmark])
    print_sign(benchmark)
    
    scatter_points: List[Dict[float, float]] = []

    # 读取每一个query的runtime
    for idx, query in enumerate(dataset, start=0):
        if benchmark == "Chatbot-Arena":
            query_idx = query["index"]
        else:
            query_idx = idx + 1
        model1_runtime = (read_jsonl(config_data, benchmark, "model_0", query_idx))["runtime"]
        model2_runtime = (read_jsonl(config_data, benchmark, "model_1", query_idx))["runtime"]
        scatter_points.append({Models[0]:model1_runtime, Models[1]:model2_runtime})
        
    # 绘制散点图：准备原始数据
    x_vals_all = [point[Models[0]] for point in scatter_points]
    y_vals_all = [point[Models[1]] for point in scatter_points]

    # 1) 保存包含所有点的图
    plot_path_with = os.path.join(out_dir, f"{Models[0]}_vs_{Models[1]}_scatter_with_outliers.png")
    save_scatter_plot(x_vals_all, y_vals_all, (Models[0], Models[1]), benchmark + ' (with outliers)', plot_path_with, dpi=300)

    # 2) 生成并保存去除离群点后的图
    x_filtered, y_filtered, removed = remove_outliers(x_vals_all, y_vals_all, z_threshold=args.z_threshold)
    plot_path_without = os.path.join(out_dir, f"{Models[0]}_vs_{Models[1]}_scatter_without_outliers.png")

    if args.remove_outliers:
        # only remove when user requested — otherwise keep identical to original
        save_scatter_plot(x_filtered, y_filtered, (Models[0], Models[1]), benchmark + f" (without outliers, removed={removed})", plot_path_without, dpi=300)
    else:
        # still save a second copy for comparison, but it's identical to the first
        save_scatter_plot(x_vals_all, y_vals_all, (Models[0], Models[1]), benchmark + ' (without outliers)', plot_path_without, dpi=300)

    print(f"Saved plots:\n - {plot_path_with}\n - {plot_path_without}")

if __name__ == '__main__':
    main()