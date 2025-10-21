import os
import json
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from pathlib import Path
import argparse
import yaml
import traceback

# 自定义的包
from utils.Benchmarks.benchmarks import load_dataset
from utils.knn_router import KNNRouter


def parse_args():
    parser = argparse.ArgumentParser(description="Process some parameters.")

    parser.add_argument('--config', type=str, default="./config/Qwen3-0.6B-no-think_AND_Qwen3-14B-no-think.yaml",
                        help="Specify the config file")

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

    # ---------------------------- Initialize KNN router class ------------------------
    knn_router = KNNRouter(k=5, random_seed=42, metrics=["minkowski","cosine"])



    # ------------------------------------ Main Iteration ---------------------------------
    results: Dict[str, Any] = {}
    # Iteration for benchmarks
    for benchmark in Benchmarks:
        # print_sign(benchmark)
        outputs_dir = os.path.join(runtime_dir, "outputs", f"{benchmark}", "knn")
        ensure_dir(outputs_dir)

        results = knn_router.run_full_evaluation(
            models=config_data["Models"],
            benchmark_path=config_data["Benchmarks"][benchmark],
            benchmark = benchmark,
            max_samples=10000,
            test_size=0.30
        )

        base_name = ((str(args.config)).split('/')[-1]).split('.yaml')[0]
        metric_tag = knn_router.metrics[0] if knn_router.metrics else 'minkowski'
        output_file = f"{base_name}_knn_k{knn_router.k}_{metric_tag}_no-latency"

        # Pull aggregates from results
        times = results.get('times', {})
        total_latency_each = results.get('total_latency_each', {})
        total_tokens_each = results.get('total_tokens_each', {})

        Models = list(times.keys()) if times else ['Small', 'Large']
        total = int(sum(times.get(m, 0) for m in Models)) if times else results.get('test_samples', 0)
        accuracy = float(results.get('knn_achieved_correctness', 0.0))

        summary = {
            "benchmark": benchmark,
            "config": str(args.config),
            "total_queries": total,
            "model_selection_percent": {},
            "accuracy": accuracy,
            "total_latency": float(sum(total_latency_each.values())) if total_latency_each else 0.0,
            "average_latency_per_query": (
                        float(sum(total_latency_each.values())) / total) if total > 0 and total_latency_each else None,
            "model_latency": {},
            "total_output_tokens": int(sum(total_tokens_each.values())) if total_tokens_each else 0,
            "average_output_tokens_per_query": (
                        float(sum(total_tokens_each.values())) / total) if total > 0 and total_tokens_each else None,
            "model_tokens": {}
        }

        for model in Models:
            pct = (times.get(model, 0) / total * 100.0) if total > 0 else None
            avg_latency = (total_latency_each.get(model, 0.0) / times.get(model, 0)) if times.get(model,
                                                                                                  0) > 0 else None
            avg_tokens = (total_tokens_each.get(model, 0) / times.get(model, 0)) if times.get(model, 0) > 0 else None
            summary["model_selection_percent"][model] = pct
            summary["model_latency"][model] = {
                "total": float(total_latency_each.get(model, 0.0)),
                "average": None if avg_latency is None else float(avg_latency)
            }
            summary["model_tokens"][model] = {
                "total": int(total_tokens_each.get(model, 0)),
                "average": None if avg_tokens is None else float(avg_tokens)
            }

        summary_path = os.path.join(outputs_dir, output_file + ".summary.json")
        try:
            with open(summary_path, 'w', encoding='utf-8') as sf:
                json.dump(summary, sf, ensure_ascii=False, indent=2)
            print(f"Summary saved to: {summary_path}")
        except Exception:
            print("写入 summary JSON 时发生错误:\n", traceback.format_exc())
