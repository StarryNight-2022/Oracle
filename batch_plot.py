# 批处理指定Benchmark情况时：Oracle路由+Random路由+绘图
import os
from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Process some parameters.")
    
    parser.add_argument('--benchmark', 
                    type=str, 
                    default="GSM8K",
                    choices=["GSM8K","MMLU"],
                    help="Specify the benchmark")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    benchmark = str(args.benchmark)
    
    config_path = Path("/home/ouyk/project/ICDCS/Oracle/config/plot")
    config_file_list = []
    # 读取config目录下所有yaml文件
    for file in os.listdir(config_path):
        if file.endswith(".yaml"):
            config_file_list.append(os.path.join(config_path, file))

    os.chdir("/home/ouyk/project/ICDCS/Oracle")
    for config_file in config_file_list:
        os.system(f"python3 gen_oracle.py --config {config_file} --latency_constraint -1")
        os.system(f"python3 gen_random.py --config {config_file}")
        os.system(f"python3 plot_diagram.py --config {config_file} --benchmark {benchmark}")