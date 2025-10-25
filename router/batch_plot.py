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
    
    parser.add_argument('--mode', 
                    type=int, 
                    required=True,
                    choices=[0, 1, 2, 3],  # 0:gen_random; 1:plot 2c diagram; 2:plot 3c diagram
                    help="Specify the running mode")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # latency_constraint = -1
    latency_constraint = 10.0
    choice = 2
    
    args = parse_args()
    benchmark = str(args.benchmark)
    
    config_path = Path("/home/ouyk/project/ICDCS/Oracle/config/plot/oracle")
    config_file_list = []
    # 读取config目录下所有yaml文件
    for file in os.listdir(config_path):
        if file.endswith(".yaml"):
            config_file_list.append(os.path.join(config_path, file))

    os.chdir("/home/ouyk/project/ICDCS/Oracle/router")
    if args.mode == 0:
        for config_file in config_file_list:
            os.system(f"python gen_random.py --config {config_file}")
    elif args.mode == 1:
        for config_file in config_file_list:
            os.system(f"python3 gen_oracle.py --config {config_file} --latency_constraint {latency_constraint} --choice {choice}")
            # os.system(f"python gen_random.py --config {config_file}")
            os.system(f"python plot_2c_diagram.py --config {config_file} --latency_constraint {latency_constraint} --choice {choice} --benchmark {benchmark}")
    elif args.mode == 2:
        for config_file in config_file_list:
            os.system(f"python plot_3c_diagram.py --config {config_file} --choice {choice} --benchmark {benchmark}")
    elif args.mode == 3:
        for config_file in config_file_list:
            os.system(f"python plot_2c_3c_combined.py --config {config_file} --choice {choice} --benchmark {benchmark}")
    else:
        raise ValueError(f"Don't supports mode_{args.mode}!")