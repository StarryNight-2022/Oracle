from typing import List, Tuple, Any
import numpy as np
import os
import json
import traceback
from pathlib import Path

def remove_outliers(data: List[Tuple[float, bool]], m: float = 2.0) -> List[Tuple[float, bool]]:
    latencies = [item[0] for item in data]
    mean = np.mean(latencies)
    std = np.std(latencies)
    filtered_data = [item for item in data if abs(item[0] - mean) <= m * std]
    return filtered_data

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