from typing import List, Tuple, Any, Union
import numpy as np
import os
import json
import traceback
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter

def remove_outliers(data: List[Union[Tuple[float], Tuple[float, bool]]], m: float = 2.0) -> List[Union[Tuple[float], Tuple[float, bool]]]:
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
        
def plot_histogram(data, info_dict, title="", 
                   xlabel="lables", ylabel="times", show_stats=True, save_dir="."):
    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 频次分布图（条形图）
    counter = Counter(data)
    values = list(counter.keys())
    frequencies = list(counter.values())
    
    # 在坐标轴上绘制条形图，而不是在figure上
    ax.bar(values, frequencies, alpha=0.7, color='coral', edgecolor='black')
    ax.set_title(f'{title}', fontsize=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    
    # 添加数值标签
    for value, freq in zip(values, frequencies):
        ax.text(value, freq + max(frequencies)*0.01, f'{freq}', 
                ha='center', va='bottom', fontsize=9)
        
    # 将 info_dict 作为图例显示在图的右侧
    info_text = '\n'.join([f'Label {k}: {v[0]} - {v[1]}' for k, v in info_dict.items()])
    plt.gcf().text(0.95, 0.5, info_text, fontsize=10, va='center', ha='left',
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    
    
    plt.tight_layout()
    save_dir = os.path.join(save_dir, f"{title.replace(' ', '_')}_histogram.png")
    plt.savefig(save_dir, dpi=300, bbox_inches='tight')