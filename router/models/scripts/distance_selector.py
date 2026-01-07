# 首先将所有数据按照output_length<=500进行滤波
# 每次计算距离时，固定x轴(output_length_a)不变，计算对应点在y轴(output_length_b)方向的距离，将这些距离记录下来，同时记录这个点的index
# 筛选出距离最远的10个点对与距离最近的10个点对
from typing import Dict, List, Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import os
import yaml
import numpy as np
import json
from router.models.scripts.dataset.our_datasets import prepare_training_data, train_test_split, data_require_template, data_choice

# 主函数
def main(embedding_model:str):
    config_file = "/home/ouyk/project/ICDCS/Oracle/config/router_model_GSM8K.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    model_A = "Qwen3-0.6B-temp-0-no-thinking"   # use its embedding as inputs
    model_B = "Qwen3-14B-temp-0-no-thinking"    # use its output_length as lables
    record_a = os.path.join(config["Data"]["data_dir"], model_A, "without_outliers.npy")
    record_b = os.path.join(config["Data"]["data_dir"], model_B, "without_outliers.npy")
    index_list_a = np.load(record_a).tolist()
    index_list_b = np.load(record_b).tolist()
    index_list = list(set(index_list_a) & set(index_list_b)) # 取交集
    data_require = data_require_template.copy()
    data_require["output_tokens_a"] = data_choice.X
    data_require["output_tokens_b"] = data_choice.X
    
    # 准备数据 lable_strategy: 0->Fixed Intervals, 1->Flexible Intervals
    X, _, range_dict = prepare_training_data(config, index_list, model_A, model_B, embedding_model, data_require=data_require, lable_strategy=2)
    
    output_tokens_a = X["output_tokens_a"]
    output_tokens_b = X["output_tokens_b"]
    
    # 绘制散点图
    plt.scatter(output_tokens_a, output_tokens_b, alpha=0.5)
    plt.title('Scatter Plot of Output Tokens')
    plt.xlabel('Output Tokens A')
    plt.ylabel('Output Tokens B')
    plt.grid(True)
    plt.savefig("scatter_plot.png")
    plt.close()
    
    point_dict:List[Dict[str, int]] = []
    for idx, item in enumerate(index_list, start=0):
        point_dict.append({
            "output_tokens_a": output_tokens_a[idx].item(),
            "raw_index": item,
            "idx": idx,
        })
        
    # 基于"output_tokens_a"进行排序
    point_dict.sort(key=lambda x: x["output_tokens_a"])
    
    # 统计出 "output_tokens_a"每个数值出现的次数
    value_count:Dict[int, int] = {}
    for item in point_dict:
        value_count[item["output_tokens_a"]] = value_count.get(item["output_tokens_a"], 0) + 1
        
    # 绘制 "output_tokens_a" 每个数值出现的次数的直方图
    plt.bar(value_count.keys(), value_count.values(), alpha=0.5)
    plt.title('Histogram of Output Tokens A')
    plt.xlabel('Output Tokens A')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig("histogram_output_tokens_a.png")
    plt.close()
    
    # 建立各个"output_tokens_a"数值对应的index列表
    value_index_dict:Dict[int, List[int]] = {}
    for item in point_dict:
        value_index_dict[item["output_tokens_a"]] = value_index_dict.get(item["output_tokens_a"], []) + [item["idx"]]
    
    distance_record:List[Tuple[int, int, int]] = []  # 需要包括 distance, 两个点的"idx"
    # 基于value_index_dict，计算每个"output_tokens_a"数值对应的"output_tokens_b"的距离
    for i in value_index_dict.keys():
        index_list = value_index_dict[i]
        if len(index_list) <= 1:
            continue
        
        for j in range(len(index_list)):
            for k in range(j+1, len(index_list)):
                idx_a = index_list[j]
                idx_b = index_list[k]
                distance = abs(output_tokens_b[idx_a].item() - output_tokens_b[idx_b].item())
                distance_record.append((distance, idx_a, idx_b))
                
    # 按照distance进行排序
    distance_record.sort(key=lambda x: x[0])
    
    # 筛选出距离最远的10个点对与距离最近的10个点对
    closest_10 = distance_record[:10]
    farthest_10 = distance_record[-10:]
    
    # 使用idx_a与idx_b从point_dict中获取对应的raw_index
    def get_raw_indices(idx_a:int, idx_b:int) -> Tuple[int, int]:
        raw_index_a = point_dict[idx_a]["raw_index"]
        raw_index_b = point_dict[idx_b]["raw_index"]
        return raw_index_a, raw_index_b
    
    print("Closest 10 point pairs:")
    nearest_set = []
    farthest_set = []
    for distance, idx_a, idx_b in closest_10:
        raw_index_a, raw_index_b = get_raw_indices(idx_a, idx_b)
        nearest_set.append((raw_index_a, raw_index_b))
        print(f"Distance: {distance}, Raw Indices: ({raw_index_a}, {raw_index_b})")

    print("Farthest 10 point pairs:")
    for distance, idx_a, idx_b in farthest_10:
        raw_index_a, raw_index_b = get_raw_indices(idx_a, idx_b)
        farthest_set.append((raw_index_a, raw_index_b))
        print(f"Distance: {distance}, Raw Indices: ({raw_index_a}, {raw_index_b})")
        
    # file_dir = "/home/ouyk/project/ICDCS/Oracle/input/A100/Raw/GSM8K/Qwen3-0.6B-temp-0-no-thinking"
    file_dir = "/home/ouyk/project/ICDCS/Oracle/input/A100/Raw/GSM8K/Qwen3-14B-temp-0-no-thinking"
    
    print(120*"-")
    print(58*"-"+"farthest"+58*"-")
    print(120*"-")
    
    for pair in farthest_set:
        response = []
        with open(os.path.join(file_dir, f"train_{pair[0]}.jsonl")) as f:
            data = json.load(f)
            response.append(data["full_response"])
        
        with open(os.path.join(file_dir, f"train_{pair[1]}.jsonl")) as f:
            data = json.load(f)
            response.append(data["full_response"])
        
        print(f"pair index:{pair[0]}, {pair[1]}")    
        print(response[0], "\n", "*"*80, "\n", response[1], "\n")
        print("-"*80, "\n", "-"*80)
        
    print(120*"-")
    print(58*"-"+"nearest"+58*"-")
    print(120*"-")
    
    for pair in nearest_set:
        response = []
        with open(os.path.join(file_dir, f"train_{pair[0]}.jsonl")) as f:
            data = json.load(f)
            response.append(data["full_response"])
        
        with open(os.path.join(file_dir, f"train_{pair[1]}.jsonl")) as f:
            data = json.load(f)
            response.append(data["full_response"])
        
        print(f"pair index:{pair[0]}, {pair[1]}")
        print(response[0], "\n", "*"*80, "\n", response[1], "\n")
        print("-"*80, "\n", "-"*80)


if __name__ == '__main__':
    embedding_model="Qwen3-Embeddings-0.6B"
    # embedding_model="bert-embedding"
    main(embedding_model)