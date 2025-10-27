# Implementation for getting inputs from "Qwen3-0.6B num of output tokens"
# 可以直接使用此前profile的结果，只需要load此前的*.jsonl文件
from typing import Dict, List, Any
import os
import json
import traceback
import numpy as np

class offline_tokens():
    # 需要指定index_list参数来确保移除了指定的outliers
    def __init__(self, config: Dict, index_list:List[int], model:str):
        self.benchmark = config["Data"]["benchmark"]
        self.num_data  = config["Data"]["num_data"]
        self.data_dir  = os.path.join(config["Data"]["data_dir"], model)
        self.index_list = index_list
        self.access_count = 0
        self.data_list = []
        self.load_datasets()
    
    def __iter__(self):
        self.access_count += 1
        for item in self.data_list:
            yield item
    
    def __len__(self):
        return len(self.datasets)
    
    def load_datasets(self):
        for idx in self.index_list:
            self.data_list.append(self.read_jsonl(idx))
    
    # 每次装载一个结果，选出"length_of_output_token_ids"
    def read_jsonl(self, idx: int):
        filepath = os.path.join(
            self.data_dir,
            f"train_{idx}.jsonl")
        try:
            with open(filepath, 'r') as file:
                line = file.readline()
                return (json.loads(line))["length_of_output_token_ids"]
        except Exception:
            print(traceback.format_exc())
            return None
    
if __name__ == "__main__":
    import yaml
    config_file = "/home/ouyk/project/ICDCS/Oracle/config/router_model.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    model_A = "Qwen3-0.6B-temp-0-no-thinking"
    model_B = "Qwen3-14B-temp-0-no-thinking"
    record = os.path.join(config["Data"]["data_dir"], model_B, "without_outliers.npy")
    index_list = (np.load(record)).tolist()
    
    # 获取到在GSM8K数据集上每一条query对应的num_tokens
    for data in offline_tokens(config, index_list, model_A):
        print(data)