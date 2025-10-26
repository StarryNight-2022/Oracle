# Implementation for getting inputs from "Qwen3-0.6B num of output tokens"
# 可以直接使用此前profile的结果，只需要load此前的*.jsonl文件
from typing import Dict, List, Any
import os
import json
import traceback

class offline_tokens():
    def __init__(self, config: Dict):
        self.benchmark = config["Data"]["benchmark"]
        self.data_dir  = config["Data"]["data_dir"]
        self.num_data  = config["Data"]["num_data"]
        self.access_count = 0
        self.data_lists = []
        self.load_datasets()
    
    def __iter__(self):
        self.access_count += 1
        for item in self.data_lists:
            yield item
    
    def __len__(self):
        return len(self.datasets)
    
    def load_datasets(self):
        for idx in range(1, self.num_data+1):
            self.data_lists.append(self.read_jsonl(idx))
    
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
    
    # 获取到在GSM8K数据集上每一条query对应的num_tokens
    for data in offline_tokens(config):
        print(data)