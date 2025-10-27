# This file is responsible for generating lables
# Specifically, labels are ids for different output length range.
# There is two approaches to specify the output length range:
# 1.keep the same interval
# 2.make sure every interval has same number of queries.
# NOTE: should be extendable for future requirements.

from typing import List, Dict, Any, Tuple
import os
import json
import traceback
import numpy as np

class lable_generator():
    # 需要指定index_list参数来确保移除了指定的outliers
    def __init__(self, config:Dict, index_list:List[int], model:str):
        self.benchmark = config["Data"]["benchmark"]
        self.data_dir  = os.path.join(config["Data"]["data_dir"], model)
        self.num_tokens_range_split = config["Data"]["lables"]["num_tokens_range_split"]
        self.data_list = []
        self.index_list = index_list
        self.load_datasets()
        
    def load_datasets(self):
        for idx in self.index_list:
            self.data_list.append(self.read_jsonl(idx))
    
    # 提取 "length_of_output_token_ids"
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
    
    def gen_lables(self, strategy:int) -> Tuple[Dict, np.ndarray]:
        if strategy == 0:
            range_dict, lables = self.strategy_0()
        elif strategy == 1:
            range_dict, lables = self.strategy_1()
        else:
            raise ValueError(f"Don't supports strategy:{strategy}")
        return range_dict, lables
    
    # Fixed interval.
    def strategy_0(self) -> Tuple[Dict, np.ndarray]:
        Max = max(self.data_list)
        Min = 0
        intervals = np.linspace(Min, Max, self.num_tokens_range_split + 1)
        # 为self.num_tokens_range_split个区间生成标签
        lables = np.zeros(len(self.data_list), dtype=int)
        range_dict = {}
        for i in range(self.num_tokens_range_split):
            lower_bound = intervals[i]
            upper_bound = intervals[i + 1]
            range_dict[i] = (lower_bound, upper_bound)
            for j, length in enumerate(self.data_lists):
                if lower_bound <= length < upper_bound:
                    lables[j] = i
        return range_dict, lables
    
    # Flexible interval to make sure every interval has same number of queries.
    def strategy_1(self) -> Tuple[Dict, np.ndarray]:
        Max = max(self.data_list)
        Min = 0
        
        copy = self.data_list.copy()
        sorted_lengths = sorted(copy)
        interval_size = len(sorted_lengths) // self.num_tokens_range_split
        intervals = [Min]
        for i in range(1, self.num_tokens_range_split):
            index = i * interval_size
            intervals.append(sorted_lengths[index])
        intervals.append(Max)
        
        # 为self.num_tokens_range_split个区间生成标签
        lables = np.zeros(len(self.data_list), dtype=int)
        range_dict = {}
        for i in range(self.num_tokens_range_split):
            lower_bound = intervals[i]
            upper_bound = intervals[i + 1]
            range_dict[i] = (lower_bound, upper_bound)
            for j, length in enumerate(self.data_list):
                if lower_bound <= length < upper_bound:
                    lables[j] = i
        return range_dict, lables
    
if __name__ == "__main__":
    import yaml
    config_file = "/home/ouyk/project/ICDCS/Oracle/config/router_model.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    # model = "Qwen3-0.6B-temp-0-no-thinking"
    model = "Qwen3-14B-temp-0-no-thinking"
    record = os.path.join(config["Data"]["data_dir"], model, "without_outliers.npy")
    index_list = (np.load(record)).tolist()
    
    tool = lable_generator(config, index_list, model)
    range_dict, lables = tool.gen_lables(strategy=1)
    
    print("range_dict:", range_dict)
    print("length of lables", len(lables))