# generate lables for latency prediction
# could refer to implementation of output_tokens.py
# TODO
# NOTE: 当前的latency划分方式导致分布稀疏，不利于精确分类。需要进行处理。

from typing import List, Dict, Any, Tuple, Optional
import os
import json
import traceback
import numpy as np

class label_generator():
    # 需要指定index_list参数来确保移除了指定的outliers
    def __init__(self, config:Dict, index_list:List[int], model:str):
        self.benchmark = config["Data"]["benchmark"]
        self.data_dir  = os.path.join(config["Data"]["data_dir"], model)
        self.latency_range_split = config["Data"]["labels"]["latency_range_split"]
        self.data_list = []
        self.index_list = index_list
        self.interval = 100
        self.manually_set = [i*self.interval for i in range(self.latency_range_split + 1)]   # 用于strategy_2的手动设置的latency interval 间隔
        self.load_datasets()
        
    def load_datasets(self):
        for idx in self.index_list:
            self.data_list.append(self.read_jsonl(idx))
    
    # 提取 "runtime"
    def read_jsonl(self, idx: int):
        filepath = os.path.join(
            self.data_dir,
            f"train_{idx}.jsonl")
        try:
            with open(filepath, 'r') as file:
                line = file.readline()
                return (json.loads(line))["runtime"]
        except Exception:
            print(traceback.format_exc())
            return None
    
    def gen_lables(self, strategy:int) -> Tuple[Dict, np.ndarray]:
        if strategy == 0:
            range_dict, lables = self.strategy_0()
        elif strategy == 1:
            range_dict, lables = self.strategy_1()
        elif strategy == 2:
            range_dict, lables = self.strategy_2() 
        else:
            raise ValueError(f"{os.path.abspath(__file__)}: Latency label generator don't supports strategy:{strategy}")
        return range_dict, lables
    
    # Fixed interval.
    def strategy_0(self) -> Tuple[Dict, np.ndarray]:
        Max = max(self.data_list)
        Min = 0
        intervals = np.linspace(Min, Max, self.latency_range_split + 1)
        # 为self.latency_range_split个区间生成标签
        lables = np.zeros(len(self.data_list), dtype=int)
        range_dict = {}
        for i in range(self.latency_range_split):
            lower_bound = intervals[i]
            upper_bound = intervals[i + 1]
            range_dict[i] = (lower_bound, upper_bound)
            for j, length in enumerate(self.data_list):
                if lower_bound <= length < upper_bound:
                    lables[j] = i
        return range_dict, lables
    
    # Flexible interval to make sure every interval has same number of queries.
    def strategy_1(self) -> Tuple[Dict, np.ndarray]:
        Max = max(self.data_list)
        Min = 0
        
        copy = self.data_list.copy()
        sorted_lengths = sorted(copy)
        interval_size = len(sorted_lengths) // self.latency_range_split
        intervals = [Min]
        for i in range(1, self.latency_range_split):
            index = i * interval_size
            intervals.append(sorted_lengths[index])
        intervals.append(Max)
        
        # 为self.latency_range_split个区间生成标签
        lables = np.zeros(len(self.data_list), dtype=int)
        range_dict = {}
        for i in range(self.latency_range_split):
            lower_bound = intervals[i]
            upper_bound = intervals[i + 1]
            range_dict[i] = (lower_bound, upper_bound)
            for j, length in enumerate(self.data_list):
                if lower_bound <= length < upper_bound:
                    lables[j] = i
        return range_dict, lables
    
    # Manually specified intervals
    def strategy_2(self) -> Tuple[Dict, np.ndarray]:
        intervals = self.manually_set
        # 为len(intervals) - 1个区间生成标签
        lables = np.zeros(len(self.data_list), dtype=int)
        range_dict = {}
        for i in range(len(intervals) - 1):
            lower_bound = intervals[i]
            upper_bound = intervals[i + 1]
            range_dict[i] = (lower_bound, upper_bound)
            for j, length in enumerate(self.data_list):
                if lower_bound <= length < upper_bound:
                    lables[j] = i
        return range_dict, lables
    
if __name__ == "__main__":
    import yaml
    from router.utils.tools import plot_histogram
    
    config_file = "/home/ouyk/project/ICDCS/Oracle/config/router_model_GSM8K.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    # model = "Qwen3-0.6B-temp-0-no-thinking"
    model = "Qwen3-14B-temp-0-no-thinking"
    record = os.path.join(config["Data"]["data_dir"], model, "without_outliers.npy")
    index_list = (np.load(record)).tolist()
    
    strategy = 0
    
    tool = label_generator(config, index_list, model)
    range_dict, lables = tool.gen_lables(strategy=strategy)
    
    print("range_dict:", range_dict)
    print([lable for lable in lables])
    print("length of lables", len(lables))
    
    plot_histogram(lables, range_dict, title=f"latency lable strategy_{strategy}", save_dir="/home/ouyk/project/ICDCS/Oracle/router/models/lables/two_steps/Hist")