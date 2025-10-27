# 首先，移除离群点；在不包含离群点的数据中定义出各个区间，最后添加一个范围用于包含离群点即。
# 移除不管是训练数据还是测试数据中的离群点，将剔除这些数据之后的index写入到一个文件中

from typing import Dict, List, Any, Tuple
import os
import json
import traceback
import time
import numpy as np

class cleaner():
    def __init__(self, data_dir:str, num_data:int, model):
        self.data_dir = os.path.join(data_dir, model)
        self.num_data = num_data
        
        self.data_lists:List[Tuple[int, int]] = []
        self.load_datasets()
    
    def load_datasets(self):
        for idx in range(1, self.num_data+1):
            self.data_lists.append((idx, self.read_jsonl(idx)))
    
    # 仅读取 "length_of_output_token_ids"
    def read_jsonl(self, idx: int):
        filepath = os.path.join(
            self.data_dir,
            f"train_{idx}.jsonl")
        try:
            with open(filepath, 'r') as file:
                line = file.readline()
                return (json.loads(line)["length_of_output_token_ids"])
        except Exception:
            print(traceback.format_exc())
            return None
        
    def remove_outliers(self, m: float = 3.0) -> List[Tuple[int, int]]:
        output_len = [item[1] for item in self.data_lists]
        mean = np.mean(output_len)
        std = np.std(output_len)
        filtered_data = [item for item in self.data_lists if abs(item[1] - mean) <= m * std]
        return filtered_data    
    
    def remove(self, threshold:float):
        record = os.path.join(
            self.data_dir,
            f"without_outliers.npy")
        
        filtered_data = self.remove_outliers(m=threshold)
        index_without_outliers = [item[0] for item in filtered_data]
        
        # 写入到文件中
        np.save(record, np.array(index_without_outliers))
        print(f"Saved the index of data without outliers to {record}")

# Example 
if __name__ == "__main__":
    model = "Qwen3-0.6B-temp-0-no-thinking"
    # model = "Qwen3-14B-temp-0-no-thinking"
    data_dir = "/home/ouyk/project/ICDCS/Oracle/input/GSM8K"
    num_data = 7473
    
    # 获取到在GSM8K数据集上每一条query对应的num_tokens
    tool = cleaner(data_dir, num_data, model)
    tool.remove(threshold=3.0)