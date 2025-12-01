# 该文件负责实现对2_step_router或是end2end_router生成的决策进行评价，
# 参考标准为使用Oracle策略生成的决策，评测指标暂时定义为正确率。

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json

from router.utils.oracle_router import Oracle
from router.utils.tools import read_jsonl
from router.models.router.config import MODEL_IDS
from tqdm import tqdm

# This class is responsible for managing the inputs for oracle router and the real router.
class Evaluator:
    def __init__(self, config:Dict, model_list:List[str], latency_constraint:float):
        self.config = config
        self.model_list = model_list
        self.latency_constraint = latency_constraint
        
        self.inputs:List[Dict[str, Any]] = []
        self.iter_data:List = []
        self.Oracle_Judge = Oracle(config=None, 
                                   model_list=self.model_list)
        self.prepare_inputs()
        self.gen_oracle()
        self.access_count = 0
    
    def prepare_inputs(self):
        for idx in range(self.config["Data"]["num_data"]):
            input:Dict[str, Any] = {}
            for model in self.model_list:
                model_id = MODEL_IDS[model]
                input[model] = read_jsonl(self.config, self.config["Data"]["benchmark"], f"model_{model_id}", idx+1)
            self.inputs.append(input)    
    
    # generate oracle choice
    def gen_oracle(self):
        for input in self.inputs:
            choice = self.Oracle_Judge.get_oracle(input, self.latency_constraint, choice=2)
            self.iter_data.append((input[self.model_list[0]]["prompt"], choice))
    
    # 为外部使用构造迭代器
    def __iter__(self):
        self.access_count += 1
        for item in self.iter_data:
            yield item
    
    # 迭代器的必要属性
    def __len__(self):
        return len(self.iter_data)

# Example
if __name__ == "__main__":
    import yaml
    config_file = "/home/ouyk/project/ICDCS/Oracle/config/router_model_GSM8K.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    # router/models/router/2_step.py
    from router.models.router.two_step_v1 import Router
    router = Router(config=config)
        
    model_list:List[str] = ["Qwen3-0.6B-temp-0-no-thinking", "Qwen3-14B-temp-0-no-thinking"]
    latency_constraint = 10

    # 记录对于每一条prompt，oracle_router与real_router的决策输出。
    record_list:List[Tuple] = []
    correct_count = 0
    
    evaluator = Evaluator(config=config, model_list=model_list, latency_constraint=latency_constraint)
    for idx, item in tqdm(enumerate(evaluator, start=1), total=len(evaluator)):
        prompt, oracle_choice = item
        
        choice = router.route(prompt=prompt,
                              model_name_list=model_list,
                              latency_constraint=latency_constraint)
        # print(f"idx: {idx}, oracle_choice: {oracle_choice['model']}, choice: {choice}")
        
        record_list.append((idx, prompt, oracle_choice["model"], choice))
        if oracle_choice["model"] == choice:
            correct_count += 1
    
    print(f"accuracy: {correct_count}/{idx} = {(correct_count/idx)*100:.2f}%")