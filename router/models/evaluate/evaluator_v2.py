# 该文件负责实现对2_step_router或是end2end_router生成的决策进行评价，
# 参考标准为使用Oracle策略生成的决策，评测指标暂时定义为正确率。

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json

from router.utils.oracle_router import Oracle
from router.utils.tools import read_jsonl

# This class is responsible for managing the inputs for oracle router and the real router.
class Evaluator:
    def __init__(self, config:Dict, model_list:List[str]):
        self.config = config
        self.prompts:List[Tuple[int, str]] = []
        self.inputs:List[Dict[str, Any]] = []
        self.iter_data:List = []
        self.model_list = model_list
        self.data_dir = self.config["data_dir"]
        self.Oracle_Judge = Oracle(self.model_list)
        self.load_prompts()
        self.prepare_inputs()
        self.gen_oracle()

    def load_prompts(self):
        input_dir = self.config["Benchmarks"]
        with open(input_dir, "r") as f:
            for idx, line in enumerate(f.readlines()):
                item = json.loads(line)
                prompt: str = item["prompt"]
                self.prompts.append((idx, prompt))
    
    def prepare_inputs(self):
        for idx, _ in self.prompts:
            input:Dict[str, Any] = {}
            for model in self.model_list:
                self.inputs[model] = read_jsonl(self.config, config["Data"]["benchmark"], model, idx)
            self.inputs.append(input)    
    
    # generate oracle choice
    def gen_oracle(self):
        for input in self.inputs:
            choice = self.Oracle_Judge.get_oracle(input, self.latency_constraint, choice=2)
    
    # 生成可被迭代访问的数据
    def construct_data(self):
        pass
    
    # 为外部使用构造迭代器
    def __iter__(self):
        self.access_count += 1
        for item in self.data_list:
            yield item
    
    # 迭代器的必要属性
    def __len__(self):
        return len(self.prompts)

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
    
    evaluator = Evaluator(config=config, model_list=model_list)
    for idx, item in enumerate(evaluator, start=1):
        prompt, oracle_choice = item
        
        choice = router.route(prompt=prompt,
                              model_name_list=model_list,
                              latency_constraint=latency_constraint)
        
        record_list.append((idx, prompt, oracle_choice, choice))