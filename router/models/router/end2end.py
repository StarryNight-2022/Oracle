# 该文件负责实现端到端版本的路由
# 将时延判断与路由决策合二为一
# @input:
#   - model_name_list:List[str]
#   - prompt: str 
#   - latency_constraint: float
from typing import List, Dict
import os
import numpy as np

class Router():
    def __init__(self, config:Dict):
        self.config = config
    
    def route(self, prompt:str, model_name_list:List[str], latency_constraint:float)->str:
        pass
    
if __name__ == "__main__":
    import yaml
    config_file = "/home/ouyk/project/ICDCS/Oracle/config/routing_test.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    router = Router(config=config)
    
    model_name_list:List[str] = ['Qwen3-0.6B-no-thinking', 'Qwen3-14B-no-thinking']
    latency_constraint:float = 5
    test_prompt:str = "Ken created a care package to send to his brother, who was away at boarding school.  Ken placed a box on a scale, and then he poured into the box enough jelly beans to bring the weight to 2 pounds.  Then, he added enough brownies to cause the weight to triple.  Next, he added another 2 pounds of jelly beans.  And finally, he added enough gummy worms to double the weight once again.  What was the final weight of the box of goodies, in pounds?"
    
    choice = router.route(prompt=test_prompt,
                          model_name_list=model_name_list,
                          latency_constraint=latency_constraint)
    
    print(f"end2end router's choice is {choice}")