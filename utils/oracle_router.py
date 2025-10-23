# 这个文件负责实现oracle判定决策
# - Iterate through all queries
# - Always pick the correct model (Correct=True)
# - If both models are correct, pick the "small" model
# - If both models are wrong, pick the "large" model (since the decision is with respect to small)
# - Compute:
#   - Total and average (across all GSM tasks) latency:
#     - Based on the model used by oracle, you add its runtime! obvious!
#   - Total and average output tokens:
#     - Based on the model used by oracle, you add its (length) output tokens!
#   - Large model total and average output tokens:
#     - We will use it later. In case Large is a GPT model, the tokens will be a proxy for the dollar cost, so we can use that for now! When the large model is used by oracle, you add its (length) output tokens!
#   - Overall correctness:
#     - How many of the total questions does the oracle get correct?
#     - This might be less than 100% since we have cases where both models are wrong!
#   - percentage_to_large compute how many times the oracle selects the large model!
from typing import List, Dict, Any, Optional, Union
from copy import deepcopy
from utils.config import model_size

class Oracle:
    def __init__(self, config: Any):
        self.models:List[str] = [model_info['name'] for model_info in config['Models'].values()]
        # model size unit is "B" means Billian of params.
        self.model_size = model_size
    
    # TODO: Need modify
    def get_oracle(self, results: Dict[str, Any], latency_constraint: Union[float, int, None], choice: int) -> Dict[str, Any]:  # 延迟限制的单位为秒:second
        # The 1st strategy without latency constraint
        if choice == 0:
            oracle = self.strategy_0(results)
        # The 2nd strategy with latency constraint
        elif choice == 1:
            if isinstance(latency_constraint, (int, float)):
                oracle = self.strategy_1(results, latency_constraint)
            else:
                raise ValueError("latency_constraint should be a int/float value. Can't be None!")
        # The 3rd strategy with latency constraint
        elif choice == 2:
            if isinstance(latency_constraint, (int, float)):
                oracle = self.strategy_2(results, latency_constraint)
            else:
                raise ValueError("latency_constraint should be a int/float value. Can't be None!")
        else:
            raise NotImplementedError("Don't supports that oracle strategy.")
        
        return oracle
    
    def strategy_0(self, results: Dict[str, Any]):
        # 仅选取results中的部分字段(correctness, runtime, length_of_output_token_ids)写入raw中，并加入model_size
        raw: Dict[str, Any] = {}
        for model in list(results.keys()):
            raw[model] = {}  # 先初始化内层字典
            raw[model]["correctness"] = results[model]["correctness"]
            raw[model]["latency"] = results[model]["runtime"]
            raw[model]["output_tokens"] = results[model]["length_of_output_token_ids"]
            raw[model]["model_size"] = self.model_size[model]
        # 对于每一个模型
        judge_standards:Dict[str, Any] = {"correctness":{}, "latency":{}, "size":{}, "output_tokens":{}}
        correct_model_list = deepcopy(self.models)
        # step1. 填充判断依据 + 基于correctness进行筛选
        for idx, model in enumerate(self.models, start=0):
            result = results[model]
            correctness = result["correctness"]
            # If answer is correct, then keep this model
            judge_standards["correctness"][model] = correctness
            judge_standards["latency"][model] = result["runtime"]
            judge_standards["size"][model] = self.model_size[model]
            judge_standards["output_tokens"][model] = result["length_of_output_token_ids"]
            if correctness == True:
                continue
            # If answer if incorrect, then remove this model
            else:
                correct_model_list.remove(model)
        
        # 对 judge_standards["size"] 进行排序, from small to large
        judge_standards["size"] = dict(sorted(judge_standards["size"].items(), key=lambda item: item[1]))

        # a.(All Wrong)If all models are unacceptable, choose the largest one.
        if len(correct_model_list) == 0:
            largest_model = list(judge_standards["size"].keys())[-1]
            oracle_choice = largest_model
        # b.(One Correct)If only one model is acceptable, choose it.    
        elif len(correct_model_list) == 1:
            oracle_choice = correct_model_list[0]
        # c.(Lots Correct)If lots of models are acceptable, choose the smallest one.
        elif len(correct_model_list) > 1:
            smallest_model = list(judge_standards["size"].keys())[0]
            oracle_choice = smallest_model
        else:
            raise ValueError("The oracle strategy can't do the judgement.")
        
        # 构建返回值:
        oracle:Dict[str, Any] = {
            "raw": raw,
            "model": oracle_choice,
            "correctness": judge_standards["correctness"][oracle_choice], 
            "latency": judge_standards["latency"][oracle_choice],
            "output_tokens": judge_standards["output_tokens"][oracle_choice],
            }
    
        return oracle
    
    #NOTE: Did some modifications based on strategy without latency constraint
    def strategy_1(self, results: Dict[str, Any], latency_constraint: Union[float, int]):
        # 仅选取results中的部分字段(correctness, runtime, length_of_output_token_ids)写入raw中，并加入model_size
        raw: Dict[str, Any] = {}
        for model in list(results.keys()):
            raw[model] = {}  # 先初始化内层字典
            raw[model]["correctness"] = results[model]["correctness"]
            raw[model]["latency"] = results[model]["runtime"]
            raw[model]["output_tokens"] = results[model]["length_of_output_token_ids"]
            raw[model]["model_size"] = self.model_size[model]
        # 对于每一个模型
        judge_standards:Dict[str, Any] = {"correctness":{}, "latency":{}, "size":{}, "output_tokens":{}}
        correct_model_list = deepcopy(self.models)
        # step1. 填充判断依据 + 基于correctness进行筛选
        for idx, model in enumerate(self.models, start=0):
            result = results[model]
            correctness = result["correctness"]
            judge_standards["correctness"][model] = correctness
            judge_standards["latency"][model] = result["runtime"]
            judge_standards["size"][model] = self.model_size[model]
            judge_standards["output_tokens"][model] = result["length_of_output_token_ids"]
            if correctness == True:
                # If answer is correct, then keep this model
                continue
            # If answer if incorrect, then remove this model
            else:
                correct_model_list.remove(model)
        
        # 对 judge_standards["size"] 进行排序, from small to large
        judge_standards["size"] = dict(sorted(judge_standards["size"].items(), key=lambda item: item[1]))
        
        # a.(All Wrong)If all models are unacceptable, choose the largest one.
        if len(correct_model_list) == 0:
            largest_model = list(judge_standards["size"].keys())[-1]
            oracle_choice = largest_model
        # b.(One Correct)If only one model is acceptable, choose it.
        elif len(correct_model_list) == 1:
            oracle_choice = correct_model_list[0]
        # c.(Lots Correct)If lots of models are acceptable, choose the smallest one.
        elif len(correct_model_list) > 1:
            latency_within_list = deepcopy(correct_model_list)
            
            # 确保进入这个环节的每一个模型都给出了正确的回复
            for idx, model in enumerate(self.models, start=0):
                if judge_standards["correctness"][model] != True:
                    judge_standards["correctness"].pop(model)
                    judge_standards["latency"].pop(model)
                    judge_standards["size"].pop(model)
                    judge_standards["output_tokens"].pop(model)
            
            # 对 judge_standards["size"] 进行排序, from small to large
            judge_standards["size"] = dict(sorted(judge_standards["size"].items(), key=lambda item: item[1]))
            
            # 判断是否每一个都符合延迟限制，不符合延迟限制的直接移除.
            for idx, model in enumerate(correct_model_list, start=0):
                if judge_standards["latency"][model] > latency_constraint:
                    latency_within_list.remove(model)
            
            # a.(All Wrong)If all models are unacceptable, choose the largest one.
            if len(latency_within_list) == 0:
                largest_model = list(judge_standards["size"].keys())[-1]
                oracle_choice = largest_model
            # b.(One Correct)If only one model is acceptable, choose it.    
            elif len(latency_within_list) == 1:
                oracle_choice = latency_within_list[0]
            # c.(Lots Correct)If lots of models are acceptable, choose the smallest one.
            elif len(latency_within_list) > 1:
                smallest_model = list(judge_standards["size"].keys())[0]
                oracle_choice = smallest_model
            else:
                raise ValueError("The oracle strategy can't do the judgement.")
        
        # 构建返回值:
        oracle:Dict[str, Any] = {
            "raw": raw,
            "model": oracle_choice,
            "correctness": judge_standards["correctness"][oracle_choice], 
            "latency": judge_standards["latency"][oracle_choice],
            "output_tokens": judge_standards["output_tokens"][oracle_choice],
            }
        
        return oracle
    
    # NOTE: Make the Latency constraint higher privilege than correctness
    def strategy_2(self, results:Dict[str, Any], latency_constraint: Union[float, int]):
        # 仅选取results中的部分字段(correctness, runtime, length_of_output_token_ids)写入raw中，并加入model_size
        raw: Dict[str, Any] = {}
        for model in list(results.keys()):
            raw[model] = {}  # 先初始化内层字典
            raw[model]["correctness"] = results[model]["correctness"]
            raw[model]["latency"] = results[model]["runtime"]
            raw[model]["output_tokens"] = results[model]["length_of_output_token_ids"]
            raw[model]["model_size"] = self.model_size[model]
        # 对于每一个模型
        judge_standards:Dict[str, Any] = {"correctness":{}, "latency":{}, "size":{}, "output_tokens":{}}
        latency_within_list = deepcopy(self.models)
        
        # step1. 填充判断依据 + 基于latency进行筛选
        for idx, model in enumerate(self.models, start=0):
            result = results[model]
            latency = result["runtime"]
            judge_standards["correctness"][model] = result["correctness"]
            judge_standards["latency"][model] = latency
            judge_standards["size"][model] = self.model_size[model]
            judge_standards["output_tokens"][model] = result["length_of_output_token_ids"]
            if latency <= latency_constraint:
                # If latency within latency-constraint, then keep this model
                continue
            # If out the latency-constraint, then remove this model
            else:
                latency_within_list.remove(model)
            
        # a.(All out)If all models are unacceptable, Timeout
        if len(latency_within_list) == 0:
            # Question timeout
            oracle_choice = "Timeout"
        # b.(Some within) Using correctness to filtering
        else:
            correct_model_list = deepcopy(latency_within_list)
            
            # 确保进入这个环节的每一个模型都符合latency-constraint
            for idx, model in enumerate(self.models, start=0):
                if judge_standards["latency"][model] > latency_constraint:
                    judge_standards["correctness"].pop(model)
                    judge_standards["latency"].pop(model)
                    judge_standards["size"].pop(model)
                    judge_standards["output_tokens"].pop(model)
                    
            # 对 judge_standards["size"] 进行排序, from small to large
            judge_standards["size"] = dict(sorted(judge_standards["size"].items(), key=lambda item: item[1]))
            # 对 judge_standards["latency"] 进行排序, from fast to slow
            judge_standards["latency"] = dict(sorted(judge_standards["latency"].items(), key=lambda item: item[1]))
            
            for idx, model in enumerate(latency_within_list, start=0):
                correctness = judge_standards["correctness"][model]
                if correctness == True:
                    continue
                else:
                    correct_model_list.remove(model)
            
            # If branch
            if len(correct_model_list) == 0:
                largest_model = list(judge_standards["size"].keys())[-1]
                oracle_choice = largest_model
            elif len(correct_model_list) == 1:
                oracle_choice = correct_model_list[0]
            elif len(correct_model_list) > 1:
                fastest_model = list(judge_standards["latency"].keys())[0]
                oracle_choice = fastest_model
            else:
                raise ValueError("The oracle strategy can't do the judgement.")
            
        # 构建返回值:
        # Represent response timeout
        if oracle_choice == "Timeout":
            # Assuming the router chooses the fastest model, but still timeout.
            judge_standards["latency"] = dict(sorted(judge_standards["latency"].items(), key=lambda item: item[1]))
            fastest_model = list(judge_standards["latency"].keys())[0]
            oracle:Dict[str, Any] = {
                "raw": raw,
                "model": fastest_model,
                "correctness": False,          # Cause timeout, don't get answer from the router. So, correctness = Fasle
                "latency": latency_constraint, # Cause timeout, the latency after routed should be the specific latency-constraint 
                "output_tokens": 0,            # Cause timeout, don't get answer from the router. So, output_tokens = 0
                }
        else:
            oracle:Dict[str, Any] = {
                "raw": raw,
                "model": oracle_choice,
                "correctness": judge_standards["correctness"][oracle_choice], 
                "latency": judge_standards["latency"][oracle_choice],
                "output_tokens": judge_standards["output_tokens"][oracle_choice],
                }
        return oracle