# 该文件负责实现两步执行版本的路由
# 第一部分为输出长度预测 -> 时延预测
# 第二部分为路由决策
# @input:
#   - model_name_list:List[str]
#   - prompt: str
#   - latency_constraint: float
from typing import List, Tuple, Dict
import torch
from router.models.modeling.num_tokens_predictor_v1 import num_tokens_predictor
from router.models.modeling.modeling import MFModel
from router.models.router.config import MODEL_IDS, LLM_TIME_PARAMS

class Router():
    def __init__(self, config:Dict):
        self.config = config
        # step1 需要使用的输出tokens数量预测器
        self.tokens_predictor = num_tokens_predictor(config=self.config)
        # step2 需要用到MF模型
        self.model = config["output_length_prediction"]["embedding_model"]
        model_name_embed_dim:int = config["output_length_prediction"]["model_name_embed_dim"]
        text_embedding_dim:int = config["output_length_prediction"][self.model]["text_embedding_dim"]
        self.device = torch.device(config["output_length_prediction"]["device"])
        self.choice_maker = MFModel(dim=model_name_embed_dim,
                                    num_models=len(list(MODEL_IDS.keys())),
                                    text_dim=text_embedding_dim,
                                    use_proj= not (text_embedding_dim == model_name_embed_dim),
                                    device=self.device)
        self.choice_maker.load(path="/home/ouyk/project/ICDCS/Oracle/model/MF/mf_model.pth")
        
    # 基于输出长度，每个模型的等待时延是可计算的，无需使用网络进行学习。
    def step1(self, prompt:str, model_name_list:List[str], latency_constraint:float)->Tuple[torch.Tensor, List[str], bool]:
        '''
        @return: 
            embeddings: torch.Tensor
            models within latency_constraint: List[str]
        '''
        models_within:List[str] = []
        latency_dict:Dict[str, float] = {}
        
        output_length_prediction, embedding = self.tokens_predictor.run(prompt)
        for model in model_name_list:
            # TODO: 需要为每个候选模型在每个设备上测试得到TTFT与TPOT参数
            # TODO: 后续考虑添加一个根据每次实际运行参数动态更新的机制，计数+平均即可。
            # # latency = a * num_tokens + b
            b = LLM_TIME_PARAMS[model]["b"]
            a = LLM_TIME_PARAMS[model]["a"]
            # 取上限与下限的平均值
            latency_prediction = ((b + a * output_length_prediction[0]) + (b + a *output_length_prediction[1]))/2
            latency_dict[model] = latency_prediction
            # 预测该模型会发生超时 Timeout
            if latency_prediction > latency_constraint:
                pass
            elif latency_prediction <= latency_constraint:
                models_within.append(model)
            else:
                pass
        
        # 所有模型均超时，返回最快的模型
        if models_within == []:
            # 找到最快的模型
            fastest_model = min(latency_dict, key=latency_dict.get)
            return [fastest_model], embedding, True
        # 未超时，返回所有在约束内的模型
        else:
            return models_within, embedding, False
    
    # 这部分可以借鉴RouteLLM
    def step2(self, embedding:torch.Tensor, models_within:List[str])->str:
        # print("models_within", models_within)
        llm_chosen:str = self.choice_maker.choose(model_list=models_within,
                                                  prompt_embed=embedding)
        return llm_chosen
    
    def route(self, prompt:str, model_name_list:List[str], latency_constraint:float)->str:
        models_within, embedding, timeout = self.step1(prompt=prompt,
                                              model_name_list=model_name_list,
                                              latency_constraint=latency_constraint)
        if timeout == True:
            return models_within[0]
        else:
            choice = self.step2(embedding=embedding,
                            models_within=models_within)
            return choice
    
if __name__ == "__main__":
    import yaml
    config_file = "/home/ouyk/project/ICDCS/Oracle/config/2_step_routing_test.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    router = Router(config=config)
    
    model_name_list:List[str] = ['Qwen3-0.6B-temp-0-no-thinking', 'Qwen3-14B-temp-0-no-thinking']
    latency_constraint:float = 5
    test_prompt:str = "Ken created a care package to send to his brother, who was away at boarding school.  Ken placed a box on a scale, and then he poured into the box enough jelly beans to bring the weight to 2 pounds.  Then, he added enough brownies to cause the weight to triple.  Next, he added another 2 pounds of jelly beans.  And finally, he added enough gummy worms to double the weight once again.  What was the final weight of the box of goodies, in pounds?"
    
    choice = router.route(prompt=test_prompt,
                          model_name_list=model_name_list,
                          latency_constraint=latency_constraint)
    
    print(f"2_step router's choice is {choice}")