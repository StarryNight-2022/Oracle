# 该文件通过OpenAI API调用基于vLLM部署的Server，计划使用Qwen3-0.6B模型进行推理得到输出tokens的数值，
# 再使用后续模型进行其它LLM输出长度的预测；这个后续模型可以是一个简单的线性回归，也可以使用简单的MLP。
from openai import OpenAI
from typing import Dict, List, Any, Optional, Tuple
import os
import json
import traceback
import time
import numpy as np

class online_inference():
    # 需要指定index_list参数来确保移除了指定的outliers
    '''
    @config 当config不为None时才会进行数据的装载
    '''
    def __init__(self, 
                 llm:str = "Qwen3-0.6B",
                 config: Optional[Dict] = None,
                 index_list:Optional[List[int]] = None, 
                 model:Optional[str] = None, 
                 input_text:Optional[bool] = False, 
                 output_text:Optional[bool] = False):
        self.model = llm
        self.config = config
        # LLM API
        self.api_key  = os.environ.get("Local_LLM_Key")
        # LLM API
        if self.model == "Qwen3-0.6B":
            self.api_key  = os.environ.get("Local_LLM_Key")
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="http://localhost:8000/v1",
            )
        elif self.model == "Qwen2.5-0.5B":
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="http://localhost:8000/v1",
            )  
        else:
            raise NotImplementedError(f"Don't support that LLM model:{self.model}")
        
        self.access_count = 0
        self.data_list = []
        
        # 当config不为None时，才会进行数据的装载
        if self.config != None:
            self.benchmark = config["Data"]["benchmark"]
            self.num_data  = config["Data"]["num_data"]
            self.data_dir  = os.path.join(config["Data"]["data_dir"], model)
            self.index_list = index_list
            self.input = input_text
            self.output = output_text
            self.load_datasets()
    
    def __iter__(self):
        self.access_count += 1
        for item in self.data_list:
            response = self.gen(item)
            yield response
    
    def __len__(self):
        return len(self.data_list)
    
    def load_datasets(self):
        for idx in self.index_list:
            self.data_list.append(self.read_jsonl(idx))
    
    # 每次装载一个结果，选出"prompt"
    def read_jsonl(self, idx: int):
        filepath = os.path.join(self.data_dir, f"train_{idx}.jsonl")
        try:
            with open(filepath, 'r') as file:
                line = file.readline() 
                return (json.loads(line))["prompt"]
        except Exception:
            print(traceback.format_exc())
            return None
    
    '''
    @param prompt 输入的文本
    @return 一个元组，第一个元素是模型的输出，第二个元素是模型输入的tokens数值，第三个元素是模型输出的tokens数值
    '''
    def gen(self, prompt:str) -> Tuple[str, int, int]:
        try:
            _extra_body_params={}
            _extra_body_params["chat_template_kwargs"]= {"enable_thinking": False}
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],       
                temperature=0.0,              
                top_p=0.95,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                n=1,
                extra_body=_extra_body_params)
        except:
            raise NotImplementedError("Please make sure you have already started the vLLM server!")
        return response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens

# Example 
if __name__ == "__main__":
    import yaml
    config_file = "/home/ouyk/project/ICDCS/Oracle/config/router_model_GSM8K.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    model_A = "Qwen3-0.6B-temp-0-no-thinking"
    model_B = "Qwen3-14B-temp-0-no-thinking"
    record = os.path.join(config["Data"]["data_dir"], model_B, "without_outliers.npy")
    index_list = (np.load(record)).tolist()  # 为了确保移除了outliers
    
    # 获取到在GSM8K数据集上每一条query对应的num_tokens
    for inference in online_inference(config=config, index_list=index_list, model=model_A, llm="Qwen3-0.6B"):
        answer, in_tokens, out_tokens = inference
        print(answer, in_tokens, out_tokens)
        break
