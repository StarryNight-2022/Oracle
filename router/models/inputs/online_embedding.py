# Implementation for getting inputs from "Online embedding API deployed based on vLLM"
# 使用vLLM在本地部署一个Embedding API，在运行时使用该API获取embedding
# 需要进行调用开销计时
from openai import OpenAI
from typing import Dict, List, Any
import os
import json
import traceback
import time
import numpy as np

class online_embedding():
    # 需要指定index_list参数来确保移除了指定的outliers
    def __init__(self, config: Dict, index_list:List[int], model:str, input_text:bool = False, output_text:bool = False, embedding_model:str = "Qwen3-Embeddings-0.6B"):
        self.benchmark = config["Data"]["benchmark"]
        self.num_data  = config["Data"]["num_data"]
        self.data_dir  = os.path.join(config["Data"]["data_dir"], model)
        self.model = embedding_model
        self.index_list = index_list
        self.input = input_text
        self.output = output_text

        # Embedding API
        self.api_key  = os.environ.get("Local_Embedding_Key")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="http://localhost:8000/v1",
        )
        
        self.access_count = 0
        self.data_list = []
        self.load_datasets()
    
    def __iter__(self):
        self.access_count += 1
        for item in self.data_list:
            embedding = self.embed(item)
            yield embedding
    
    def __len__(self):
        return len(self.datasets)
    
    def load_datasets(self):
        for idx in self.index_list:
            self.data_list.append(self.read_jsonl(idx))
    
    # 每次装载一个结果，选出"prompt"
    def read_jsonl(self, idx: int):
        filepath = os.path.join(
            self.data_dir,
            f"train_{idx}.jsonl")
        try:
            with open(filepath, 'r') as file:
                line = file.readline()
                if self.input and not self.output:    
                    return (json.loads(line))["prompt"]
                elif self.output and not self.input:
                    return (json.loads(line))["prompt"]
                else:
                    raise ValueError("You can only choice one between input_text and output_text!")
        except Exception:
            print(traceback.format_exc())
            return None
    
    def embed(self, prompt:str) -> List[float]:
        responses = self.client.embeddings.create(
            input=[prompt],
            model=self.model,
        )
        return responses.data[0].embedding
    
class online_embedding_profile():
    # 需要指定index_list参数来确保移除了指定的outliers
    def __init__(self, config: Dict, index_list:List[int], model:str, input_text:bool = False, output_text:bool = False, embedding_model:str = "Qwen3-Embeddings-0.6B"):
        self.benchmark = config["Data"]["benchmark"]
        self.num_data  = config["Data"]["num_data"]
        self.data_dir  = os.path.join(config["Data"]["data_dir"], model)
        self.model = embedding_model
        self.index_list = index_list
        self.input = input_text
        self.output = output_text

        # Embedding API
        if self.model == "Qwen3-Embeddings-0.6B":
            self.api_key  = os.environ.get("Local_Embedding_Key")
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="http://localhost:8000/v1",
            )
        elif self.model == "bert-embedding":
            self.client = OpenAI(
                api_key="",
                base_url="http://localhost:8000/v1",
            )
        
        self.access_count = 0
        self.data_list = []
        self.load_datasets()
    
    def __iter__(self):
        self.access_count += 1
        for item in self.data_list:
            embedding, time = self.embed(item)
            yield embedding, time
    
    def __len__(self):
        return len(self.datasets)
    
    def load_datasets(self):
        for idx in self.index_list:
            self.data_list.append(self.read_jsonl(idx))
    
    # 每次装载一个结果，选出"prompt"
    def read_jsonl(self, idx: int):
        filepath = os.path.join(
            self.data_dir,
            f"train_{idx}.jsonl")
        try:
            with open(filepath, 'r') as file:
                line = file.readline()
                if self.input and not self.output:    
                    return (json.loads(line))["prompt"]
                elif self.output and not self.input:
                    return (json.loads(line))["full_response"]
                else:
                    raise ValueError("You can only choice one between input_text and output_text!")
        except Exception:
            print(traceback.format_exc())
            return None
    
    def embed(self, prompt:str) -> List[float]:
        start = time.time()
        responses = self.client.embeddings.create(
            input=[prompt],
            model=self.model,
        )
        end = time.time()
        return responses.data[0].embedding, (end-start)

# Example 
if __name__ == "__main__":
    import yaml
    config_file = "/home/ouyk/project/ICDCS/Oracle/config/router_model_GSM8K.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    model_A = "Qwen3-0.6B-temp-0-no-thinking"
    model_B = "Qwen3-14B-temp-0-no-thinking"
    record = os.path.join(config["Data"]["data_dir"], model_B, "without_outliers.npy")
    index_list = (np.load(record)).tolist()
    
    # 获取到在GSM8K数据集上每一条query对应的num_tokens
    for embedding in online_embedding(config, index_list, model_A, embedding_model="Qwen3-Embeddings-0.6B"):
        print(len(embedding))