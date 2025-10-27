# Implementation for getting inputs from "Online embedding API deployed based on vLLM"
# 使用vLLM在本地部署一个Embedding API，在运行时使用该API获取embedding
# 需要进行调用开销计时
from openai import OpenAI
from typing import Dict, List, Any
import os
import json
import traceback

class online_embedding():
    def __init__(self, config: Dict, embedding_model:str = "Qwen3-Embeddings-0.6B"):
        self.benchmark = config["Data"]["benchmark"]
        self.data_dir  = config["Data"]["data_dir"]
        self.num_data  = config["Data"]["num_data"]
        self.access_count = 0
        self.data_lists = []
        self.load_datasets()
        
        # Embedding API
        self.api_key  = os.environ.get("Local_Embedding_Key")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="http://localhost:8000/v1",
        )
        self.model = embedding_model
    
    def __iter__(self):
        self.access_count += 1
        for item in self.data_lists:
            embedding = self.embed(item)
            yield embedding
    
    def __len__(self):
        return len(self.datasets)
    
    def load_datasets(self):
        for idx in range(1, self.num_data+1):
            self.data_lists.append(self.read_jsonl(idx))
    
    # 每次装载一个结果，选出"prompt"
    def read_jsonl(self, idx: int):
        filepath = os.path.join(
            self.data_dir,
            f"train_{idx}.jsonl")
        try:
            with open(filepath, 'r') as file:
                line = file.readline()
                return (json.loads(line))["prompt"]
        except Exception:
            print(traceback.format_exc())
            return None
    
    def embed(self, prompt:str) -> List[float]:
        responses = self.client.embeddings.create(
            input=[prompt],
            model=self.model,
        )
        return responses.data[0].embedding

# Example 
if __name__ == "__main__":
    import yaml
    config_file = "/home/ouyk/project/ICDCS/Oracle/config/router_model.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    # 获取到在GSM8K数据集上每一条query对应的num_tokens
    for embedding in online_embedding(config, embedding_model="Qwen3-Embeddings-0.6B"):
        print(len(embedding))