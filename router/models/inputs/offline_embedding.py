# 提前使用Embedding模型获取到数据集中每个Prompt的Embedding并存储起来
from openai import OpenAI, AsyncOpenAI
import os
from typing import List

class online_api_embedding():
    def __init__(self, embedding_model:str = "Qwen3-Embeddings-0.6B"):
        self.api_key  = os.environ.get("Local_Embedding_Key")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="http://localhost:8000/v1",
        )
        self.model = embedding_model
        
    def embed(self, prompt:str) -> List[float]:
        responses = self.client.embeddings.create(
            input=[prompt],
            model=self.model,
        )
        embedding = responses['data'][0]['embedding']
        return embedding
    
# Generate prompts' embeddings [直接运行该文件负责生成指定数据集的embedding文件]
if __name__ == "__main__":
    model = online_api_embedding(embedding_model="Qwen3-Embeddings-0.6B")
    prompt = "Hello my name is"
    # embedding: List[float]
    embedding = model.embed(prompt)