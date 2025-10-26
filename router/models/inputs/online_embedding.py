# Implementation for getting inputs from "Online embedding API deployed based on vLLM"
# 使用vLLM在本地部署一个Embedding API，在运行时使用该API获取embedding
# 需要进行调用开销计时
from openai import OpenAI, AsyncOpenAI
import os
from typing import List

class online_embedding():
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
    
# Example
if __name__ == "__main__":
    model = online_embedding(embedding_model="Qwen3-Embeddings-0.6B")
    prompt = "Hello my name is"
    # embedding: List[float]
    embedding = model.embed(prompt)