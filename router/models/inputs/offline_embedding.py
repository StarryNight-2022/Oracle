# 提前使用Embedding模型获取到数据集中每个Prompt的Embedding并存储起来
from openai import OpenAI, AsyncOpenAI
import os
from typing import List, Tuple, Dict
from router.models.inputs.online_embedding import online_embedding_profile
import numpy as np
import yaml
from tqdm import tqdm

from router.utils.tools import ensure_dir

# 读取提前生成好的npy数据
class offline_embedding():
    # 需要指定index_list参数来确保移除了指定的outliers
    def __init__(self, config:Dict, index_list:List[int], model:str, embedding_model:str = "Qwen3-Embeddings-0.6B", gen:bool=False):
        self.benchmark = config["Data"]["benchmark"]
        ensure_dir(os.path.join(config["Data"]["embeddings_dir"], model))
        self.embeddings_dir = os.path.join(config["Data"]["embeddings_dir"], model, f"{self.benchmark}_embeddings.npy")
        self.model = model
        self.embedding_model = embedding_model
        self.index_list = index_list
        
        self.embedding_list: List[Tuple[int, float, List[float]]] = []
        if gen == False:
            try:
                self.load_data()
            except:
                raise FileNotFoundError(f"Please generate the {self.benchmark}_embeddings.npy first!")
        self.access_count = 0
    
    def __iter__(self):
        self.access_count += 1
        for item in self.embedding_list:
            idx, time, embedding = item
            yield idx, time, embedding
    
    def __len__(self):
        return len(self.embedding_list)
    
    def gen_data(self):
        # 获取到在GSM8K数据集上每一条query对应的num_tokens
        idx = 0
        for embedding, time in tqdm(online_embedding_profile(config, index_list=self.index_list, model=self.model, embedding_model=self.embedding_model)):
            idx += 1
            self.embedding_list.append((idx, time, embedding))
            
        # 将embedding_list存储为*.npy
        structured_data = self.list_to_structured_array(self.embedding_list)
        np.save(os.path.join(self.embeddings_dir), structured_data)
    
    def load_data(self):
        loaded_data = np.load(self.embeddings_dir, allow_pickle=True)
        self.embedding_list = self.structured_array_to_list(loaded_data)

    def list_to_structured_array(self, data: List[Tuple[int, float, List[float]]]) -> np.ndarray:
        """将列表数据转换为结构化数组"""
        if not data:
            return np.array([], dtype=[
                ('id', 'i4'),
                ('time', 'f4'),
                ('embedding', 'f4', (0,))
            ])
        
        embedding_dim = len(data[0][2])
        dtype = [
            ('id', 'i4'),
            ('time', 'f4'),
            ('embedding', 'f4', (embedding_dim,))
        ]
        
        return np.array(
            [(id_val, time, emb) for id_val, time, emb in data],
            dtype=dtype
        )

    def structured_array_to_list(self, data: np.ndarray) -> List[Tuple[int, float, List[float]]]:
        """将结构化数组转换回列表格式"""
        return [
            (row['id'], row['time'], row['embedding'].tolist())
            for row in data
        ]

# Generate prompts' embeddings [直接运行该文件负责生成指定数据集的embedding文件]
# 首先调用 online_embedding 生成 embedding，并使用numpy的npy/npz格式存储下来，以备调用。
if __name__ == "__main__":
    embedding_model="Qwen3-Embeddings-0.6B"
    config_file = "/home/ouyk/project/ICDCS/Oracle/config/router_model.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    model_A = "Qwen3-0.6B-temp-0-no-thinking"
    model_B = "Qwen3-14B-temp-0-no-thinking"
    record = os.path.join(config["Data"]["data_dir"], model_B, "without_outliers.npy")
    index_list = (np.load(record)).tolist()
    
    # Generate embeddings with vLLM(生成embeddings数据)
    # tool = offline_embedding(config, index_list, model_A, embedding_model, gen=True)
    # tool.gen_data()
    
    # 获取到在GSM8K数据集上每一条query对应的num_tokens
    for idx, time, embedding in offline_embedding(config, index_list, model_A, embedding_model, gen=False):
        print(idx, time, len(embedding))