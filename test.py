import torch
from router.models.modeling.modeling import Qwen3_Embedding

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PATH = '/home/ouyk/project/Runtime/Model/Qwen3-Embedding-0.6B'

model = Qwen3_Embedding(weight_dir=MODEL_PATH,
                        device=device,
                        max_length=8192)

text = "The capital of China is Beijing."

embedding = model(text)

print(embedding)