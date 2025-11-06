# 该文件负责进行Latency的预测，具体有两个技术路线
# 1.使用小模型Qwen3-0.6B的输出长度来预测大模型的输出长度，这里需要拟合二者之间的关系（如何实现往不同模型的映射？）
# 2.使用Qwen3-Embeddings-0.6B得到prompt的Embedding，再使用模型拟合这个向量与输出长度的关系。（可以先映射到Qwen3-0.6B的输出长度）
import torch
from router.models.modeling.modeling import Bert_MLP

device = torch.device("cuda:1")

predictor = Bert_MLP(bert_dir="/home/ouyk/project/ICDCS/Oracle/model/Bert_Base",
                     classifier_dir="",
                     bert_hidden_dim=768,
                     hidden_size=192,
                     output_size=16,
                     device=device)

out = predictor("Plants create [MASK] through a process known as photosynthesis.")

print(out.to("cpu"))