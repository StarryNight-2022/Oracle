# 该文件负责进行Latency的预测，具体有两个技术路线
# 1.使用小模型Qwen3-0.6B的输出长度来预测大模型的输出长度，这里需要拟合二者之间的关系（如何实现往不同模型的映射？）
# 2.使用Qwen3-Embeddings-0.6B得到prompt的Embedding，再使用模型拟合这个向量与输出长度的关系。（可以先映射到Qwen3-0.6B的输出长度）
import torch
from router.models.modeling.modeling import Bert, MLP
import numpy as np

class tokens_predictor():
    def __init__(
        self,
        bert_dir:str = "/home/ouyk/project/ICDCS/Oracle/model/Fine_Tuned",
        classifier_dir:str = "/home/ouyk/project/ICDCS/Oracle/model/Classifier/Qwen3-0.6B.bin",
        bert_hidden_dim:int = 768,
        hidden_size:int = 192,
        output_size: int = 16,
        device: int = torch.device("cuda:1"),
        dtype: int = torch.float32
        ):
        
        self.bert = Bert(
            bert_dir=bert_dir,
            device=device,
            dtype=dtype
            )

        self.classifier = MLP(
            input_size=bert_hidden_dim,
            hidden_size=hidden_size,
            output_size=output_size,
            device=device,
            dtype=dtype
            )
        
        self.classifier.load_state_dict(torch.load(classifier_dir))
        self.classifier.eval()
    
    def run(self, prompt:str)->int:
        with torch.no_grad():
            embedding = self.bert(prompt)
            out = self.classifier(embedding)
        result = np.argmax(out.cpu().numpy(), axis=-1)
        return result[0]

if __name__ == "__main__":
    bert_dir="/home/ouyk/project/ICDCS/Oracle/model/Fine_Tuned"
    classifier_dir="/home/ouyk/project/ICDCS/Oracle/model/Classifier/Qwen3-0.6B.bin"
    bert_hidden_dim=768
    hidden_size=192
    output_size=16
    device = torch.device("cuda:1")
    dtype=torch.float32
    predictor = tokens_predictor(bert_dir=bert_dir,
                                 classifier_dir=classifier_dir,
                                 bert_hidden_dim=bert_hidden_dim,
                                 hidden_size=hidden_size,
                                 output_size=output_size,
                                 device=device,
                                 dtype=dtype)
    
    result = predictor.run("Plants create [MASK] through a process known as photosynthesis.")
    print(result)