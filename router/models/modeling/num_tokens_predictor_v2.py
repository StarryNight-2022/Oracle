# 该文件负责进行Latency的预测，具体有两个技术路线
# 1.使用小模型Qwen3-0.6B的输出长度来预测大模型的输出长度，这里需要拟合二者之间的关系（如何实现往不同模型的映射？）
# 2.使用Qwen3-Embeddings-0.6B得到prompt的Embedding，再使用模型拟合这个向量与输出长度的关系。（可以先映射到Qwen3-0.6B的输出长度）
import torch
from router.models.modeling.modeling import Bert, MLP_2
from router.models.inputs.online_embedding import online_embedding
import numpy as np
from typing import Dict, Optional, List, Tuple
import os
import joblib
from sklearn.neighbors import KNeighborsClassifier
from router.models.router.config import MODEL_IDS

# NOTE: 区别于v1版本，v2版本的预测模型输入变量发生了变化，这里需要指定每个候选模型的名称 model_list:List[str]
#       目前该版本仅支持使用MLP模型，在训练该MLP模型时可能需要对训练代码进行调整。
class num_tokens_predictor():
    def __init__(self, config:Dict):
        self.config:Dict = config
        self.model = config["output_length_prediction"]["embedding_model"]
        self.predictor_choice = config["output_length_prediction"]["prediction_model"]
        self.mlp_dir:str = config["output_length_prediction"][self.model]["mlp_dir"]
        self.knn_dir:str = config["output_length_prediction"][self.model]["knn_dir"]
        self.embedding_dim:int = config["output_length_prediction"][self.model]["text_embedding_dim"]
        self.hidden_size:int = config["output_length_prediction"][self.model]["hidden_size"]
        self.num_ranges: int = config["output_length_prediction"]["num_tokens_range_split"]
        self.range_interval: int = config["output_length_prediction"]["range_interval"]
        self.device: int = torch.device(config["output_length_prediction"]["device"])
        
        self.dtype: torch.dtype = None
        if config["output_length_prediction"][self.model]["dtype"] == "torch.float32":
            self.dtype = torch.float32
        elif config["output_length_prediction"][self.model]["dtype"] == "torch.float16":
            self.dtype = torch.float16
        
        self.embedding_model = online_embedding(embedding_model=self.model)
        
        if self.predictor_choice == "mlp":
            if os.path.exists(self.mlp_dir):
                # NOTE: 由于vLLM占用存储，使用同一个显卡设备也许会出现报错，必要的话对vLLM进行配置以预留部分显存空间。
                self.predictor = MLP_2(
                    num_models=len(list(MODEL_IDS.keys())),
                    input_size=self.embedding_dim,
                    hidden_size=self.hidden_size,
                    output_size=self.num_ranges,
                    device=self.device,
                    dtype=self.dtype,
                    test=True,
                    )
                
                # self.predictor.load_state_dict(torch.load(self.mlp_dir))
                self.predictor.eval()
            else:
                raise FileNotFoundError("You have to train a MLP classifier first!")
        elif self.predictor_choice == "knn":
            raise NotImplementedError(f"For num_tokens_predictor_v2, knn predictor is removed!")
        else:
            raise NotImplementedError(f"Don't support that predictor choice: {self.predictor_choice}")
        
        self.range_dict:Dict[str, Tuple[int, int]] = {}
        range_list = [i*self.range_interval for i in range(self.num_ranges + 1)]
        for i in range(self.num_ranges):
            self.range_dict[f"{i}"] = (range_list[i], range_list[i+1])
    
    def run(self, prompt:str, model_list:List[str])->Tuple[List[Tuple[int, int]], torch.Tensor]:
        embedding = self.embedding_model.embed(prompt)
        
        if self.predictor_choice == "mlp":
            embedding_torch = torch.tensor(embedding, device=self.device, dtype=self.dtype)
            with torch.no_grad():
                out = self.predictor(embedding_torch, model_list)
            result = (np.argmax(out.cpu().numpy(), axis=-1)).tolist()
            return [self.range_dict[str(item)] for item in result], embedding_torch
        else:
            raise NotImplementedError(f"Don't support prefiction model {self.predictor_choice}")

if __name__ == "__main__":
    import yaml
    config_file = "/home/ouyk/project/ICDCS/Oracle/config/2_step_routing_test.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    predictor = num_tokens_predictor(config)
    print(predictor.range_dict)
    
    model_list=["Qwen3-0.6B-temp-0-no-thinking", "Qwen3-14B-temp-0-no-thinking"]
    
    # NOTE: Need specify the "range_dict", then we can get the range of the number of output tokens.
    result, embedding = predictor.run(prompt="Ken created a care package to send to his brother, who was away at boarding school.  Ken placed a box on a scale, and then he poured into the box enough jelly beans to bring the weight to 2 pounds.  Then, he added enough brownies to cause the weight to triple.  Next, he added another 2 pounds of jelly beans.  And finally, he added enough gummy worms to double the weight once again.  What was the final weight of the box of goodies, in pounds?",
                           model_list=model_list)
    for idx, model in enumerate(model_list, start=0):
        print(f"Prediction for number of output tokens for model {model} is {result[idx]}.")