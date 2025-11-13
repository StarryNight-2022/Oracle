# constructing our router model
from typing import List, Union, Any, Optional
import numpy as np
#---------------------- KNN -----------------------
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pickle
#---------------------- MLP -----------------------
import torch
import torch.nn as nn
#------------------ Transformers ------------------
from transformers import BertModel, AutoTokenizer
#--------------------------------------------------
from router.models.router.config import MODEL_IDS

class KNN():
    def __init__(self, checkpoint:str):
        """根据超参数构建KNN分类器。使用距离加权；cosine 需 brute 搜索。"""
        # 加载模型
        with open(checkpoint, 'rb') as f:
            self.model = pickle.load(f)
        self.scaler = StandardScaler()
        
    def forward(self, X:np.ndarray)->np.ndarray:
        try:
            X_scaled = self.scaler.fit_transform(X)
        except:
            raise ValueError(f"X should be np.array, now is {type(X)}!")
            
        return self.model.predict(X_scaled)

class MLP(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, output_size:int, device:torch.device, dtype:torch.dtype):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size, device=device, dtype=dtype)
        self.relu = nn.ReLU()
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.relu(self.fc1(x))
        return x
    
class MLP_1(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, output_size:int, device:torch.device, dtype:torch.dtype):
        super(MLP_1, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, device=device, dtype=dtype)
        self.fc2 = nn.Linear(hidden_size, hidden_size, device=device, dtype=dtype)
        self.fc3 = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)
        self.relu = nn.ReLU()
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Bert(nn.Module):
    # "/home/ouyk/project/ICDCS/Oracle/model/Fine_Tuned"
    def __init__(self, 
                 bert_dir:str, 
                 device:torch.device,
                 dtype:torch.dtype):
        super(Bert, self).__init__()
        self.device = device
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            "/home/ouyk/project/ICDCS/Oracle/model/Bert_Base"
        )
        
        # 预训练的Bert模型
        self.bert = BertModel.from_pretrained(
            bert_dir,
            dtype=dtype,
            attn_implementation="sdpa"
        ).to(self.device)

        # 冻结bert的参数
        for name, param in self.bert.named_parameters():
            param.requires_grad = False
        self.bert.eval()
        
    def forward(self, prompt:str):
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        # 进行输入截断[:512]
        if inputs["input_ids"].shape[-1] > 512:
            inputs["input_ids"] = inputs["input_ids"][:, :512]
            inputs["token_type_ids"] = inputs["token_type_ids"][:, :512]
            inputs["attention_mask"] = inputs["attention_mask"][:, :512]
            
        inputs.to(self.device)
        
        # NOTE: Maybe there are some problems in training. Will fix when appear.
        with torch.no_grad():
            bert_outputs = self.bert(**inputs) 
        pooled_output = bert_outputs[1]

        return pooled_output

# TODO: 为了适应动态的输入模型数量，需要对此模型结构进行修改。
class MFModel(torch.nn.Module):
    def __init__(
        self,
        dim:int = 128,
        num_models:int = 2,
        text_dim:int = 1024,
        num_classes:int = 1,
        use_proj:bool = True,
        device:torch.device = None,
    ):
        '''
        @dim: Model name embedding dimension.\n
        @num_models: The number of models in MODEL_IDS.\n
        @text_dim: User prompt embedding.\n
        '''
        super().__init__()
        self._name = "TextMF"
        self.use_proj = use_proj
        self.P = torch.nn.Embedding(num_models, dim)

        if self.use_proj:
            self.text_proj = torch.nn.Sequential(
                torch.nn.Linear(text_dim, dim, bias=False)
            )
        else:
            assert (
                text_dim == dim
            ), f"text_dim {text_dim} must be equal to dim {dim} if not using projection"

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(dim, num_classes, bias=False)
        )

    def get_device(self):
        return self.P.weight.device

    def forward(self, model_ids:List[int], prompt_embed:torch.Tensor):
        model_ids = torch.tensor(model_ids, dtype=torch.long).to(self.get_device())

        model_embed = self.P(model_ids) # [len(model_list), dim]
        model_embed = torch.nn.functional.normalize(model_embed, p=2, dim=1) # [len(model_list), dim]
        
        prompt_embed = self.text_proj(prompt_embed) # [dim]

        return self.classifier(model_embed * prompt_embed).squeeze() # [len(model_list), 1]

    @torch.no_grad()
    def choose(self, model_list:List[str], prompt_embed:torch.Tensor)->str:
        model_ids = [MODEL_IDS[model_name] for model_name in model_list]
        logits = self.forward(model_ids, prompt_embed) # [len(model_list), 1]
        choice = torch.argmax(logits, dim=0)
        return model_list[choice.cpu().numpy()[0]]

    def load(self, path):
        self.load_state_dict(torch.load(path))