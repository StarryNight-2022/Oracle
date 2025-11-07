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
        self.fc1 = nn.Linear(input_size, hidden_size, dtype=dtype).to(device)
        self.fc2 = nn.Linear(hidden_size, hidden_size, dtype=dtype).to(device)
        self.fc3 = nn.Linear(hidden_size, output_size, dtype=dtype).to(device)
        self.relu = nn.ReLU().to(device)
    
    def forward(self, x:torch.Tensor):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Bert_MLP(nn.Module):
    # "/home/ouyk/project/ICDCS/Oracle/model/Bert_Base"
    def __init__(self, 
                 bert_dir:str, 
                 classifier_dir:str,
                 bert_hidden_dim:int, 
                 hidden_size:int, 
                 output_size:int,
                 device:torch.device,
                 dtype:torch.dtype,
                 fine_tune:bool):
        super(Bert_MLP, self).__init__()
        self.device = device
        
        if self.train:
            pass
        elif self.eval:
            # 预训练的Bert模型相关内容
            self.tokenizer = AutoTokenizer.from_pretrained(
                bert_dir,
            )
        else:
            raise ValueError("The model can only be one of training mode or evaluating mode!")
        # 预训练的Bert模型
        self.bert = BertModel.from_pretrained(
            bert_dir,
            dtype=dtype,
            attn_implementation="sdpa"
        ).to(self.device)
        
        if fine_tune == True:
            self.bert.train()
        elif fine_tune == False:
            # 冻结bert的参数
            for name, param in self.bert.named_parameters():
                param.requires_grad = False
            self.bert.eval()
        else:
            raise ValueError(f"param fine_tune can't be {fine_tune}")
        
        self.dropout = nn.Dropout(0.1)   # 仅在训练过程有效
        
        self.classifier = MLP(input_size=bert_hidden_dim,
                                hidden_size=hidden_size,
                                output_size=output_size,
                                device=self.device,
                                dtype=dtype).train()
        
        # Evaluate
        if classifier_dir != "":
            self.classifier.load_state_dict(classifier_dir)
            self.classifier.eval()
        
    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,):
        
        input_ids.to(self.device)
        
        # NOTE: Maybe there are some problems in training. Will fix when appear.
        bert_outputs = self.bert(input_ids,
                                attention_mask,
                                token_type_ids,
                                position_ids,
                                head_mask,
                                inputs_embeds,
                                labels,
                                output_attentions,
                                output_hidden_states,
                                return_dict,) 
        pooled_output = bert_outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits