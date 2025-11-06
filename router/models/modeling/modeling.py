# constructing our router model
from typing import List, Union
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
    def __init__(self, input_size:int, hidden_size:int, output_size:int):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, x:torch.Tensor):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        # x = self.logsoftmax(x)
        return x
    
class Bert_MLP(nn.Module):
    # "/home/ouyk/project/ICDCS/Oracle/model/Bert_Base"
    def __init__(self, 
                 bert_dir:str, 
                 bert_hidden_dim:int, 
                 hidden_size:int, 
                 output_size:int, 
                 device:torch.device):
        super(Bert_MLP, self).__init__()
        self.device = device
        
        # 预训练的Bert模型相关内容
        self.tokenizer = AutoTokenizer.from_pretrained(
            bert_dir,
        )
        # 预训练的Bert模型
        self.bert = BertModel.from_pretrained(
            bert_dir,
            dtype=torch.float16,
            attn_implementation="sdpa"
        ).to(self.device)
        
        self.dropout = nn.Dropout(0.1)   # 仅在训练过程有效
        
        self.fc1 = nn.Linear(bert_hidden_dim, hidden_size, dtype=torch.float16).to(self.device)
        self.fc2 = nn.Linear(hidden_size, hidden_size, dtype=torch.float16).to(self.device)
        self.fc3 = nn.Linear(hidden_size, output_size, dtype=torch.float16).to(self.device)
        self.relu = nn.ReLU().to(self.device)

    def classifier(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
    def forward(self, query:str):
        input_ids = self.tokenizer(query, return_tensors="pt").to(self.device)
        bert_outputs = self.bert(**input_ids)
        pooled_output = bert_outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits