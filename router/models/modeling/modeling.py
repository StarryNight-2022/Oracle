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
    def __init__(self, input_size:int, output_size:int):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, output_size)
    
    def forward(self, x:torch.Tensor):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x