# This is a classify mission. Take prompts' embeddings as inputs, and predict the output length range(one-hot encoding)
# Question: How to define the range of output length?

from typing import List, Union, Tuple, Dict
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import random
import yaml
import os
import joblib

# 自行实现的内容
from router.models.scripts.datasets import prepare_training_data, train_test_split

if __name__ == "__main__":
    embedding_model="Qwen3-Embeddings-0.6B"
    config_file = "/home/ouyk/project/ICDCS/Oracle/config/router_model_GSM8K.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
        
    n_neighbors = config["Data"]["labels"]["num_tokens_range_split"]
        
    model_A = "Qwen3-0.6B-temp-0-no-thinking"   # use its embedding as inputs
    # model_B = "Qwen3-14B-temp-0-no-thinking"    # use its output_length as lables
    model_B = "Qwen3-0.6B-temp-0-no-thinking"    # use its output_length as lables
    record = os.path.join(config["Data"]["data_dir"], model_B, "without_outliers.npy")
    index_list = (np.load(record)).tolist()
    data_require = {
        "input_embeddings": True,
        "output_embeddings": False,
        "output_tokens": False,
        "latency": True,
        "output_tokens_label": False,
        "latency_label": True,
    }
    
    # 准备数据 lable_strategy: 0->Fixed Intervals, 1->Flexible Intervals
    X, Y, range_dict = prepare_training_data(config, index_list, model_A, model_B, embedding_model, data_require=data_require, lable_strategy=2)
    
    y = Y["latency"]
    
    # 分割数据
    # X_train, X_test:{"input_embeddings": np.ndarray, "output_embeddings":np.ndarray, "output_tokens":np.ndarray}
    X_train, X_test, y_train, y_test, _, _ = train_test_split(X, y, test_ratio=0.2)
    
    # 实例化KNN模型
    # knn = KNeighborsClassifier(n_neighbors, weights="distance", metric="cosine", algorithm="brute")
    knn = KNeighborsClassifier(n_neighbors, weights="distance", metric="minkowski")
    
    # train KNN
    knn.fit(X_train["input_embeddings"], y_train)
    
    # evaluate KNN
    accuracy = knn.score(X_test["input_embeddings"], y_test)
    print(f"模型准确率: {accuracy:.4f}")
    
    # 保存模型
    model_path = 'knn_model.joblib'
    joblib.dump(knn, model_path)