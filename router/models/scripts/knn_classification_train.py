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
from router.models.scripts.dataset.our_datasets import prepare_training_data, train_test_split, data_require_template, data_choice

if __name__ == "__main__":
    embedding_model="Qwen3-Embeddings-0.6B"
    config_file = "/home/ouyk/project/ICDCS/Oracle/config/router_model_GSM8K.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
        
    n_neighbors = config["Data"]["labels"]["num_tokens_range_split"]
        
    model_A = "Qwen3-0.6B-temp-0-no-thinking"   # use its embedding as inputs
    model_B = "Qwen3-14B-temp-0-no-thinking"    # use its output_length as lables
    max_tokens = 32768
    record_a = os.path.join(config["Data"]["data_dir"], model_A, "without_outliers.npy")
    record_b = os.path.join(config["Data"]["data_dir"], model_B, "without_outliers.npy")
    index_list_a = np.load(record_a).tolist()
    index_list_b = np.load(record_b).tolist()
    index_list = list(set(index_list_a) & set(index_list_b)) # 取交集
    data_require = data_require_template.copy()
    data_require["input_embeddings_a"] = data_choice.X
    # data_require["output_embeddings_a"] = data_choice.X
    data_require["latency_label_b"] = data_choice.Y
    # data_require["output_tokens_b"] = data_choice.Y
    
    # 准备数据 lable_strategy: 0->Fixed Intervals, 1->Flexible Intervals
    X, Y, range_dict = prepare_training_data(config, index_list, model_A, model_B, embedding_model, data_require=data_require, lable_strategy=1)
    print(f"range_dict: {range_dict}")
    y = Y["latency_label_b"].flatten()
    
    # 分割数据
    # X_train, X_test:{"input_embeddings": np.ndarray, "output_embeddings":np.ndarray, "output_tokens":np.ndarray}
    X_train, X_test, y_train, y_test, _, _ = train_test_split(X, y, test_ratio=0.2)
    
    # 实例化KNN模型
    knn = KNeighborsClassifier(n_neighbors, weights="distance", metric="cosine", algorithm="brute")
    # knn = KNeighborsClassifier(n_neighbors, weights="distance", metric="minkowski")
    
    # train KNN
    knn.fit(X_train["input_embeddings_a"], y_train)
    
    # evaluate KNN
    accuracy = knn.score(X_test["input_embeddings_a"], y_test)
    print(f"模型准确率: {accuracy:.4f}")
    
    # 保存模型
    model_path = 'knn_model.joblib'
    joblib.dump(knn, model_path)