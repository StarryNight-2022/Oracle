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

# 自行实现的内容
from router.models.inputs.offline_embedding import offline_embedding
from router.models.lables.gen_lables import lable_generator

random_seed = 2025

def prepare_training_data(config:Dict, index_list:List[int], model_A:str, model_B:str, embedding_model:str) -> Tuple[np.ndarray, Dict, np.ndarray]:
    embedding_list = []
    # 获取到在GSM8K数据集上每一条query对应的num_tokens
    for idx, time, embedding in offline_embedding(config, index_list, model_A, embedding_model, gen=False):
        embedding_list.append(embedding)
    x = np.array(embedding_list)
    
    tool = lable_generator(config, index_list, model_B)
    range_dict, lables = tool.gen_lables(strategy=1)
    
    return (x, lables, range_dict)


def train_test_split(X: np.ndarray, y: np.ndarray, test_ratio: float = 0.3) -> Tuple:
    """
    训练测试集分割
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    n_samples = len(X)
    n_test = int(n_samples * test_ratio)

    # 随机打乱索引
    indices = list(range(n_samples))
    random.shuffle(indices)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    return (
        X[train_indices], X[test_indices],
        y[train_indices], y[test_indices],
        np.array(train_indices), np.array(test_indices)
    )

if __name__ == "__main__":
    embedding_model="Qwen3-Embeddings-0.6B"
    config_file = "/home/ouyk/project/ICDCS/Oracle/config/router_model.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
        
    model_A = "Qwen3-0.6B-temp-0-no-thinking"   # use its embedding as inputs
    model_B = "Qwen3-14B-temp-0-no-thinking"    # use its output_length as lables 
    record = os.path.join(config["Data"]["data_dir"], model_B, "without_outliers.npy")
    index_list = (np.load(record)).tolist()
    
    # 准备数据
    x, y, range_dict = prepare_training_data(config, index_list, model_A, model_B, embedding_model)
    print("range_dict:\n", range_dict)
    print("length of X:", len(x))
    print("length of y:", len(y))
    
    # 分割数据
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(x, y, test_ratio=0.3)
    
    # 训练KNN模型
    knn = KNeighborsClassifier(n_neighbors=5)