from typing import List, Union, Tuple, Dict
import numpy as np
import random

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