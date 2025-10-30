from typing import List, Union, Tuple, Dict
import numpy as np
import random

# 自行实现的内容
from router.models.inputs.offline_embedding import offline_embedding
from router.models.inputs.offline_num_tokens import offline_tokens
from router.models.lables.two_steps.output_tokens import lable_generator

random_seed = 2025

def prepare_training_data(config:Dict, index_list:List[int], model_A:str, model_B:str, embedding_model:str, lable_strategy:int) -> Tuple[Dict[str, np.ndarray], Dict, np.ndarray]:
    in_embedding_list = []
    out_embedding_list = []
    num_tokens_list = []
    # 获取到在GSM8K数据集上每一条query对应的embedding
    for idx, time, embedding in offline_embedding(config, index_list, model_A, input_text=True, output_text=False, embedding_model=embedding_model, gen=False):
        in_embedding_list.append(embedding)
    x_in = np.array(in_embedding_list)
    
    # 获取到在GSM8K数据集上model_A每一条output对应的embedding
    for idx, time, embedding in offline_embedding(config, index_list, model_A, input_text=False, output_text=True, embedding_model=embedding_model, gen=False):
        out_embedding_list.append(embedding)
    x_out = np.array(out_embedding_list)
    
    # 获取到在GSM8K数据集上model_A每一条query对应的输出tokens数量
    for data in offline_tokens(config, index_list, model_A):
        num_tokens_list.append(data)
    x_tokens = np.array(num_tokens_list).reshape(-1, 1)
    
    # 获取model_B输出tokens数量的分类标签
    tool = lable_generator(config, index_list, model_B)
    range_dict, lables = tool.gen_lables(strategy=lable_strategy)
    
    x = {"input_embeddings": x_in, "output_embeddings":x_out, "output_tokens":x_tokens}
    
    return (x, lables, range_dict)


def train_test_split(X: Dict[str, np.ndarray], y: np.ndarray, test_ratio: float = 0.3) -> Tuple:
    """
    训练测试集分割
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    n_samples = len(X["input_embeddings"])
    n_test = int(n_samples * test_ratio)

    # 随机打乱索引
    indices = list(range(n_samples))
    random.shuffle(indices)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    x0_train = X["input_embeddings"][train_indices]
    x1_train = X["output_embeddings"][train_indices]
    x2_train = X["output_tokens"][train_indices]
    x0_test  = X["input_embeddings"][test_indices]
    x1_test  = X["output_embeddings"][test_indices]
    x2_test  = X["output_tokens"][test_indices]
    
    return (
        {"input_embeddings": x0_train, "output_embeddings":x1_train, "output_tokens":x2_train}, 
        {"input_embeddings": x0_test, "output_embeddings":x1_test, "output_tokens":x2_test},
        y[train_indices], y[test_indices],
        np.array(train_indices), np.array(test_indices)
    )