from typing import List, Union, Tuple, Dict
import numpy as np
import random
import datasets
import os

# 自行实现的内容
from router.models.inputs.queries import queries
from router.models.inputs.offline_embedding import offline_embedding
from router.models.inputs.offline_num_tokens import offline_tokens
from router.models.inputs.offline_latency import offline_latency
from router.models.lables.two_steps.output_tokens import label_generator as tokens_label_generator
from router.models.lables.two_steps.latency import label_generator as latency_label_generator

random_seed = 2025

# data_require = {
#     "input_embeddings": True,
#     "output_embeddings": False,
#     "output_tokens": False,
#     "latency": True,
#     "output_tokens_label": False,
#     "latency_label": True,
# }

def prepare_training_data(config:Dict, index_list:List[int], model_A:str, model_B:str, embedding_model:str, data_require:Dict[str, bool], lable_strategy:int) -> Tuple[Dict[str, np.ndarray], Dict, Dict]:
    x = {}
    label = {}
    range_dict = {}
    # 获取到在GSM8K数据集上每一条query对应的embedding
    if data_require["input_embeddings"]:
        in_embedding_list = []
        for idx, time, embedding in offline_embedding(config, index_list, model_A, input_text=True, output_text=False, embedding_model=embedding_model, gen=False):
            in_embedding_list.append(embedding)
        x_in = np.array(in_embedding_list)
        x["input_embeddings"] = x_in
    
    # 获取到在GSM8K数据集上model_A每一条output对应的embedding
    if data_require["output_embeddings"]:
        out_embedding_list = []    
        for idx, time, embedding in offline_embedding(config, index_list, model_A, input_text=False, output_text=True, embedding_model=embedding_model, gen=False):
            out_embedding_list.append(embedding)
        x_out = np.array(out_embedding_list)
        x["output_embeddings"] = x_out
    
    # 获取到在GSM8K数据集上model_A每一条query对应的输出tokens数量
    if data_require["output_tokens"]:
        num_tokens_list = []
        for data in offline_tokens(config, index_list, model_A):
            num_tokens_list.append(data)
        x_tokens = np.array(num_tokens_list).reshape(-1, 1)
        x["output_tokens"] = x_tokens
    
    # 获取model_B输出tokens数量的分类标签
    if data_require["output_tokens_label"]:
        tool = tokens_label_generator(config, index_list, model_B)
        range_dict, output_tokens_lables = tool.gen_lables(strategy=lable_strategy)
        label["output_tokens"] = output_tokens_lables
        range_dict["output_tokens"] = range_dict
    
    # 获取到在GSM8K数据集上model_A每一条query对应的推理latency
    if data_require["latency"]:
        latency_list = []
        for data in offline_latency(config, index_list, model_A):
            latency_list.append(data)
        x_latency = np.array(latency_list).reshape(-1, 1)
        x["latency"] = x_latency
        
    # 获取model_B推理latency的分类标签
    if data_require["latency_label"]:
        tool = latency_label_generator(config, index_list, model_B)
        range_dict, latency_labels = tool.gen_lables(strategy=lable_strategy)
        label["latency"] = latency_labels
        range_dict["latency"] = range_dict
    
    return (x, label, range_dict) 


def train_test_split(X: Dict[str, np.ndarray], y: np.ndarray, test_ratio: float = 0.3) -> Tuple:
    """
    训练测试集分割
    """
    train_dict: Dict = {}
    test_dict: Dict = {}
    
    random.seed(random_seed)
    np.random.seed(random_seed)

    n_samples = len(X["input_embeddings"])
    n_test = int(n_samples * test_ratio)

    # 随机打乱索引
    indices = list(range(n_samples))
    random.shuffle(indices)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    for key in X.keys():
        x_train = X[key][train_indices]
        x_test  = X[key][test_indices]
        train_dict[key] = x_train
        test_dict[key] = x_test
    
    return (
        train_dict, test_dict,
        y[train_indices], y[test_indices],
        np.array(train_indices), np.array(test_indices)
    )

def gen_fine_tuning_data(config:Dict, index_list:List[int], model_A:str, model_B:str, lable_strategy:int, test_ratio:float, save_dir:str):
    # List[List[str, int]]
    query = []
    train_samples: List[List[str, int]] = []
    test_samples: List[List[str, int]] = []
    
    # input: 模型A的prompts
    for data in queries(config, index_list, model_A):
        query.append(data)
    
    # label: 模型B的tokens_label
    tool = tokens_label_generator(config, index_list, model_B)
    range_dict, labels = tool.gen_lables(strategy=lable_strategy)
    
    # label: 模型B的latency_label
    # tool = latency_label_generator(config, index_list, model_B)
    # range_dict, labels = tool.gen_lables(strategy=lable_strategy)
    
    # 切分
    random.seed(random_seed)
    np.random.seed(random_seed)

    n_samples = len(index_list)
    n_test = int(n_samples * test_ratio)

    # 随机打乱索引
    indices = list(range(n_samples))
    
    random.shuffle(indices)
    
    train_indices = indices[n_test:]
    test_indices = indices[:n_test]
    
    for idx in train_indices:
        train_samples.append([query[idx], labels[idx]])
        
    for idx in test_indices:
        test_samples.append([query[idx], labels[idx]])
    
    # 保存为npy文件
    np.save(os.path.join(save_dir ,'train_data.npy'), np.array(train_samples, dtype=object))
    np.save(os.path.join(save_dir ,'test_data.npy'), np.array(test_samples, dtype=object))
    print("数据集创建完成！")