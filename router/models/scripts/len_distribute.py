import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import os
import yaml
import numpy as np
from scipy.stats import norm

# 自定义内容
from router.models.scripts.dataset.our_datasets import prepare_training_data, train_test_split, data_require_template, data_choice

# 主函数
def main(embedding_model:str):
    config_file = "/home/ouyk/project/ICDCS/Oracle/config/router_model_GSM8K.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    model_A = "Qwen3-0.6B-temp-0-no-thinking"   # use its embedding as inputs
    model_B = "Qwen3-14B-temp-0-no-thinking"    # use its output_length as lables
    max_tokens = 32768
    record_a = os.path.join(config["Data"]["data_dir"], model_A, "without_outliers.npy")
    index_list = np.load(record_a).tolist()
    data_require = data_require_template.copy()
    data_require["output_tokens_a"] = data_choice.Y

    
    # 准备数据 lable_strategy: 0->Fixed Intervals, 1->Flexible Intervals
    X, Y, range_dict = prepare_training_data(config, index_list, model_A, model_B, embedding_model, data_require=data_require, lable_strategy=2)
    
    y = Y["output_tokens_a"]
    
    # 绘制y的直方图
    plt.hist(y, bins=50, density=True, alpha=0.6, color='b')
    # y 轴是概率密度
    plt.ylabel('Probability Density')
    # x 轴是输出长度
    plt.xlabel('Output Length')
    # 存储
    plt.savefig(f"len_distribute_{embedding_model}.png")
    
    
if __name__ == '__main__':
    embedding_model="Qwen3-Embeddings-0.6B"
    # embedding_model="bert-embedding"
    main(embedding_model)