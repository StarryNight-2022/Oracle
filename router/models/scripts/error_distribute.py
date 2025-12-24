# 首先使用线性回归
# 接着学习实际值与线性回归之间的误差

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
    record_b = os.path.join(config["Data"]["data_dir"], model_B, "without_outliers.npy")
    index_list_a = np.load(record_a).tolist()
    index_list_b = np.load(record_b).tolist()
    index_list = list(set(index_list_a) & set(index_list_b)) # 取交集
    data_require = data_require_template.copy()
    data_require["input_embeddings_a"] = data_choice.X
    data_require["output_embeddings_a"] = data_choice.X
    data_require["output_tokens_a"] = data_choice.Y
    data_require["output_tokens_b"] = data_choice.Y
    
    # 准备数据 lable_strategy: 0->Fixed Intervals, 1->Flexible Intervals
    X, Y, range_dict = prepare_training_data(config, index_list, model_A, model_B, embedding_model, data_require=data_require, lable_strategy=2)
    
    y_real = Y["output_tokens_b"]
    y_hat = Y["output_tokens_a"] * 0.7093 + 93.4910  # 使用线性回归的结果
    
    y = (y_real - y_hat).flatten()
    
    # 分割数据
    X_train, X_test, y_train, y_test, _, _ = train_test_split(X, y, test_ratio=0.2)
    
    # 使用正态分布拟合y的分布
    mu, std = norm.fit(y)
    print(f"Fitted normal distribution: mu = {mu}, std = {std}")
    
    # 绘制y的分布直方图，添加正态分布曲线，绘制典型置信区间（如95%置信区间）
    plt.hist(y, bins=50, alpha=0.75, density=True)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    # 绘制95%置信区间
    plt.fill_between(x, p, where=(x >= mu - 1.96 * std) & (x <= mu + 1.96 * std), color='gray', alpha=0.5)
    print(f"95% confidence interval: [{mu - 1.96 * std:.4f}, {mu + 1.96 * std:.4f}]")
    plt.title('Distribution of y(y_real - y_hat)')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    plt.savefig("error_distribution.png")
    
if __name__ == '__main__':
    embedding_model="Qwen3-Embeddings-0.6B"
    # embedding_model="bert-embedding"
    main(embedding_model)