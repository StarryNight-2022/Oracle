# 该脚本用于训练MLP_6模型，使用Qwen3-Embedding-0.6B的input_embedding预测Qwen3-0.6B的output_length;
# regression任务与Classification任务均可尝试 

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import os
import yaml
import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# 自定义内容
from router.models.scripts.dataset.our_datasets import prepare_training_data, train_test_split, data_require_template, data_choice

# 主函数
def main(embedding_model:str):
    config_file = "/home/ouyk/project/ICDCS/Oracle/config/router_model_GSM8K.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
        
    model_A = "Qwen3-0.6B-temp-0-no-thinking"   # use its embedding as inputs
    max_tokens = 32768
    record_a = os.path.join(config["Data"]["data_dir"], model_A, "without_outliers.npy")
    index_list = np.load(record_a).tolist()
    data_require = data_require_template.copy()
    data_require["input_embeddings_a"] = data_choice.X
    data_require["output_tokens_a"] = data_choice.Y
    
    # 准备数据 lable_strategy: 0->Fixed Intervals, 1->Flexible Intervals
    X, Y, range_dict = prepare_training_data(config, index_list, model_A, "", embedding_model, data_require=data_require, lable_strategy=2)
    
    y = Y["output_tokens_a"].squeeze(1)
    
    # 分割数据
    X_train, X_test, y_train, y_test, _, _ = train_test_split(X, y, test_ratio=0.2)
    X_train = X_train["input_embeddings_a"]
    X_test = X_test["input_embeddings_a"]
    
    # 定义并训练 MLP 回归模型
    mlp = MLPRegressor(
        hidden_layer_sizes=(128),  # 三个隐藏层，每个 100 个神经元
        # hidden_layer_sizes=(512, 256, 128),  # 三个隐藏层，每个 100 个神经元
        activation='relu',              # ReLU 激活函数
        solver='adam',                  # Adam 优化器
        alpha=0.0001,                   # L2 正则化参数
        batch_size=32,                  # 批量大小(预设值:32)
        learning_rate='constant',
        learning_rate_init=1e-3,        # 初始学习率
        max_iter=100,                   # 最大迭代次数
        random_state=42                 # 随机种子
    )
    print(X_train.shape, y_train.shape)
    mlp.fit(X_train, y_train)
    
    # 评估模型
    train_score = mlp.score(X_train, y_train)
    test_score = mlp.score(X_test, y_test)
    print(f"Train R^2 Score: {train_score:.4f}")
    print(f"Test R^2 Score: {test_score:.4f}")
    
    # 计算MAE
    from sklearn.metrics import mean_absolute_error
    train_mae = mean_absolute_error(y_train, mlp.predict(X_train))
    test_mae = mean_absolute_error(y_test, mlp.predict(X_test))
    print(f"Train MAE: {train_mae:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    # 绘制Train与Evaluate的散点图，两个子图
    plt.figure(figsize=(12, 5))
    
    # 训练集散点图
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, mlp.predict(X_train), alpha=0.5)
    plt.plot([y_train.min(), y_train.max()], 
             [y_train.min(), y_train.max()], 'r--', lw=2)
    plt.xlabel("Actual Output Tokens")
    plt.ylabel("Predicted Output Tokens")
    plt.title(f"Train: Actual vs Predicted (R^2: {train_score:.4f}, MAE: {train_mae:.4f})")
    plt.grid(True, alpha=0.3)
    
    # 测试集散点图
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, mlp.predict(X_test), alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], 
             [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Output Tokens")
    plt.ylabel("Predicted Output Tokens")
    plt.title(f"Test: Actual vs Predicted (R^2: {test_score:.4f}, MAE: {test_mae:.4f})")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    plt.savefig("mlp_regressor_eval.png")
    
    

if __name__ == '__main__':
    embedding_model="Qwen3-Embeddings-0.6B"
    # embedding_model="bert-embedding"
    main(embedding_model)