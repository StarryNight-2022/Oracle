# 该脚本用于训练MLP_4_Classification模型，模型结合了prompt_embed、output_embedding以及output_length三个输入，目标是预测输出长度

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import os
import yaml
import numpy as np

# 自定义内容
from router.models.modeling.modeling import MLP_4_Classification
from router.models.scripts.dataset.our_datasets import prepare_training_data, train_test_split, data_require_template, data_choice

# 评估函数
def eval_model(model, eval_loader, device="cpu"):
    model.eval()
    target_list = []
    output_list = []
    
    with torch.no_grad():
        for inputs_prompt_embed, inputs_output_embed, inputs_output_length, targets in eval_loader:
            inputs_prompt_embed = inputs_prompt_embed.to(device)
            inputs_output_embed = inputs_output_embed.to(device)
            inputs_output_length = inputs_output_length.to(device)
            targets = targets.to(device)
            outputs = model(inputs_prompt_embed, inputs_output_embed, inputs_output_length)
            
            target_list.append(targets.cpu().numpy())
            output_list.append(outputs.cpu().numpy())
    
    # 合并数据
    targets_all = np.concatenate(target_list)
    outputs_all = np.concatenate(output_list)
    
    # 计算分类准确率
    correct = (np.round(outputs_all) == targets_all).sum()
    accuracy = correct / len(targets_all)
    print(f"Classification Accuracy: {accuracy:.4f}")
    
    # 绘制散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(targets_all, outputs_all, alpha=0.5)
    plt.plot([targets_all.min(), targets_all.max()], 
             [targets_all.min(), targets_all.max()], 'r--', lw=2)
    plt.xlabel("Actual Output Tokens")
    plt.ylabel("Predicted Output Tokens")
    plt.grid(True, alpha=0.3)
    plt.savefig("mlp_4_eval.png")
    plt.close()

# 主函数
def main(embedding_model:str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config_file = "/home/ouyk/project/ICDCS/Oracle/config/router_model_GSM8K.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
        
    n_classes = config["Data"]["labels"]["num_tokens_range_split"]
    # n_classes = config["Data"]["labels"]["latency_range_split"]
    
    # 超参数设置()
    if embedding_model == "Qwen3-Embeddings-0.6B":
        dtype=torch.float32
        input_size = 1024
        num_epochs = 1000
        batch_size = 32
        learning_rate = 1e-4
        alpha = 0.05
        weight_decay = 1e-5
    elif embedding_model == "bert-embedding":
        dtype=torch.float32
        input_size = 768
        num_epochs = 1000
        batch_size = 32
        learning_rate = 1e-5
        alpha = 0.1
        weight_decay = 1e-5
    else:
        raise NotImplementedError(f"Don't support {embedding_model}")
        
    model_A = "Qwen3-0.6B-temp-0-no-thinking"   # use its embedding as inputs
    model_B = "Qwen3-14B-temp-0-no-thinking"    # use its output_length as lables
    record_a = os.path.join(config["Data"]["data_dir"], model_A, "without_outliers.npy")
    record_b = os.path.join(config["Data"]["data_dir"], model_B, "without_outliers.npy")
    index_list_a = np.load(record_a).tolist()
    index_list_b = np.load(record_b).tolist()
    index_list = list(set(index_list_a) & set(index_list_b)) # 取交集
    data_require = data_require_template.copy()
    data_require["input_embeddings_a"] = data_choice.X
    data_require["output_embeddings_a"] = data_choice.X
    data_require["output_tokens_a"] = data_choice.X
    data_require["output_tokens_label_b"] = data_choice.Y
    
    # 准备数据 lable_strategy: 0->Fixed Intervals, 1->Flexible Intervals
    X, Y, range_dict = prepare_training_data(config, index_list, model_A, model_B, embedding_model, data_require=data_require, lable_strategy=1)
    print(f"range_dict: {range_dict}")
    y = Y["output_tokens_label_b"]
    
    # 分割数据
    _, X_test, _, y_test, _, _ = train_test_split(X, y, test_ratio=0.2)
    
    # 转换为TensorDataset
    val_dataset = TensorDataset(torch.tensor(X_test["input_embeddings_a"], dtype=torch.float32), torch.tensor(X_test["output_embeddings_a"], dtype=torch.float32), torch.tensor(X_test["output_tokens_a"], dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    
    eval_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型
    model = MLP_4_Classification(config=config, embedding_dim=input_size, device=device, dtype=dtype)
    model.load_state_dict(torch.load("/home/ouyk/project/ICDCS/Oracle/mlp_model.pth"))
    
    # 评估模型
    eval_model(
        model, eval_loader, device
    )
    

if __name__ == '__main__':
    embedding_model="Qwen3-Embeddings-0.6B"
    # embedding_model="bert-embedding"
    main(embedding_model)