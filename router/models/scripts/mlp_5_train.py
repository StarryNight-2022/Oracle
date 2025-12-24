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

# 自定义内容
from router.models.modeling.modeling import MLP_5
from router.models.scripts.dataset.our_datasets import prepare_training_data, train_test_split, data_require_template, data_choice


# 训练函数
def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001, weight_decay=1e-5, device="cpu"):
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_total = 0
        
        for inputs_prompt_embed, outputs_embed, targets in train_loader:
            inputs_prompt_embed = inputs_prompt_embed.to(device)
            outputs_embed = outputs_embed.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs_prompt_embed, outputs_embed)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_total += targets.size(0)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_total = 0
        
        with torch.no_grad():
            for inputs_prompt_embed, outputs_embed, targets in val_loader:
                inputs_prompt_embed = inputs_prompt_embed.to(device)
                outputs_embed = outputs_embed.to(device)
                targets = targets.to(device)
                outputs = model(inputs_prompt_embed, outputs_embed)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_total += targets.size(0)
        
        # 计算平均损失和准确率
        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {train_loss_avg:.4f}')
            print(f'Val Loss: {val_loss_avg:.4f}')
            print('-' * 50)
    
    return train_losses, val_losses

# 主函数
def main(embedding_model:str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    config_file = "/home/ouyk/project/ICDCS/Oracle/config/router_model_GSM8K.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
        
    n_classes = config["Data"]["labels"]["num_tokens_range_split"]
    
    # 超参数设置()
    if embedding_model == "Qwen3-Embeddings-0.6B":
        dtype=torch.float32
        input_size = 1024
        num_epochs = 400
        batch_size = 32
        learning_rate = 1e-4
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
    
    # 转换为TensorDataset
    train_dataset = TensorDataset(torch.tensor(X_train["input_embeddings_a"], dtype=torch.float32), torch.tensor(X_train["output_embeddings_a"], dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    val_dataset = TensorDataset(torch.tensor(X_test["input_embeddings_a"], dtype=torch.float32), torch.tensor(X_test["output_embeddings_a"], dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32).unsqueeze(1))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
     # 初始化模型
    model = MLP_5(embedding_dim=input_size, device=device, dtype=dtype)
    print(model)
    # model.load_state_dict(torch.load("/home/ouyk/project/ICDCS/Oracle/mlp_model_2.1.pth"))
    
    # 训练模型
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, num_epochs, learning_rate, weight_decay, device
    )
    
    # 保存模型
    model_path = 'mlp_model.pth'
    torch.save(model.state_dict(), model_path)
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    

if __name__ == '__main__':
    embedding_model="Qwen3-Embeddings-0.6B"
    # embedding_model="bert-embedding"
    main(embedding_model)