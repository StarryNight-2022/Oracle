import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import os
import yaml
import numpy as np

# 自定义内容
from router.models.modeling.modeling import MLP
from router.models.scripts.datasets import prepare_training_data, train_test_split

# 训练函数
def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001, device="cpu"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_total += targets.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == targets).sum().item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.data, targets)
                val_loss += loss.item()
                val_total += targets.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
        
        # 计算平均损失和准确率
        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%')
            print('-' * 50)
    
    return train_losses, val_losses, train_accs, val_accs

# 主函数
def main():
    device = torch.device("cuda:1")
    embedding_model="Qwen3-Embeddings-0.6B"
    config_file = "/home/ouyk/project/ICDCS/Oracle/config/router_model.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
        
    n_classes = config["Data"]["labels"]["num_tokens_range_split"]
    
    # 超参数设置
    input_size = 1024
    output_size = n_classes
    num_epochs = 100
    batch_size = 32
    learning_rate = 0.001
        
    model_A = "Qwen3-0.6B-temp-0-no-thinking"   # use its embedding as inputs
    # model_B = "Qwen3-14B-temp-0-no-thinking"    # use its output_length as lables
    model_B = "Qwen3-0.6B-temp-0-no-thinking"    # use its output_length as lables
    record = os.path.join(config["Data"]["data_dir"], model_B, "without_outliers.npy")
    index_list = (np.load(record)).tolist()
    
    # 准备数据
    x, y, range_dict = prepare_training_data(config, index_list, model_A, model_B, embedding_model)
    print("range_dict:\n", range_dict)
    print("length of X:", len(x))
    print("length of y:", len(y))
    
    # 分割数据
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(x, y, test_ratio=0.3)
    
    # 转换为TensorDataset
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型
    model = MLP(input_size, output_size).to(device)
    print(model)
    
    # 训练模型
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, num_epochs, learning_rate, device
    )
    
    # 保存模型
    model_path = 'mlp_model.pth'
    torch.save(model.state_dict(), model_path)
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')

if __name__ == '__main__':
    main()