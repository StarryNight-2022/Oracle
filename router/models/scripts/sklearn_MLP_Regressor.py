import matplotlib.pyplot as plt
import os
import yaml
import numpy as np

from router.models.scripts.dataset.our_datasets import prepare_training_data, train_test_split, data_require_template, data_choice

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    config_file = "/home/ouyk/project/ICDCS/Oracle/config/router_model_GSM8K.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
        
    n_classes = config["Data"]["labels"]["num_tokens_range_split"]
    
    embedding_model = "Qwen3-Embeddings-0.6B"
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
    data_require["output_tokens_a"] = data_choice.X
    data_require["output_tokens_b"] = data_choice.Y
    
    # 准备数据 lable_strategy: 0->Fixed Intervals, 1->Flexible Intervals
    X, Y, range_dict = prepare_training_data(config, index_list, model_A, model_B, embedding_model, data_require=data_require, lable_strategy=2)
    
    y = Y["output_tokens_b"].flatten()
    
    # 分割数据
    X_train, X_test, y_train, y_test, _, _ = train_test_split(X, y, test_ratio=0.2)
    
    # 对 X_train, X_test 中的 output_tokens_a 取 -log(output_length/50)；y_train, y_test 也取 -log(output_length/50)
    X_train["output_tokens_a"] = X_train["output_tokens_a"]/50
    X_test["output_tokens_a"] = X_test["output_tokens_a"]/50
    y_train = y_train/50
    y_test = y_test/50
    
    # 需要对 X_train 中多个数组进行拼接，分别是 input_embeddings_a, output_embeddings_a, output_tokens_a
    X_train = np.concatenate((X_train["input_embeddings_a"], X_train["output_embeddings_a"], X_train["output_tokens_a"]), axis=1)
    X_test = np.concatenate((X_test["input_embeddings_a"], X_test["output_embeddings_a"], X_test["output_tokens_a"]), axis=1)
    
    # 特征标准化（MLP 对特征缩放敏感）
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 定义并训练 MLP 回归模型
    mlp = MLPRegressor(
        # hidden_layer_sizes=(100, 100, 100),  # 三个隐藏层，每个 100 个神经元
        hidden_layer_sizes=(512, 256, 128),  # 三个隐藏层，每个 100 个神经元
        activation='relu',              # ReLU 激活函数
        solver='adam',                  # Adam 优化器
        alpha=0.0001,                   # L2 正则化参数
        batch_size=32,                  # 批量大小(预设值:32)
        learning_rate='constant',
        learning_rate_init=1e-3,        # 初始学习率
        max_iter=100,                   # 最大迭代次数
        random_state=42                 # 随机种子
    )
    mlp.fit(X_train, y_train)
    
    # 评估模型
    train_score = mlp.score(X_train, y_train)
    test_score = mlp.score(X_test, y_test)
    print(f"Train R^2 Score: {train_score:.4f}")
    print(f"Test R^2 Score: {test_score:.4f}")
    
    # 计算MAE
    from sklearn.metrics import mean_absolute_error
    train_mae = mean_absolute_error(50*y_train, 50*mlp.predict(X_train))
    test_mae = mean_absolute_error(50*y_test, 50*mlp.predict(X_test))
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