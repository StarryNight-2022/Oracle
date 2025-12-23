# 该文件负责使用线性回归预测输出长度，输入为Qwen3-0.6B的输出长度，输出为其它模型的输出长度
# 测试结果表明，对于同一个模型，输出长度与延迟之间的关系是线性的；
# 但是对于两个不同LLM的输出长度，线性关系并不显著。
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

# 自行实现的内容
from router.models.scripts.dataset.our_datasets import prepare_training_data, data_require_template, data_choice

# 导入自定义的train_test_split并重命名，避免冲突
from router.models.scripts.dataset.our_datasets import train_test_split as our_train_test_split

if __name__ == "__main__":
    import yaml
    import matplotlib.pyplot as plt
    
    embedding_model = "Qwen3-Embeddings-0.6B"
    config_file = "/home/ouyk/project/ICDCS/Oracle/config/router_model_GSM8K.yaml"
    
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
        
    n_neighbors = config["Data"]["labels"]["num_tokens_range_split"]
        
    model_A = "Qwen3-0.6B-temp-0-no-thinking"   # use its embedding as inputs
    model_B = "Qwen3-14B-temp-0-no-thinking"    # use its output_length as lables
    max_tokens = 32768
    record_a = os.path.join(config["Data"]["data_dir"], model_A, "without_outliers.npy")
    record_b = os.path.join(config["Data"]["data_dir"], model_B, "without_outliers.npy")
    index_list_a = np.load(record_a).tolist()
    index_list_b = np.load(record_b).tolist()
    index_list = list(set(index_list_a) & set(index_list_b)) # 取交集
    data_require = data_require_template.copy()
    data_require["output_tokens_a"] = data_choice.X
    data_require["output_tokens_b"] = data_choice.Y
    
    # 准备数据 label_strategy: 0->Fixed Intervals, 1->Flexible Intervals
    X, Y, range_dict = prepare_training_data(config, index_list, model_A, model_B, embedding_model, data_require=data_require, lable_strategy=2)
    
    y = Y["output_tokens_b"]
    
    # 确保X["output_tokens_a"]是二维数组
    X_features = X["output_tokens_a"]
    
    # 检查并重塑数组形状
    if len(X_features.shape) == 1:
        print(f"重塑X_features形状: 从{X_features.shape}到({X_features.shape[0]}, 1)")
        X_features = X_features.reshape(-1, 1)
    else:
        print(f"X_features形状: {X_features.shape}")
    
    # 确保y是一维数组
    if len(y.shape) > 1:
        print(f"重塑y形状: 从{y.shape}到({y.shape[0]},)")
        y = y.ravel()  # 或者 y = y.flatten()
    
    # 使用sklearn的train_test_split
    X_train, X_test, y_train, y_test = sklearn_train_test_split(
        X_features, y, test_size=0.2, random_state=42
    )
    
    print(f"训练集形状: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"测试集形状: X_test={X_test.shape}, y_test={y_test.shape}")
    
    # 训练线性回归模型
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    
    # 线性回归模型系数
    print(f"系数 (coef): {reg.coef_}")
    print(f"截距 (intercept): {reg.intercept_}")
    
    # 模型评估
    y_pred = reg.predict(X_test)
    
    # 计算性能指标
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"均方误差 (MSE): {mse:.2f}")
    print(f"平均绝对误差 (MAE): {mae:.2f}")
    print(f"R²分数: {r2:.4f}")
    
    # 输出一些预测示例
    print("\n预测示例 (前10个):")
    print("实际值 -> 预测值")
    for i in range(min(10, len(y_test))):
        print(f"{y_test[i]:.1f} -> {y_pred[i]:.1f}")
    
    # 绘制预测值与实际值的散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, label='predict')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='ideal')
    
    # 为了绘制平滑的回归线，我们需要生成一些点
    # 首先，确保数据是一维的
    y_test_flat = y_test.flatten() if len(y_test.shape) > 1 else y_test
    y_pred_flat = y_pred.flatten() if len(y_pred.shape) > 1 else y_pred
    
    # 按实际值排序
    sorted_indices = np.argsort(y_test_flat)
    y_test_sorted = y_test_flat[sorted_indices]
    y_pred_sorted = y_pred_flat[sorted_indices]
    
    plt.plot(y_test_sorted, y_pred_sorted, 'g-', lw=1, alpha=0.7, label='predict line')
    
    plt.xlabel('actual output tokens')
    plt.ylabel('predict output tokens')
    plt.title('actual vs predict output tokens (linear regression)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加文本显示回归方程
    if len(reg.coef_) == 1:
        equation_text = f"y = {reg.coef_[0]:.4f}x + {reg.intercept_:.4f}"
    else:
        # 如果有多个特征，显示第一个特征的系数
        equation_text = f"y = {reg.coef_[0]:.4f}x₁ + ... + {reg.intercept_:.4f}"
    
    plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 添加性能指标到图中
    metrics_text = f"MSE = {mse:.2f}\nMAE = {mae:.2f}\nR² = {r2:.4f}"
    plt.text(0.05, 0.85, metrics_text, transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig("actual_vs_predicted_output_tokens.png", dpi=300)
    
    # # 可选：保存模型
    # joblib.dump(reg, "linear_regression_model.pkl")
    # print("模型已保存到 linear_regression_model.pkl")