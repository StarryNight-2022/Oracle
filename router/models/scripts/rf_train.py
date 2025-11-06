# Random Forest

from typing import List, Union, Tuple, Dict
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import random
import yaml
import os
import joblib
import matplotlib.pyplot as plt

# 自行实现的内容
from router.models.scripts.datasets import prepare_training_data, train_test_split

if __name__ == "__main__":
    embedding_model="Qwen3-Embeddings-0.6B"
    config_file = "/home/ouyk/project/ICDCS/Oracle/config/router_model_GSM8K.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
        
    n_neighbors = config["Data"]["labels"]["num_tokens_range_split"]
        
    model_A = "Qwen3-0.6B-temp-0-no-thinking"   # use its embedding as inputs
    # model_B = "Qwen3-14B-temp-0-no-thinking"    # use its output_length as lables
    model_B = "Qwen3-0.6B-temp-0-no-thinking"    # use its output_length as lables
    record = os.path.join(config["Data"]["data_dir"], model_B, "without_outliers.npy")
    index_list = (np.load(record)).tolist()
    data_require = {
        "input_embeddings": True,
        "output_embeddings": False,
        "output_tokens": False,
        "latency": True,
        "output_tokens_label": False,
        "latency_label": True,
    }
    
    # 准备数据 lable_strategy: 0->Fixed Intervals, 1->Flexible Intervals
    X, Y, range_dict = prepare_training_data(config, index_list, model_A, model_B, embedding_model, data_require=data_require, lable_strategy=2)
    
    y = Y["latency"]
    
    # 分割数据
    # X_train, X_test:{"input_embeddings": np.ndarray, "output_embeddings":np.ndarray, "output_tokens":np.ndarray}
    X_train, X_test, y_train, y_test, _, _ = train_test_split(X, y, test_ratio=0.2)
    
    # 实例化KNN模型
    # knn = KNeighborsClassifier(n_neighbors, weights="distance", metric="cosine", algorithm="brute")
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0, oob_score=True)
    
    # train KNN
    rf.fit(X_train["input_embeddings"], y_train)
    
    # 预测与评估
    y_pred = rf.predict(X_test["input_embeddings"])
    print("训练集准确率:", accuracy_score(y_test, y_pred))
    print("OOB分数:", rf.oob_score_)
    
    # 可视化特征重要性
    feature_importances = rf.feature_importances_
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_importances)), feature_importances)
    plt.xlabel('Feature Index')
    plt.ylabel('Importance Score')
    plt.title('Feature Importances from Random Forest')
    plt.savefig('feature_importances.png')
    
