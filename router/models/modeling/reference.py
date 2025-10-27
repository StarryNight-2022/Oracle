"""
kNN路由器实现 - 用于GSM8K数据集的模型选择

功能：
1. 从GSM8K评估结果中提取特征
2. 使用70-30分割训练kNN模型
3. 预测最优模型选择（0.6B vs 14B Qwen）
"""

import json
import random
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import re

class KNNRouter:
    def __init__(self, k: int = 5, random_seed: int = 42, metrics: List[str] = None):
        """
        初始化kNN路由器
        """

        self.k = k
        self.random_seed = random_seed
        self.metrics = metrics
        self.knn_model = None  # 延后到调参后再构建
        self.scaler = StandardScaler()
        self.is_trained = False

        self.model_size = {
            "Deepseek-v3.2-Exp-temp-0-chat": 685,
            "Deepseek-v3.2-Exp-temp-0-reasoner": 685,
            "GPT-4o-mini-temp-0": 8,
            "o4-mini-temp-1": 0,
            "Qwen3-0.6B-temp-0-en-thinking": 0.6,
            "Qwen3-0.6B-temp-0-no-thinking": 0.6,
            "Qwen3-14B-temp-0-en-thinking": 14,
            "Qwen3-14B-temp-0-no-thinking": 14,
        }

    def extract_features(self, question: str) -> List[float]:
        """
        从问题文本中提取特征
        """
        features = []

        # 基础文本特征
        features.append(len(question))  # 字符长度
        features.append(len(question.split()))  # 单词数

        # 数学复杂度特征
        numbers = re.findall(r'\d+', question)
        features.append(len(numbers))  # 数字个数

        # 数学运算符号
        math_ops = re.findall(r'[+\-*/=<>]', question)
        features.append(len(math_ops))  # 数学符号个数

        # 问题类型特征
        features.append(1 if 'how many' in question.lower() else 0)  # 计数问题
        features.append(1 if 'how much' in question.lower() else 0)  # 金额问题
        features.append(1 if 'total' in question.lower() else 0)     # 总计问题
        features.append(1 if 'average' in question.lower() else 0)   # 平均问题

        # 句子复杂度
        sentences = re.split(r'[.!?]+', question)
        features.append(len(sentences))  # 句子数
        features.append(np.mean([len(s.split()) for s in sentences if s.strip()]))  # 平均句长

        return features

    def load_results(self, model_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        results = []
        model_dir = Path(model_path)

        if not model_dir.exists():
            raise FileNotFoundError(f"模型结果目录不存在: {model_path}")

        # 获取所有train_*.jsonl文件
        jsonl_files = sorted(model_dir.glob("train_*.jsonl"))
        if max_samples:
            jsonl_files = jsonl_files[:max_samples]

        for file_path in jsonl_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    line = f.readline().strip()
                    if line:
                        data = json.loads(line)
                        results.append(data)
            except Exception as e:
                print(f"警告: 无法读取文件 {file_path}: {e}")
                continue

        return results

    def load_questions(self, gsm8k_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        加载GSM8K原始问题数据
        """
        questions = []
        with open(gsm8k_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        questions.append(data)
                    except json.JSONDecodeError:
                        continue
        return questions

    def prepare_training_data(
        self,
        small_model_path: str,
        large_model_path: str,
        gsm8k_path: str,
        max_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        准备训练数据
        """
        print("正在加载数据...")

        # 加载模型结果
        small_results = self.load_results(small_model_path, max_samples)
        large_results = self.load_results(large_model_path, max_samples)
        questions = self.load_questions(gsm8k_path, max_samples)

        # 确保数据长度一致
        min_len = min(len(small_results), len(large_results), len(questions))
        small_results = small_results[:min_len]
        large_results = large_results[:min_len]
        questions = questions[:min_len]

        print(f"加载了 {min_len} 个样本")

        # 提取特征和标签
        features = []
        labels = []
        metas = []  # 保存与样本对齐的 small/large 正确性，便于评估下游正确率

        for i in range(min_len):
            # 提取问题特征
            question_text = questions[i]['question']
            question_features = self.extract_features(question_text)

            # 添加模型性能特征
            small_correct = small_results[i].get('correctness', False)
            large_correct = large_results[i].get('correctness', False)

            # 可选：运行时长与输出token数（若字段缺失则回退为0）
            def _get_tokens(rec):
                if rec.get('length_of_output_token_ids') is not None:
                    return int(rec.get('length_of_output_token_ids'))
                if isinstance(rec.get('output_token_ids'), list):
                    return int(len(rec.get('output_token_ids')))
                return 0

            small_runtime = float(small_results[i].get('runtime', 0.0) or 0.0)
            large_runtime = float(large_results[i].get('runtime', 0.0) or 0.0)
            small_tokens = _get_tokens(small_results[i])
            large_tokens = _get_tokens(large_results[i])

            # 组合特征（仅使用在路由时可提前获得的信息，避免标签泄漏）
            combined_features = question_features
            features.append(combined_features)

            # Oracle
            if small_correct:
                labels.append(0)  # 0表示选择小模型
            else:
                labels.append(1)
            # 保存元信息（下游正确率计算需要）
            metas.append([
                int(bool(small_correct)), int(bool(large_correct)),
                float(small_runtime), float(large_runtime),
                int(small_tokens), int(large_tokens)
            ])

        return np.array(features), np.array(labels), np.array(metas)

    def train_test_split(self, X: np.ndarray, y: np.ndarray, metas: np.ndarray, test_size: float = 0.3) -> Tuple:
        """
        训练测试集分割
        """
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        n_samples = len(X)
        n_test = int(n_samples * test_size)

        # 随机打乱索引
        indices = list(range(n_samples))
        random.shuffle(indices)

        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        return (
            X[train_indices], X[test_indices],
            y[train_indices], y[test_indices],
            metas[train_indices], metas[test_indices],
            np.array(train_indices), np.array(test_indices)
        )

    def _build_knn(self, k: int, metric: str) -> KNeighborsClassifier:
        """根据超参数构建KNN分类器。使用距离加权；cosine 需 brute 搜索。"""
        if metric == "cosine":
            return KNeighborsClassifier(n_neighbors=k, weights="distance", metric="cosine", algorithm="brute")
        return KNeighborsClassifier(n_neighbors=k, weights="distance", metric=metric)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        print("正在训练kNN模型...")
        # 标准化 + 训练（使用全部外层训练样本）
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.knn_model = self._build_knn(k=self.k, metric=self.metrics[0])
        self.knn_model.fit(X_train_scaled, y_train)    # 训练了

        self.is_trained = True
        print(f"kNN模型训练完成 (k={self.k}, metric={self.metrics[0]})")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测模型选择

        Args:
            X: 特征矩阵

        Returns:
            预测标签
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train()方法")
        X_scaled = self.scaler.transform(X)
        return self.knn_model.predict(X_scaled)


    def run_full_evaluation(
        self,
        models : List[dict[str,Any]],
        benchmark_path: str,
        benchmark: str,
        max_samples: Optional[int] = None,
        test_size: float = 0.3,
    ) -> Dict[str, Any]:
        """
        运行完整的kNN评估流程

        Args:
            small_model_path: 小模型结果路径
            large_model_path: 大模型结果路径
            gsm8k_path: GSM8K原始数据路径
            max_samples: 最大样本数限制
            test_size: 测试集比例

        Returns:
            完整评估结果
        """
        small,small_model_path, large,large_model_path =self._identify_small_large(models,benchmark)

        print("开始kNN路由器评估...")

        # 准备数据
        X, y, metas = self.prepare_training_data(small_model_path,large_model_path,benchmark_path,max_samples)

        # 分割数据
        X_train, X_test, y_train, y_test, metas_train, metas_test, train_indices, test_indices = self.train_test_split(X, y, metas, test_size)

        print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")

        # 训练模型
        self.train(X_train, y_train)

        # 评估模型
        results = self.evaluate(X_test, y_test, metas_test,small,large)

        # 添加数据集信息
        results['total_samples'] = len(X)
        results['train_samples'] = len(X_train)
        results['test_samples'] = len(X_test)
        results['k_value'] = self.k

        print("评估完成!")
        print(f"label_match_accuracy: {results['label_match_accuracy']:.4f}")
        print(f"knn_achieved_correctness: {results['knn_achieved_correctness']:.4f}")
        print(f"oracle_achieved_correctness: {results['oracle_achieved_correctness']:.4f}")
        print(f"knn_large_model_ratio: {results['knn_large_model_ratio']:.4f}")
        print(f"oracle_large_model_ratio: {results['oracle_large_model_ratio']:.4f}")

        return results