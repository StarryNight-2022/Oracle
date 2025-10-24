import random
from typing import Dict, Any, List, Tuple, Optional

# Fixed-seed RNG for reproducibility across calls
_RNG = random.Random(42)

class RandomRouter:
    def __init__(self):
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

    def _rank_models_by_size(self, results: Dict[str, Any]) -> List[str]:
        # Determine model names ranked by size (smallest to largest)
        sizes: List[Tuple[str, float]] = []
        for name in results.keys():
            sizes.append((name, float(self.model_size.get(name, float("inf")))))
        # Sort ascending by size, tie-break by name for determinism
        sizes.sort(key=lambda x: (x[1], x[0]))
        return [name for name, _ in sizes]

    def randomLLMs(
            self,
            results: Dict[str, Any],
            percentage_to_larges: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        随机路由到多个模型中的一个

        Args:
            results: 模型结果字典
            percentage_to_large: 从大到小的模型选择比例列表，如果为None则使用均匀分布

        Returns:
            包含选择模型信息的字典
        """
        if not results:
            return {"model": None, "correctness": False, "latency": 0.0, "output_tokens": 0}

        ranked_models = self._rank_models_by_size(results)  # 从小到大排序
        num_models = len(ranked_models)

        if percentage_to_larges is None:
            # 均匀分布
            chosen_model = _RNG.choice(ranked_models)
        else:
            # 验证输入
            percentage_to_large_copy = percentage_to_larges.copy()
            if not all(0 <= p <= 100 for p in percentage_to_large_copy):
                raise ValueError("所有比例必须在[0, 100]范围内")

            if len(percentage_to_large_copy) < num_models:
                # 检查比例总和是否为100
                total_percentage = sum(percentage_to_large_copy)
                remaining_percentage = 100.0 - total_percentage
                equal_share = remaining_percentage / (num_models - len(percentage_to_large_copy))
                for _ in range(num_models - len(percentage_to_large_copy)):
                    percentage_to_large_copy.append(equal_share)

            # 将比例转换为权重（从大到小对应模型）
            # percentage_to_large[0] 对应最大的模型，percentage_to_large[-1] 对应最小的模型
            weights = [p / 100.0 for p in reversed(percentage_to_large_copy)]  # 反转以匹配从小到大排序的模型

            # 根据权重随机选择模型
            chosen_idx = _RNG.choices(range(num_models), weights=weights)[0]
            chosen_model = ranked_models[chosen_idx]

        chosen_result = results[chosen_model]
        return {
            "model": chosen_model,
            "correctness": chosen_result.get("correctness", False),
            "latency": chosen_result.get("runtime", 0.0),
            "output_tokens": chosen_result.get("length_of_output_token_ids", 0),
        }
