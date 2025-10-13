from typing import Any, Dict, List


class Simple3Router:
    def __init__(self):
        # Use same size map as in RandomRouter for consistency
        self.model_size: Dict[str, float] = {
            "Deepseek-v3.2-Exp-temp-0-chat": 685,
            "Deepseek-v3.2-Exp-temp-0-reasoner": 685,
            "GPT-4o-mini-temp-0": 8,
            "o4-mini-temp-1": 0,
            "Qwen3-0.6B-temp-0-en-thinking": 0.6,
            "Qwen3-0.6B-temp-0-no-thinking": 0.6,
            "Qwen3-14B-temp-0-en-thinking": 14,
            "Qwen3-14B-temp-0-no-thinking": 14,
        }

    def _rank_by_size(self, results: Dict[str, Any]) -> List[str]:
        sizes: List[tuple[str, float]] = []
        for name in results.keys():
            sizes.append((name, float(self.model_size.get(name, float("inf")))))
        sizes.sort(key=lambda x: (x[1], x[0]))
        return [n for n, _ in sizes]

    def oracle3LLMs(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Per-query oracle decision across provided models (expect 3 models).
        Selection:
          - If any correct, choose the smallest correct (by model size)
          - If none correct, choose the largest

        Return format matches RandomRouter.random2LLMs:
          {"model", "correctness", "latency", "output_tokens"}
        """
        if not results:
            return {"model": None, "correctness": False, "latency": 0.0, "output_tokens": 0}

        ranked = self._rank_by_size(results)
        # smallest -> largest
        smallest_to_largest = ranked
        largest_model = smallest_to_largest[-1]

        # Find smallest correct
        chosen_name = None
        for name in smallest_to_largest:
            if bool(results[name].get("correctness", False)):
                chosen_name = name
                break
        if chosen_name is None:
            chosen_name = largest_model

        chosen = results[chosen_name]
        return {
            "model": chosen_name,
            "correctness": bool(chosen.get("correctness", False)),
            "latency": float(chosen.get("runtime", 0.0)),
            "output_tokens": int(chosen.get("length_of_output_token_ids", 0)),
        }


