import random
from typing import Dict, Any, List, Tuple
from utils.config import model_size

# Fixed-seed RNG for reproducibility across calls
_RNG = random.Random(42)

class RandomRouter:
    def __init__(self):
        self.model_size = model_size

    def _identify_small_large(self, results: Dict[str, Any]) -> Tuple[str, str]:
        # Determine small and large model names based on known sizes; fallback to name sort
        sizes: List[Tuple[str, float]] = []
        for name in results.keys():
            sizes.append((name, float(self.model_size.get(name, float("inf")))))
        # Sort ascending by size, tie-break by name for determinism
        sizes.sort(key=lambda x: (x[1], x[0]))
        small = sizes[0][0]
        large = sizes[-1][0]
        return small, large

    def random2LLMs(
        self,
        results: Dict[str, Any],
        percentage_to_large: float,
    ) -> Dict[str, Any]:
        assert 0 <= percentage_to_large <= 100, "percentage_to_large must be in [0, 100]"
        small, large = self._identify_small_large(results)
        route_to_large = _RNG.random() < (percentage_to_large / 100.0)
        chosen = large if route_to_large else small
        chosen_result = results[chosen]
        return {
            "model": chosen,
            "correctness": chosen_result.get("correctness", False),
            "latency": chosen_result.get("runtime", 0.0),
            "output_tokens": chosen_result.get("length_of_output_token_ids", 0),
        }
