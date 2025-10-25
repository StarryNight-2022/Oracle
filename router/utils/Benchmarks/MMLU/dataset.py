# 来自/home/ouyk/project/ICDCS/Oracle/profile/scripts/run/qwen3-0.6B-temp-0-en-think.py的load_jsonl()方法
import json
from typing import List, Dict, Any, Optional

# 装载jsonl格式的数据集
def load_MMLU(path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """逐行读取 JSONL 文件。若提供 max_samples 则截断。"""
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
                if max_samples is not None and len(data) >= max_samples:
                    break
            except json.JSONDecodeError:
                # 跳过坏行
                continue
    return data