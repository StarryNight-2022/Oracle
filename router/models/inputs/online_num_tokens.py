# 直接实时部署一个Qwen3-0.6B小模型进行预先推理，得到输出长度。
# 需要统计调用过程时间开销
from typing import Dict, List
from router.utils.Benchmarks.benchmarks import load_dataset
