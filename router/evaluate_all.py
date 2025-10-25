# 负责读取./input目录下所有的数据，对每一条模型的回复进行正确性判断，并将判断结果(Ture/False)添加到每一个jsonl文件中。
# 对每个Benchmark进行遍历
import os
from typing import List, Dict, Any
import argparse
import yaml
from pathlib import Path
import json
import yaml
import traceback
from tqdm import tqdm

# 自定义的包
from utils.Benchmarks.benchmarks import load_dataset, parse_answer

def parse_args():
    parser = argparse.ArgumentParser(description="Process some parameters.")
    
    parser.add_argument('--benchmark', 
                        type=str, 
                        default="GSM8K",
                        choices=["GSM8K","MMLU"],
                        help="Specify the benchmark")
    
    parser.add_argument('--input_dir', 
                        type=str, 
                        default="/home/ouyk/project/ICDCS/Oracle/input",
                        help="Specify the input file storage path")

    args = parser.parse_args()
    return args

def is_folder_empty(path: Path) -> bool:
    # 使用scandir，遇到第一个条目就返回False
    with os.scandir(path) as it:
        for entry in it:
            return False  # 有至少一个条目，不为空
    return True  # 没有条目，为空

# 指定读取每一条prompt对应回复的jsonl文件
def read_profile_result(file_path: Path) -> Any:
    try:
        with open(file_path, 'r') as file:
            line = file.readline()
            return json.loads(line)
    except Exception:
        print(traceback.format_exc())
        return None
    
def add_correctness(file_path: Path, predict: str, lable: str):
    # 修改该文档中的"index"字段
    with open(file_path, 'r') as f:
        line = f.readline()
    data = json.loads(line)
    # 添加correctness字段
    data['correctness'] = (predict == lable)
    # 写回文件  
    with open(file_path, 'w') as f:
        f.write(json.dumps(data) + '\n')  # 写回文件，保持jsonl格式

if __name__ == "__main__":
    args = parse_args()
    benchmark = str(args.benchmark)
    workspace = Path(os.path.join(str(args.input_dir), benchmark))
    models:List[str] = [item for item in os.listdir(workspace) if os.path.isdir(os.path.join(workspace, item))]

    #--------------------------------- Load the YAML file -------------------------------
    record_path = Path(os.path.join(workspace, "evaluate_record.yaml"))
    if record_path.exists():
        record_file = record_path
    else:
        raise FileNotFoundError(f"配置文件不存在或不可读取：{record_path}")
    with open(record_file, 'r', encoding='utf-8') as f:
        record = yaml.safe_load(f)         # record: <class 'dict'>
    
    #---------------------------------- Load the dataset ---------------------------------
    dataset = load_dataset(benchmark, record["dataset_dir"])
    
    for model in models:
        model_path = os.path.join(workspace, model)
        if is_folder_empty(model_path):    # 目录为空
            pass
        else:                              # 目录不为空
            start_idx = record["Models"][model]["start_idx"]         # 说明上次evaluate到这一条数据，最后约定一下如何标定下标以防冲突。
            if start_idx >= len(dataset):
                print(f"模型:{model}在基准测试集:{benchmark}上的evaluate已经完成！")
                continue
            print(f"模型:{model}在基准测试集:{benchmark}上开始evaluate，evaluate数据下标从{start_idx+1}开始，到{len(dataset)}结束！")
            accuracy = 0
            total_runtime = 0
            total_tokens = 0
            # 遍历数据集
            for idx, sample in tqdm(enumerate(dataset[start_idx:], start=start_idx+1)):
                lable = parse_answer(benchmark, sample["answer"])
                file_path = Path(os.path.join(model_path, f"train_{idx}.jsonl"))
                profile_result = read_profile_result(file_path)
                profile_idx = profile_result["index"]
                total_runtime += float(profile_result["runtime"])
                total_tokens += int(profile_result["length_of_output_token_ids"])
                if str(profile_idx) == str(idx):
                    predict = parse_answer(benchmark, profile_result["full_response"])
                    add_correctness(file_path, predict, lable)
                    if predict == lable:
                        accuracy += 1
                else:
                    raise ValueError(f"标准答案index:{idx}与profile输出的index:{profile_idx}不匹配")
            record["Models"][model]["start_idx"] = idx  # 更新evaluate到的数据下标
            record["Models"][model]["accuracy"] = f"{accuracy/(len(dataset)-start_idx)*100:.2f}%"  # 写入Accuracy
            record["Models"][model]["total_runtime"] = f"{total_runtime:.2f} seconds"  # 写入Total_runtime
            record["Models"][model]["avg_runtime"] = f"{total_runtime/(len(dataset)-start_idx):.2f} seconds"  # 写入Avg_runtime
            record["Models"][model]["total_tokens"] = f"{total_tokens} tokens" # 写入Total_tokens
            record["Models"][model]["avg_tokens"] = f"{total_tokens/(len(dataset)-start_idx):.2f} tokens" # 写入Avg_tokens
            # 将record写回yaml文件
            with open(record_file, 'w') as f:
                yaml.dump(record, f)
            print(f"模型:{model}在基准测试集:{benchmark}上完成evaluate！")
            print(f"模型:{model}在基准测试集:{benchmark}上的Accuracy={accuracy/(len(dataset)-start_idx)*100:.2f}%")
            print(f"模型:{model}在基准测试集:{benchmark}上的Total_runtime={total_runtime:.2f} seconds")
            print(f"模型:{model}在基准测试集:{benchmark}上的Avg_runtime={total_runtime/(len(dataset)-start_idx):.2f} seconds")
            print(f"模型:{model}在基准测试集:{benchmark}上的Total_tokens={total_tokens:.2f} tokens")
            print(f"模型:{model}在基准测试集:{benchmark}上的Avg_tokens={total_tokens/(len(dataset)-start_idx):.2f} tokens")

