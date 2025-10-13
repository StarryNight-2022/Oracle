from .GSM8K import load_GSM8K, parse_answer_GSM8K
from .MMLU import load_MMLU, parse_answer_MMLU

def load_dataset(dataset:str, path:str):
    if dataset == "GSM8K":
        return load_GSM8K(path)
    if dataset == "MMLU":
        return load_MMLU(path)
    
def parse_answer(dataset:str, String:str):
    if dataset == "GSM8K":
        return parse_answer_GSM8K(String)
    if dataset == "MMLU":
        return parse_answer_MMLU(String)