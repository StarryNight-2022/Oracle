from .GSM8K import load_GSM8K, parse_answer_GSM8K

def load_dataset(dataset:str, path:str):
    if dataset == "GSM8K":
        return load_GSM8K(path)
    
def parse_answer(dataset:str, String:str):
    if dataset == "GSM8K":
        return parse_answer_GSM8K(String)