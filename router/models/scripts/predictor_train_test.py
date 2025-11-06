# This file is responsible for fine-tune Bert-base-uncased and train the downstream MLP classifier
# step1. fine-tune the Bert model with just a few epochs (like 3 epochs).  
# step2. frozen the Bert model, and just train the downstream classifier model(MLP).
# And there are a few problem to figure out:
# Q1: What dataset we gonna use? A paper used LMSYS-Chat-1M dataset.
# Q2: The format of dataset? queries and labels.

# Answer: 
# For Q1: We can try GSM8K first, if the results is not so good we can try LMSYS-Chat-1M.
# For Q2: 
#   In Bert fine-tune stage: 
#       Use the quries in datasets as inputs, and use all models' output length as labels.
#   In Downstream classifier training stage:
#       We can train classifier for every LLM referring to MixLLM
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from router.models.modeling.modeling import Bert_MLP
import torch

# Bert fine-tune stage
def stage1(device:torch.device, dtype:torch.dtype):
    bert_dir="/home/ouyk/project/ICDCS/Oracle/model/Bert_Base"
    device = torch.device("cuda:1")

    # Step 2: Load the pretrained Bert model and tokenizer
    model = Bert_MLP(bert_dir,
                     classifier_dir="",
                     bert_hidden_dim=768,
                     hidden_size=192,
                     output_size=16,
                     device=device,
                     dtype=dtype,
                     fine_tune=False).train()
    
    tokenizer = AutoTokenizer.from_pretrained(
                bert_dir,
            )

    # Step 8: Save the fine-tuned model
    model.bert.save_pretrained("./model/Fine_Tuned")
    tokenizer.save_pretrained("./model/Fine_Tuned")
    
# Downstream classifier training stage
def stage2(LLM:str, device:torch.device, dtype:torch.dtype):
    bert_dir="/home/ouyk/project/ICDCS/Oracle/model/Fine_Tuned"
    device = torch.device("cuda:1")

    # Step 2: Load the pretrained Bert model and tokenizer
    model = Bert_MLP(bert_dir,
                     classifier_dir="",
                     bert_hidden_dim=768,
                     hidden_size=192,
                     output_size=16,
                     device=device,
                     dtype=dtype,
                     fine_tune=False).train()
    
    tokenizer = AutoTokenizer.from_pretrained(
                bert_dir,
            )

    # Step 8: Save the fine-tuned model
    torch.save(model.classifier.state_dict(), f"/home/ouyk/project/ICDCS/Oracle/model/Classifier/{LLM}.bin")

if __name__ == "__main__":
    device = torch.device("cuda:1")
    dtype = torch.float16
    
    # Bert fine-tune stage
    stage1(device=device,
           dtype=dtype)

    # Downstream classifier training stage
    stage2(LLM="Qwen3-0.6B",
           device=device,
           dtype=dtype)