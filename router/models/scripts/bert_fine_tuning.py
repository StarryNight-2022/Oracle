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
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
from datasets import load_dataset
import yaml
import numpy as np

# TODO: Fix some problems in training. Maybe the model forward process or loss calculation process has some problem.

# 自定义内容
from router.models.scripts.dataset.our_datasets import prepare_training_data, train_test_split

# Bert fine-tune stage
def stage1(device:torch.device, dtype:torch.dtype, epochs:int, batch_size:int):
    bert_dir="/home/ouyk/project/ICDCS/Oracle/model/Bert_Base"
    
    config_file = "/home/ouyk/project/ICDCS/Oracle/config/router_model_GSM8K.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
        
    n_classes = config["Data"]["labels"]["num_tokens_range_split"]
    # n_classes = config["Data"]["labels"]["latency_range_split"]
    save_dir = config["Bert_Fine_Tuning"]["data_save_dir"]
    
    # TODO: Change to our dataset. 
    # Step 1: Load and prepare the dataset
    dataset = load_dataset(
        '/home/ouyk/project/ICDCS/Oracle/router/models/scripts/dataset/fine_tuning_dataset.py',
        data_files={
            'train': os.path.join(save_dir, 'train_data.npy'),
            'test': os.path.join(save_dir, 'test_data.npy')
        },
        trust_remote_code=True
    )

    # 使用数据集
    train_dataset = dataset['train']
    eval_dataset = dataset['test']

    # Step 2: Load the pretrained Bert model and tokenizer
    model = BertForSequenceClassification.from_pretrained(
            bert_dir,
            dtype=dtype,
            attn_implementation="sdpa",
            num_labels = n_classes
            ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(
                bert_dir,
            )

    # Step 3: Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["prompt"], padding="max_length", truncation=True, max_length=128)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    # Step 4: Set up the data collator
    # Use a padding collator for sequence classification so we don't overwrite
    # the dataset's scalar labels with token-level language-model labels.
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Step 5: Define training arguments
    training_args = TrainingArguments(
        output_dir="./model/Fine_Tuned", # Directory to save the model
        overwrite_output_dir=True,
        num_train_epochs=epochs,         # Number of training epochs
        per_device_train_batch_size=batch_size,   # Batch size per device (use function arg)
        save_steps=500,                  # Save checkpoint every 500 steps
        save_total_limit=2,              # Keep only the last 2 checkpoints
        logging_dir="./logs",            # Directory for logs
        logging_steps=100,               # Log every 100 steps
        # evaluation_strategy="steps",     # Evaluate every `eval_steps`
        eval_steps=500,                  # Evaluation frequency
        learning_rate=5e-5,              # Learning rate
        weight_decay=0.01,               # Weight decay
        fp16=True,                       # Use mixed precision (if GPU supports it)
    )

    # Step 6: Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Step 7: Fine-tune the model
    trainer.train()

    # Step 8: Save the fine-tuned model
    model.bert.save_pretrained("./model/Fine_Tuned")
    tokenizer.save_pretrained("./model/Fine_Tuned")

    # step 9: Evaluate the model, calculate accuracy
    acc = 0
    for data in eval_dataset:
        inputs = tokenizer(data["prompt"], return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        if predictions.item() == data["labels"]:
            acc += 1
    accuracy = acc / len(eval_dataset)
    print(f"Evaluation Accuracy: {accuracy * 100:.2f}%")
        
if __name__ == "__main__":
    device = torch.device("cuda:0")
    dtype = torch.float32
    
    # Bert fine-tune stage
    stage1(device=device,
           dtype=dtype,
           epochs=10,
           batch_size=8)

    # # Downstream classifier training stage
    # stage2(LLM="Qwen3-0.6B",
    #        device=device,
    #        dtype=dtype,
    #        epochs=3,
    #        batch_size=32)