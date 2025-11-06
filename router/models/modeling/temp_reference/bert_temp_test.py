import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import BertForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(
    "/home/ouyk/project/ICDCS/Oracle/model/Bert_Base"
)
# model = AutoModelForMaskedLM.from_pretrained(
#     "/home/ouyk/project/ICDCS/Oracle/model/Bert_Base",
#     dtype=torch.float16,
#     device_map="auto",
#     attn_implementation="sdpa",
# )
model = BertForSequenceClassification.from_pretrained(
    "/home/ouyk/project/ICDCS/Oracle/model/Bert_Base",
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa",
)
inputs = tokenizer("Plants create [MASK] through a process known as photosynthesis.", return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

masked_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
predicted_token_id = predictions[0, masked_index].argmax(dim=-1)
predicted_token = tokenizer.decode(predicted_token_id)

print(f"The predicted token is: {predicted_token}")