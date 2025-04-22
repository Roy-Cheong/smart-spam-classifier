from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

# ✅ Create SMS transformer output folder
save_dir = "./model/transformer_sms"
os.makedirs(save_dir, exist_ok=True)

# ✅ Load the trained model from correct SMS checkpoint
model = AutoModelForSequenceClassification.from_pretrained("./results/checkpoint-2230")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# ✅ Save to transformer_sms folder
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print("✅ SMS Transformer model and tokenizer saved to ./model/transformer_sms")
