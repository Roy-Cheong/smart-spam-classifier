from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import transformers
import sys
print(f"Transformers version: {transformers.__version__}")
print(f"Python path: {sys.executable}")

df = pd.read_csv("data/spam_full.tsv", sep="\t", header=None, names=["label", "message"])

# Shuffle and drop nulls
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df = df.dropna()

# Convert labels to numeric
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split into train/test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Load pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenization function
def tokenize(batch):
    return tokenizer(batch["message"], padding=True, truncation=True)

# Apply tokenizer to both sets
train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# Set format for PyTorch
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Load pre-trained model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Training configuration
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    load_best_model_at_end=True,
    logging_dir="./logs",
    logging_steps=5,
    save_total_limit=1,
    report_to="none"
)

# Fix: dynamic padding per batch
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator 
)

# 🚀 Train the model!
trainer.train()

# 🧠 Evaluate the model
eval_results = trainer.evaluate()
print("\n📊 Final Evaluation:")
print(eval_results)
