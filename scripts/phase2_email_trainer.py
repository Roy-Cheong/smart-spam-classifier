# phase2_email_trainer.py

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import torch
import transformers

print(f"✅ PyTorch version: {torch.__version__}")
print(f"✅ Transformers version: {transformers.__version__}")

# === Load Cleaned Dataset ===
df = pd.read_csv("data/emails_combined_clean.csv")

# Check distribution
print("\n✅ Label distribution:")
print(df["label"].value_counts())

# Train-test split
train_df, test_df = train_test_split(
    df[["text", "label"]],
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

print(f"🧠 Train set: {len(train_df)} rows")
print(f"🧪 Test set: {len(test_df)} rows")

# Convert to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

# Tokenizer and tokenization
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Load model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Training arguments
training_args = TrainingArguments(
    output_dir="./model/transformer_email",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train model
trainer.train()

# Save model and tokenizer
trainer.save_model("./model/transformer_email")
tokenizer.save_pretrained("./model/transformer_email")
print("\n✅ Model saved to model/transformer_email")

# Evaluate
results = trainer.evaluate()
print("\n📊 Final Evaluation:")
print(results)

# Classification Report
preds_output = trainer.predict(test_dataset)
preds = np.argmax(preds_output.predictions, axis=1)
true = preds_output.label_ids

print("\n🧾 Detailed Classification Report:")
print(classification_report(true, preds, target_names=["ham", "spam"]))
