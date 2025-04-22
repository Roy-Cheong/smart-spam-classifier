from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load from saved model
model_path = "./model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Sample messages
messages = [
    "WINNER! Claim your free prize now!",
    "Hey, can we meet at 7pm?",
    "Free crypto airdrop for all users!",
    "I’ll bring the slides for tomorrow’s class."
]

# Run predictions
print("📩 Predictions:\n")
for msg in messages:
    inputs = tokenizer(msg, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1)
        label = torch.argmax(probs).item()
        emoji = "🚫 SPAM" if label == 1 else "✅ HAM"
        print(f"{emoji} → '{msg}'\n   → Confidence: {probs.squeeze().tolist()}")
