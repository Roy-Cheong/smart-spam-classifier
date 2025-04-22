from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# === Load model and tokenizer once ===
model_dir = "./model/transformer_email"
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def predict_email_spam(message):
    inputs = tokenizer(message, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        print(f"\n🔎 Raw logits: {logits}")
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        print(f"🔍 Softmax probs: {probs}")
        pred_label = int(probs.argmax())
        print(f"🧠 Predicted label: {pred_label}")

    return pred_label, probs


# === CLI input handler ===
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])
        label, probs = predict_email_spam(message)

        if label == 1:
            print(f"🚫 SPAM\nConfidence: Spam: {probs[1]*100:.2f}%, Ham: {probs[0]*100:.2f}%")
        else:
            print(f"✅ HAM\nConfidence: Spam: {probs[1]*100:.2f}%, Ham: {probs[0]*100:.2f}%")
    else:
        print("📝 Please enter a message to classify.")