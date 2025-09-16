# app/utils.py
import torch
import torch.nn.functional as F
import re
from typing import List, Tuple

def move_to_device_once(model, device):
    if next(model.parameters()).device != device:
        model.to(device)
    model.eval()
    return model

@torch.inference_mode()
def classify(text: str, model, tokenizer, device) -> Tuple[int, list]:
    text = (text or "").strip()
    if not text:
        return 0, [0.0, 0.0]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    logits = model(**inputs).logits
    probs = F.softmax(logits, dim=1)[0].detach().cpu().tolist()
    return int(probs.index(max(probs))), probs

@torch.inference_mode()
def classify_batch(texts: List[str], model, tokenizer, device, batch_size: int = 32):
    results = []
    # normalize inputs
    clean_texts = [(t or "").strip() for t in texts]
    for i in range(0, len(clean_texts), batch_size):
        chunk = clean_texts[i:i+batch_size]
        if not any(chunk):  # all empty
            results.extend([(0, [0.0, 0.0]) for _ in chunk])
            continue
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True).to(device)
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1).detach().cpu().tolist()
        for p in probs:
            label = int(p.index(max(p)))
            results.append((label, p))
    return results

def get_tag(text: str) -> str:
    lowered = (text or "").lower()
    if any(w in lowered for w in ["account", "verify", "login", "suspended", "confirm"]):
        return "🪙 Phishing"
    if any(w in lowered for w in ["buy", "offer", "discount", "free", "deal", "limited", "save"]):
        return "📢 Promo"
    if any(w in lowered for w in ["won", "gift", "selected", "prize", "claim", "urgent"]):
        return "🎁 Suspicious"
    return "✅ Legit"

def highlight_keywords(text: str) -> str:
    if not text: return "_(empty message)_"
    keywords = ["free", "account", "gift", "won", "claim", "prize", "urgent", "verify", "offer"]
    for w in keywords:
        text = re.sub(fr"(?i)\b{re.escape(w)}\b", f"**{w.upper()}**", text)
    return text
