# streamlit_app.py - Reimagined Smart Spam Classifier (v2)

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import re


# ======== CONFIG ========
st.set_page_config(page_title="Smart Spam Classifier", page_icon="📬", layout="wide")

# ======== LOAD MODEL ========
@st.cache_resource
def load_model(path):
    model = AutoModelForSequenceClassification.from_pretrained(path).eval()
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer

sms_model, sms_tokenizer = load_model("Roy-Cheong/smart-spam-sms")
email_model, email_tokenizer = load_model("Roy-Cheong/smart-spam-email")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======== UTILS ========
def classify(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    model.to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    return int(probs.argmax()), probs

def get_tag(text):
    lowered = text.lower()
    if any(w in lowered for w in ["account", "verify", "login", "suspended", "confirm"]):
        return "🪙 Phishing"
    elif any(w in lowered for w in ["buy", "offer", "discount", "free", "deal", "limited", "save"]):
        return "📢 Promo"
    elif any(w in lowered for w in ["won", "gift", "selected", "prize", "claim", "urgent"]):
        return "🎁 Suspicious"
    return "✅ Legit"

def highlight_keywords(text):
    if not text:
        return "_(empty message)_"
    
    keywords = ["free", "account", "gift", "won", "claim", "prize", "urgent"]
    for word in keywords:
        text = re.sub(f"(?i)\\b{word}\\b", f"**{word.upper()}**", text)
    return text


# ======== HEADER ========
st.title("📬 Smart Spam Classifier")
st.caption("AI-powered spam detection for SMS & Email with explainability")
thresh = st.sidebar.slider("Spam Threshold", 0.0, 1.0, 0.5, step=0.01)
st.sidebar.markdown("""
### 💡 How to Use
- Use SMS, Email, or Batch tabs
- Adjust threshold for stricter spam detection
- Paste or upload messages for classification
- Review predictions, confidence & message tags
""")

# ======== TABS ========
tab1, tab2, tab3 = st.tabs(["📱 SMS", "📧 Email", "📤 Batch Upload"])

with tab1:
    st.subheader("SMS Message")
    sms_text = st.text_area("Paste SMS here:", height=100)
    if st.button("🚀 Classify SMS"):
        label, probs = classify(sms_text, sms_model, sms_tokenizer)
        tag = get_tag(sms_text)
        st.markdown(f"### {'🚫 SPAM' if label else '✅ HAM'}")
        st.progress(float(probs[label]))
        st.write(f"**Confidence:** Spam {probs[1]*100:.2f}% | Ham {probs[0]*100:.2f}%")
        st.write(f"**Tag:** {tag}")
        st.markdown("**Highlight:**")
        st.markdown(highlight_keywords(sms_text))



with tab2:
    st.subheader("Email Content")
    email_text = st.text_area("Paste email body:", height=200)
    if st.button("🚀 Classify Email"):
        label, probs = classify(email_text, email_model, email_tokenizer)
        tag = get_tag(email_text)
        st.markdown(f"### {'🚫 SPAM' if label else '✅ HAM'}")
        st.progress(float(probs[label]))
        st.write(f"**Confidence:** Spam {probs[1]*100:.2f}% | Ham {probs[0]*100:.2f}%")
        st.write(f"**Tag:** {tag}")
        st.markdown("**Highlight:**")
        st.markdown(highlight_keywords(email_text))



with tab3:
    st.subheader("Batch Classification")
    model_option = st.radio("Choose Model", ["SMS", "Email"])
    file = st.file_uploader("Upload CSV with 'message' column")
    if file:
        df = pd.read_csv(file)
        model, tokenizer = (sms_model, sms_tokenizer) if model_option == "SMS" else (email_model, email_tokenizer)
        results = []
        for msg in df['message']:
            label, probs = classify(msg, model, tokenizer)
            results.append({"message": msg, "prediction": "SPAM" if label else "HAM", "confidence": f"{probs[label]*100:.2f}%", "tag": get_tag(msg)})
        result_df = pd.DataFrame(results)
        st.dataframe(result_df)
        st.download_button("Download Results", result_df.to_csv(index=False), "results.csv")

# ======== ANALYTICS ========
with st.expander("📊 Analytics"):
    if "result_df" in locals():
        count_data = result_df["prediction"].value_counts()
        fig, ax = plt.subplots(figsize=(1.8, 1.8), dpi=100)
        count_data.plot.pie(
            autopct='%1.1f%%',
            ax=ax,
            startangle=90,
            colors=['#FF6B6B', '#6BCB77'],
            textprops={'fontsize': 7}
        )
        ax.set_ylabel("")
        ax.set_aspect("equal")
        plt.tight_layout(pad=0.2)
        st.pyplot(fig, use_container_width=False)


    st.caption("See how many of your batch messages were classified as spam or ham.")
    st.markdown("**Tag Legend:** 🪙 Phishing | 📢 Promo | 🎁 Suspicious | ✅ Legit")

# ======== FOOTER ========
st.markdown("---")
st.markdown("<div style='text-align:center; color:gray'>Built by Roy Cheong· April 2025</div>", unsafe_allow_html=True)
