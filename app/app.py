
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import torch
torch.set_num_threads(1)
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from layout import inject_css, header, footer, result_card
from utils import classify, classify_batch, get_tag, highlight_keywords, move_to_device_once


# ==== Config ====
st.set_page_config(page_title="Smart Spam Classifier", layout="centered", page_icon="📨")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inject_css()
header()

# Absolute paths
BASE_DIR = Path(__file__).resolve().parent        # .../content moderation nlp/app
ROOT_DIR = BASE_DIR.parent                        # .../content moderation nlp
LOCAL_MODEL_DIR = ROOT_DIR / "model"

# Hub fallbacks
REPOS = {
    "SMS":   "Roy-Cheong/smart-spam-sms",
    "Email": "Roy-Cheong/smart-spam-email",
}

st.sidebar.title("📊 Settings")
msg_type = st.sidebar.selectbox("Message Type", ["SMS", "Email"])
threshold = st.sidebar.slider("Spam Threshold", 0.0, 1.0, 0.5, step=0.01)

local_path = LOCAL_MODEL_DIR / f"transformer_{msg_type.lower()}"

@st.cache_resource(show_spinner="Loading model…")
def load_model(local_path: Path, repo_id: str):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    use_local = local_path.exists() and (local_path / "config.json").exists()
    if use_local:
        tok = AutoTokenizer.from_pretrained(str(local_path), local_files_only=True)
        mdl = AutoModelForSequenceClassification.from_pretrained(str(local_path), local_files_only=True)
        source = f"Local ({local_path})"
    else:
        tok = AutoTokenizer.from_pretrained(repo_id)
        mdl = AutoModelForSequenceClassification.from_pretrained(repo_id)
        source = f"Hugging Face Hub ({repo_id})"
    return mdl, tok, source

repo_id = {"SMS": "Roy-Cheong/smart-spam-sms", "Email": "Roy-Cheong/smart-spam-email"}[msg_type]
model, tokenizer, model_source = load_model(local_path, repo_id)
model = move_to_device_once(model, device)
st.sidebar.caption(f"Model source: **{model_source}**")


# ==== Tabs ====
tab1, tab2, tab3 = st.tabs(["📥 Check Message", "📂 Batch Upload", "📈 Analytics"])

# ------------------- Tab 1: Single message -------------------
with tab1:
    st.header("📨 Check if your message is spam")
    msg = st.text_area("Paste your message here:", height=180)
    check_clicked = st.button("🚦 Check", use_container_width=True)
    if check_clicked:
        with st.spinner("Analyzing…"):
            label, probs = classify(msg, model, tokenizer, device)
            prob_spam, prob_ham = float(probs[1]), float(probs[0])

            # Tri-state verdict: Spam / Review / Not Spam
            tag = get_tag(msg)
            is_spam = (label == 1) and (prob_spam >= threshold)

            needs_review = (not is_spam) and (tag != "✅ Legit") and (prob_spam >= 0.20)  # tweak 0.20 if you like

            if is_spam:
                verdict, icon, chip_cls = "Spam", "🛑", "danger-chip"
            elif needs_review:
                verdict, icon, chip_cls = "Review", "⚠️", "danger-chip"
            else:
                verdict, icon, chip_cls = "Not Spam", "✅", ""

            st.markdown(f"""
            <div style="border:1px solid #22262e; border-radius:16px; padding:16px 18px; background:rgba(255,255,255,.02)">
            <h3 style="margin:.25rem 0">{icon} {verdict}
                <span style="display:inline-block; margin-left:8px; padding:4px 10px; border-radius:999px; background:rgba(255,107,107,.12) if '{chip_cls}' else 'rgba(107,203,119,.12)'; border:1px solid rgba(255,107,107,.35) if '{chip_cls}' else 'rgba(107,203,119,.35)'; font-size:.9rem">{tag}</span>
            </h3>
            <p style="margin:.25rem 0; opacity:.85">Confidence: Spam {prob_spam:.2%} • Not Spam {prob_ham:.2%}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Highlighted Message:**")
            st.markdown(highlight_keywords(msg))
    else:
        st.warning("Please enter a message.")

# ------------------- Tab 2: Batch upload -------------------
with tab2:
    st.header("📂 Upload CSV")
    st.caption("Upload a CSV with a **'message'** column.")
    file = st.file_uploader("Upload your file", type=["csv"])

    if file:
        df = pd.read_csv(file)
        if "message" not in df.columns:
            st.error("CSV must contain a 'message' column.")
        else:
            # Clean inputs (handle NaN, strip)
            texts = [str(x).strip() if pd.notna(x) else "" for x in df["message"].tolist()]

            rows = []
            for t in texts:
                label, probs = classify(t, model, tokenizer, device)
                prob_spam, prob_ham = float(probs[1]), float(probs[0])
                is_spam = (label == 1) and (prob_spam >= threshold)
                rows.append({
                    "prediction": "Spam" if is_spam else "Not Spam",
                    "spam_prob": f"{prob_spam:.4f}",
                    "not_spam_prob": f"{prob_ham:.4f}",
                    "tag": get_tag(t),
                    "message": t
                })

            result_df = pd.DataFrame(rows)
            st.dataframe(result_df[["prediction","spam_prob","not_spam_prob","tag","message"]], use_container_width=True)
            st.download_button(
                "📥 Download Results",
                result_df.to_csv(index=False),
                "classified_results.csv",
                mime="text/csv",
            )
            # Save for analytics tab
            st.session_state["last_batch_df"] = result_df

# ------------------- Tab 3: Analytics -------------------
with tab3:
    st.header("📈 Spam Distribution")
    data = st.session_state.get("last_batch_df")
    if data is not None and not data.empty:
        counts = data["prediction"].value_counts()
        # Ensure fixed order for colors
        counts = counts.reindex(["Spam", "Not Spam"]).fillna(0)

        fig, ax = plt.subplots(figsize=(2.2, 2.2), dpi=110)
        counts.plot.pie(
            autopct='%1.1f%%',
            ax=ax,
            startangle=90,
            colors=['#FF6B6B', '#6BCB77'],
            textprops={'fontsize': 8}
        )
        ax.set_ylabel("")
        ax.set_aspect("equal")
        plt.tight_layout(pad=0.2)
        st.pyplot(fig, use_container_width=False)
    else:
        st.info("Run a batch classification to see analytics.")

# ==== Footer ====
footer()
