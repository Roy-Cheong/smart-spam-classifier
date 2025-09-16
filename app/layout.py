# app/layout.py
import streamlit as st

def inject_css():
    st.markdown("""
    <style>
      .app-container {max-width: 900px; margin: 0 auto;}
      .result-card {
        border: 1px solid var(--secondaryBackgroundColor);
        border-radius: 16px; padding: 16px 18px;
        background: rgba(255,255,255,0.02);
      }
      .result-title { margin: 0; font-size: 1.25rem; }
      .muted { color: #8a8f98; font-size: 0.9rem; }
      .tag-chip {
        display: inline-block; padding: 4px 10px; border-radius: 999px;
        background: rgba(107,203,119,0.12); border: 1px solid rgba(107,203,119,0.35);
        margin-left: 8px; font-size: 0.85rem;
      }
      .danger-chip {
        background: rgba(255,107,107,0.12); border-color: rgba(255,107,107,0.35);
      }
    </style>
    """, unsafe_allow_html=True)

def header():
    st.markdown("<div class='app-container'>", unsafe_allow_html=True)
    st.title("📬 Smart Spam Classifier")
    st.caption("AI-powered spam detection for SMS & Email — clear verdicts, confidence, and highlights.")

def footer():
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; opacity:.8'>• Built by Roy Cheong · September 2025</div>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

def result_card(is_spam: bool, prob_spam: float, prob_ham: float, tag: str):
    icon = "🛑" if is_spam else "✅"
    title = "Spam" if is_spam else "Not Spam"
    chip_cls = "danger-chip" if is_spam else ""
    st.markdown(
        f"""
        <div class='result-card'>
          <h3 class='result-title'>{icon} {title}
            <span class="tag-chip {chip_cls}">{tag}</span>
          </h3>
          <p class='muted'>Confidence: Spam {prob_spam:.2%} • Not Spam {prob_ham:.2%}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
