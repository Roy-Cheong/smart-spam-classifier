# 📬 Smart Spam Classifier (SMS + Email)

A modern, AI-powered spam detection web app for SMS and email messages. Built with `Transformers`, `PyTorch`, and `Streamlit`, it delivers fast predictions, dynamic keyword highlighting, tagging, and a smooth batch classification flow.

---

## 🚀 Features

- 📱 SMS & 📧 Email spam classification
- 🤖 Transformer-powered predictions (DistilBERT via Hugging Face Hub)
- 🏷️ Auto-tagging system:
  - 🪙 Phishing — e.g., account, login, verify
  - 📢 Promo — e.g., free, offer, discount
  - 🎁 Suspicious — e.g., won, gift, prize
  - ✅ Legit — if nothing matches
- ✨ Real-time **keyword highlighting** ("attention-like")
- 📤 Batch upload with downloadable CSV results
- 📊 Compact pie chart analytics (SPAM vs. HAM)
- 🎚️ Spam threshold slider for sensitivity tuning
- 💡 Tips + tag legend for non-technical users

---

## 📂 Folder Structure

```
content-moderation-nlp/
├── assets/                # Visuals, saved plots
├── data/                  # (Optional) sample CSVs
├── script/                # Python helper scripts
├── streamlit_app_v2.py    # Main Streamlit app
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 💻 Run Locally

```bash
# Clone the repo
git clone https://github.com/Roy-Cheong/smart-spam-classifier.git
cd smart-spam-classifier

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # or source .venv/bin/activate on Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app_v2.py
```

> ⚠️ This project is tested and compatible with **Python 3.10.x** only.  
> ⚠️ Models are loaded directly from Hugging Face Hub:
> - 📱 SMS: [Roy-Cheong/smart-spam-sms](https://huggingface.co/Roy-Cheong/smart-spam-sms)
> - 📧 Email: [Roy-Cheong/smart-spam-email](https://huggingface.co/Roy-Cheong/smart-spam-email)

---

## 🧪 Try It Out

**Example input:**
```
URGENT: You have won a $1000 Amazon gift card. No purchase necessary.
```

**Expected Output:**
- ✅ Prediction: **SPAM**
- 🏷️ Tag: **🎁 Suspicious**
- ✨ Highlights: `**URGENT**`, `**WON**`, `**GIFT**`

---

## 🗃️ CSV Upload Format

Your batch file should look like:

```csv
message
Congratulations! You've been selected to win!
Please verify your account information.
See you tomorrow at the meeting.
```

Results will appear in a table + downloadable button.

---

## 🛠️ Future Improvements

- 🧠 Attention-score-based keyword overlays
- 📈 Interactive visual analytics (Altair/Plotly)

---

## 👨‍💻 Author

Built by **Roy Cheong**  
April 2025  
Passionate about making AI explainable, useful, and user-friendly.
