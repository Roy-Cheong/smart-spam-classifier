# Smart Spam Classifier

A small Streamlit app that classifies **SMS** and **Email** text as **Spam** or **Not Spam**.  
Shows confidence, highlights risky keywords, and supports batch CSV uploads.

> Privacy: We don’t store your text. Everything is processed in memory for the current session.

---

## Run locally

git clone https://github.com/<your-username>/smart-spam-classifier.git
cd smart-spam-classifier
pip install -r requirements.txt

cd app
streamlit run app.py
The app will try to load a local model in ./model/transformer_sms or ./model/transformer_email.
If not found, it automatically loads from the Hugging Face Hub:

Roy-Cheong/smart-spam-sms

Roy-Cheong/smart-spam-email


## CSV format (batch tab)
Upload a CSV with one column named message:

csv
Copy code
message
"Congrats! You won a prize, click here"
"Reminder: team meeting at 2pm"
You can download results as a CSV from the app.


## Tech
Streamlit, Transformers (Hugging Face), PyTorch, pandas, matplotlib

## Notes
Threshold slider lets you make the spam detector stricter/looser.
Keyword highlighting is for quick visibility only (not a full explanation).