# phase1_email_loader.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths to all datasets
DATA_PATHS = {
    "phishing_email": "data/phishing_email.csv",
    "SpamAssasin": "data/SpamAssasin.csv",
    "CEAS_08": "data/CEAS_08.csv",
    "Enron": "data/Enron.csv",
    "Ling": "data/Ling.csv",
    "Nazario": "data/Nazario.csv",
    "Nigerian_Fraud": "data/Nigerian_Fraud.csv"
}

def load_and_format():
    all_dfs = []

    # 1. Phishing Email
    df_phishing = pd.read_csv(DATA_PATHS["phishing_email"])
    df_phishing = df_phishing[["text_combined", "label"]].rename(columns={"text_combined": "text"})
    all_dfs.append(df_phishing)

    # 2. SpamAssasin
    df_spamass = pd.read_csv(DATA_PATHS["SpamAssasin"])
    df_spamass["subject"] = df_spamass["subject"].fillna("")
    df_spamass["body"] = df_spamass["body"].fillna("")
    df_spamass["text"] = df_spamass["subject"] + " " + df_spamass["body"]
    df_spamass = df_spamass[["text", "label"]]
    all_dfs.append(df_spamass)

    # 3. CEAS_08
    df_ceas = pd.read_csv(DATA_PATHS["CEAS_08"])
    df_ceas["subject"] = df_ceas["subject"].fillna("")
    df_ceas["body"] = df_ceas["body"].fillna("")
    df_ceas["text"] = df_ceas["subject"] + " " + df_ceas["body"]
    df_ceas = df_ceas[["text", "label"]]
    all_dfs.append(df_ceas)

    # 4. Enron
    df_enron = pd.read_csv(DATA_PATHS["Enron"])
    df_enron["subject"] = df_enron["subject"].fillna("")
    df_enron["body"] = df_enron["body"].fillna("")
    df_enron["text"] = df_enron["subject"] + " " + df_enron["body"]
    df_enron = df_enron[["text", "label"]]
    all_dfs.append(df_enron)

    # 5. Ling
    df_ling = pd.read_csv(DATA_PATHS["Ling"])
    df_ling["subject"] = df_ling["subject"].fillna("")
    df_ling["body"] = df_ling["body"].fillna("")
    df_ling["text"] = df_ling["subject"] + " " + df_ling["body"]
    df_ling = df_ling[["text", "label"]]
    all_dfs.append(df_ling)

    # 6. Nazario (pure spam)
    df_nazario = pd.read_csv(DATA_PATHS["Nazario"])
    df_nazario["subject"] = df_nazario["subject"].fillna("")
    df_nazario["body"] = df_nazario["body"].fillna("")
    df_nazario["text"] = df_nazario["subject"] + " " + df_nazario["body"]
    df_nazario = df_nazario[["text", "label"]]
    all_dfs.append(df_nazario)

    # 7. Nigerian Fraud (pure spam)
    df_nigerian = pd.read_csv(DATA_PATHS["Nigerian_Fraud"])
    df_nigerian["subject"] = df_nigerian["subject"].fillna("")
    df_nigerian["body"] = df_nigerian["body"].fillna("")
    df_nigerian["text"] = df_nigerian["subject"] + " " + df_nigerian["body"]
    df_nigerian = df_nigerian[["text", "label"]]
    all_dfs.append(df_nigerian)

    return pd.concat(all_dfs, ignore_index=True)

# Load and clean
df = load_and_format()

# Drop empty or null text entries
df.dropna(subset=["text", "label"], inplace=True)
df = df[df["text"].str.strip() != ""]

# Drop duplicates
df.drop_duplicates(subset="text", inplace=True)

# Add message length
df["msg_length"] = df["text"].apply(len)

# Show preview
print("📦 Combined Email Dataset Preview:")
print(df.head())

print("\n📊 Message Length Stats:")
print(df["msg_length"].describe())

print("\n🔢 Spam Count:")
print(df["label"].value_counts())

# Save the clean dataset
os.makedirs("data", exist_ok=True)
df.to_csv("data/emails_combined_clean.csv", index=False)
print("\n✅ Cleaned dataset saved to data/emails_combined_clean.csv")

# Plot message length distribution
plt.figure(figsize=(10, 6))
sns.histplot(df["msg_length"], bins=50, kde=True, color="skyblue")
plt.title("📧 Combined Email Message Length Distribution")
plt.xlabel("Message Length (characters)")
plt.ylabel("Frequency")
plt.tight_layout()

# Save plot
os.makedirs("assets", exist_ok=True)
plt.savefig("assets/email_message_length_distribution.png")
print("📊 Plot saved to assets/email_message_length_distribution.png")
