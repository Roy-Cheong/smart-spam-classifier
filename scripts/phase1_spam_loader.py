import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ✅ Set emoji-compatible font (Windows built-in)
plt.rcParams['font.family'] = 'Segoe UI Emoji'

# Ensure folder exists
os.makedirs("assets", exist_ok=True)

# === LOAD CLEANED DATASET ===
df = pd.read_csv("data/spam_clean.csv")

print("📦 Cleaned Dataset Preview:")
print(df.head())

# === CLASS BALANCE ===
print("\n📊 Class Distribution:")
print(df["label"].value_counts())

# === MESSAGE LENGTH STATS ===
df["msg_length"] = df["message"].apply(len)
print("\n🧮 Message Length Stats:")
print(df["msg_length"].describe())

# === LENGTH DISTRIBUTION CHART ===
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x="msg_length", hue="label", bins=40, kde=True, palette="Set2", alpha=0.7)
plt.title("🔍 Message Length Distribution by Class")
plt.xlabel("Message Length (characters)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("assets/message_length_distribution_cleaned.png")
plt.show()
