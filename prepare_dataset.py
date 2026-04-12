import pandas as pd
from sklearn.utils import resample

print("Loading datasets...")

fake = pd.read_csv("dataset/Fake.csv")
real = pd.read_csv("dataset/True.csv")

# Labels
fake["label"] = 1   # Fake
real["label"] = 0   # Real

# Combine
data = pd.concat([fake, real], axis=0)

# Handle missing
data["title"] = data["title"].fillna("")
data["text"] = data["text"].fillna("")

# Merge text
data["content"] = data["title"] + " " + data["text"]

data = data[["content", "label"]]

# Remove duplicates
data = data.drop_duplicates()

# 🔥 BALANCE DATA
real = data[data.label == 0]
fake = data[data.label == 1]

if len(real) > len(fake):
    real = resample(real, replace=False, n_samples=len(fake), random_state=42)
else:
    fake = resample(fake, replace=False, n_samples=len(real), random_state=42)

data = pd.concat([real, fake])
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
data.to_csv("dataset/news_dataset.csv", index=False)

print("✅ Dataset ready!")
print("Total:", len(data))
print("Fake:", sum(data["label"] == 1))
print("Real:", sum(data["label"] == 0))