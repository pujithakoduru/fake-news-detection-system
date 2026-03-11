import pandas as pd

fake = pd.read_csv("dataset/Fake.csv")
real = pd.read_csv("dataset/True.csv")

fake["label"] = 1
real["label"] = 0

data = pd.concat([fake, real])

data["content"] = data["title"] + " " + data["text"]

data = data[["content","label"]]

data.to_csv("dataset/news_dataset.csv", index=False)

print("Dataset ready")