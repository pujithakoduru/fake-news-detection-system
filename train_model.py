import pandas as pd
import re
import nltk
import pickle

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 🔥 CLEAN TEXT (IMPORTANT: SAME AS APP)
def clean_text(text):
    text = str(text).lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    return text

print("Loading dataset...")
data = pd.read_csv("dataset/news_dataset.csv")

data = data.dropna()
data["content"] = data["content"].astype(str)

print("Cleaning text...")
data["content"] = data["content"].apply(clean_text)

X = data["content"]
y = data["label"]

print("Tokenizing...")

vocab_size = 10000
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X)

with open("model/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

X = tokenizer.texts_to_sequences(X)
X = pad_sequences(X, maxlen=500)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Building model...")

model = Sequential()
model.add(Embedding(vocab_size, 128, input_length=500))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("Training...")
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

print("Evaluating...")
loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy:", accuracy)

y_pred = (model.predict(X_test) > 0.5)
print(classification_report(y_test, y_pred))

# Save model
model.save("model/fake_news_model.h5")

print("✅ Model saved successfully!")