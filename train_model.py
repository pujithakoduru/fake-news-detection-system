
import pandas as pd
import re
import nltk
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Download stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):

    text = str(text).lower()

    text = re.sub('[^a-zA-Z]', ' ', text)

    words = text.split()

    words = [w for w in words if w not in stop_words]

    return " ".join(words)


print("Loading dataset...")

data = pd.read_csv("dataset/news_dataset.csv")

data = data.dropna()

data["content"] = data["content"].astype(str)

print("Cleaning text...")

data["content"] = data["content"].apply(clean_text)

X = data["content"]
y = data["label"]

print("Preparing tokenizer...")

tokenizer = Tokenizer(num_words=5000)

tokenizer.fit_on_texts(X)

# Save tokenizer
with open("model/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

X = tokenizer.texts_to_sequences(X)

X = pad_sequences(X, maxlen=500)

print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Building model...")

model = Sequential()

model.add(Embedding(5000, 64))

model.add(LSTM(64))

model.add(Dense(1, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("Training model...")

history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test,y_test))

print("Evaluating model...")

loss, accuracy = model.evaluate(X_test, y_test)

print("\nModel Accuracy:", accuracy)

# Predictions
y_pred = model.predict(X_test)

y_pred = (y_pred > 0.5)

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")

cm = confusion_matrix(y_test, y_pred)

print(cm)

# Confusion matrix graph
plt.figure(figsize=(6,5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.savefig("model/confusion_matrix.png")

plt.show()

# Training accuracy graph
plt.figure()

plt.plot(history.history['accuracy'], label='Training Accuracy')

plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title('Model Accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend()

plt.savefig("model/training_accuracy.png")

plt.show()

# Training loss graph
plt.figure()

plt.plot(history.history['loss'], label='Training Loss')

plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title('Model Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend()

plt.savefig("model/training_loss.png")

plt.show()

# Save model
model.save("model/fake_news_model.h5")

print("\nModel trained and saved successfully!")








