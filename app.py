
import tensorflow as tf
import pickle
import re
import nltk

from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("model/fake_news_model.h5")

# Load tokenizer
with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

def clean_text(text):
    text = str(text).lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    news = request.form["news"]

    news = clean_text(news)

    seq = tokenizer.texts_to_sequences([news])
    padded = pad_sequences(seq, maxlen=500)

    prediction = model.predict(padded)[0][0]

    probability = round(prediction * 100, 2)

    if prediction > 0.5:
        result = "Fake News"
    else:
        result = "Real News"
        probability = round((1-prediction) * 100, 2)

    return render_template(
        "index.html",
        prediction_text=result,
        confidence=probability
    )
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)



