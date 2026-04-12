import os
import tensorflow as tf
import pickle
import re
import nltk

from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

# File processing
import pytesseract
from PIL import Image
import PyPDF2
import docx

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

nltk.download('stopwords')

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("model/fake_news_model.h5")

with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# 🔥 SAME CLEANING AS TRAINING
def clean_text(text):
    text = str(text).lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    return text

# -------- FILE EXTRACTION -------- #
def extract_text_from_image(file):
    try:
        image = Image.open(file)
        return pytesseract.image_to_string(image)
    except:
        return ""

def extract_text_from_pdf(file):
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
        return text
    except:
        return ""

def extract_text_from_doc(file):
    try:
        doc_file = docx.Document(file)
        return "\n".join([para.text for para in doc_file.paragraphs])
    except:
        return ""

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    extracted_text = ""

    # TEXT INPUT
    text_input = request.form.get("news")

    if text_input and text_input.strip() != "":
        extracted_text = text_input

    else:
        file = request.files.get("file")

        if file and file.filename != "":
            filename = file.filename.lower()

            if filename.endswith((".png", ".jpg", ".jpeg")):
                extracted_text = extract_text_from_image(file)

            elif filename.endswith(".pdf"):
                extracted_text = extract_text_from_pdf(file)

            elif filename.endswith((".doc", ".docx")):
                extracted_text = extract_text_from_doc(file)

            else:
                return render_template("index.html", error="Unsupported file type")

    if extracted_text.strip() == "":
        return render_template("index.html", error="No valid input provided")

    cleaned = clean_text(extracted_text)

    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=500)

    prediction = model.predict(padded)[0][0]

    print("Prediction value:", prediction)

    if prediction > 0.5:
        result = "Fake News"
        confidence = round(prediction * 100, 2)
    else:
        result = "Real News"
        confidence = round((1 - prediction) * 100, 2)

    return render_template(
        "index.html",
        prediction_text=result,
        confidence=confidence,
        extracted_text=extracted_text[:500]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)