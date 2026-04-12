#  AI Fake News Detection System

An intelligent web-based application that detects whether a news article is **Real or Fake** using Deep Learning and Natural Language Processing (NLP).

---

## 📌 Features

* 📝 Detect fake news from **text input**
* 🖼️ Detect fake news from **images (OCR)**
* 📄 Analyze **PDF documents**
* 📃 Analyze **Word documents (.doc/.docx)**
* 📊 Displays **prediction confidence**
* 🌐 Interactive **Flask web interface**

---

## 🧠 Technologies Used

* Python
* TensorFlow / Keras
* Natural Language Processing (NLTK)
* Flask (Web Framework)
* Scikit-learn
* Matplotlib & Seaborn
* Tesseract OCR

---

## 📂 Project Structure

fake_news_project/

├── app.py
├── train_model.py
├── prepare_dataset.py
├── requirements.txt

├── model/
│   ├── fake_news_model.h5
│   ├── tokenizer.pkl

├── templates/
│   └── index.html

├── dataset/
│   └── README.md

├── uploads/

└── .gitignore

---

## ⚙️ Installation

### 1. Clone the repository

git clone https://github.com/pujithakoduru/fake-news-detection-system.git
cd fake-news-detection-system

---

### 2. Create virtual environment

python -m venv venv
venv\Scripts\activate

---

### 3. Install dependencies

pip install -r requirements.txt

---

### 4. Install Tesseract OCR (for image processing)

Download from:
https://github.com/tesseract-ocr/tesseract

(Optional) Add path in code:

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

---

## ▶️ Run the Application

python app.py

Open in browser:
http://127.0.0.1:5000

---

## 🧪 Model Training

python prepare_dataset.py
python train_model.py

---

## 📊 Model Performance

* Accuracy: ~98%
* Uses Bidirectional LSTM
* Evaluated using confusion matrix and classification report

---

## ⚠️ Limitations

* Works best with long news content
* May misclassify short inputs
* Dataset mainly contains English news

---

## 🚀 Future Improvements

* Use BERT model for higher accuracy
* Real-time news verification
* Highlight fake parts in text
* Mobile-friendly UI

---


## 👩‍💻 Author

Pujitha Koduru

---

## ⭐ Support

If you like this project, give it a star ⭐ on GitHub!
tesseract --version