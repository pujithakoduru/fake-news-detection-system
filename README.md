# Fake News Detection System

## Overview

This project is an AI-powered Fake News Detection System built using  Natural Language Processing (NLP) and Deep Learning (LSTM).

The system analyzes news text and predicts whether the news is **Fake or Real**.

A Flask web application allows users to enter news content and receive predictions with a confidence score.

---

## Technologies Used

* Python
* TensorFlow / Keras
* Natural Language Processing
* Flask
* Scikit-Learn
* Matplotlib
* Seaborn

---

## Features

* Deep Learning model using **LSTM**
* Text preprocessing using **NLP**
* Tokenization and sequence modeling
* Confusion matrix visualization
* Training accuracy and loss graphs
* Web interface for real-time prediction
* Probability confidence score

---

## Model Performance

Accuracy: **~98%**

Evaluation metrics include:

* Precision
* Recall
* F1 Score
* Confusion Matrix

---

## Project Structure

```
fake_news_project
│
├── dataset
├── model
├── templates
├── app.py
├── train_model.py
├── prepare_dataset.py
├── requirements.txt
└── README.md
```

---

## How to Run the Project

Clone the repository:

```
git clone https://github.com/pujithakoduru/fake-news-detection-system
```

Install dependencies:

```
pip install -r requirements.txt
```

Train the model:

```
python train_model.py
```

Run the web application:

```
python app.py
```

Open in browser:

```
http://127.0.0.1:5000
```

---

## Future Improvements

* Use Transformer models like **BERT**
* Integrate fact-checking APIs
* Deploy using cloud services

---

