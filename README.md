# 🧠 Fake News Detection System

An AI-powered web application that detects whether a news article is **REAL or FAKE** using Natural Language Processing and Machine Learning, with real-world validation using live news APIs.

---

## 🚀 Features

* 🔍 Fake vs Real News Classification (Naive Bayes)
* 📊 Confidence Score Visualization (Bar Chart)
* 📰 Live News Verification using News API
* 🔗 Displays Related Articles with Source & Links
* 🧹 Text Cleaning with NLP (NLTK)
* 🌐 Interactive UI using Streamlit

---

## 🛠️ Tech Stack

* Python
* Scikit-learn
* NLTK
* Pandas
* Streamlit
* News API

---

## 📂 Project Structure

```
fake_news_detection/
│── streamlit_app.py
│── train_model.py
│── app.py
│── templates/
│   └── index.html
│── .gitignore
│── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```
git clone https://github.com/your-username/your-repo-name.git
cd fake_news_detection
```

---

### 2. Install dependencies

```
pip install -r requirements.txt
```

---

### 3. Set up API Key

Get your API key from NewsAPI and set it as an environment variable:

**Windows (PowerShell):**

```
setx NEWS_API_KEY "your_api_key_here"
```

**Linux / Mac:**

```
export NEWS_API_KEY="your_api_key_here"
```

---

### 4. Train the model

```
python train_model.py
```

This will generate:

* `model.pkl`
* `vectorizer.pkl`

---

### 5. Run the application

```
streamlit run streamlit_app.py
```

---

## 🧠 How it Works

1. Input news text
2. Text is cleaned using NLP techniques
3. Converted into numerical features (TF-IDF)
4. Classified using Naive Bayes model
5. Confidence scores are displayed
6. Related real-world articles are fetched using News API

---

## ⚠️ Notes

* News API results are approximate and may not always return exact matches
* Model accuracy depends on training data quality
* `.pkl` files are excluded from Git and must be generated locally

---

## 📈 Future Improvements

* Similarity scoring with real articles
* Deep learning-based classification
* UI enhancements (news cards, themes)
* Deployment on cloud platforms

---

## 👤 Author

Sneha D H

---

## ⭐ If you found this useful, give it a star!
