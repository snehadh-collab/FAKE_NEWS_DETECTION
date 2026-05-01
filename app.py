from flask import Flask, render_template, request
import pandas as pd
import re
import nltk
import requests
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Setup
app = Flask(__name__)

nltk.download('stopwords')
nltk.download('wordnet')

# Load data
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 1
true["label"] = 0

data = pd.concat([fake, true])
data = data.sample(frac=1).reset_index(drop=True)

# NLP setup
lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')

def clean_text(text):
    text = str(text).lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# Clean data
data["text"] = data["text"].apply(clean_text)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data["text"])
y = data["label"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction function
def predict_news(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    
    result = model.predict(vector)[0]
    prob = model.predict_proba(vector)[0]

    real_conf = prob[0] * 100
    fake_conf = prob[1] * 100

    label = "REAL NEWS" if result == 0 else "FAKE NEWS"

    return label, real_conf, fake_conf
API_KEY = "c2b4ddc9a7654a118b11f11d94e820fb"   # your key

def check_news_api(query):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={API_KEY}"
    
    response = requests.get(url)
    data = response.json()

    articles = data.get("articles", [])

    if len(articles) == 0:
        return [], 0

    titles = [article["title"] for article in articles[:5]]

    return titles, len(articles)

# Routes
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    real = 0
    fake = 0

    if request.method == "POST":
        news = request.form["news"]
        prediction, real, fake = predict_news(news)

    return render_template("index.html", prediction=prediction, real=real, fake=fake)

# Run app
if __name__ == "__main__":
    app.run(debug=True)