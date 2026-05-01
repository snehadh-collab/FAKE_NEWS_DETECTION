import streamlit as st
import pickle
import re
import nltk
import pandas as pd
import requests

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------- SETUP ---------------- #

nltk.download('stopwords')
nltk.download('wordnet')

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')

# ---------------- CLEAN TEXT ---------------- #

def clean_text(text):
    text = str(text).lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(words)

# ---------------- PREDICT ---------------- #

def predict_news(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])

    result = model.predict(vector)[0]
    prob = model.predict_proba(vector)[0]

    return result, prob

# ---------------- NEWS API ---------------- #

API_KEY = "c2b4ddc9a7654a118b11f11d94e820fb"   # 🔴 PUT YOUR KEY HERE

def check_news_api(query):
    cleaned = clean_text(query)

    # Try multiple search strategies
    keyword_sets = [
        " ".join(cleaned.split()[:6]),
        " ".join(cleaned.split()[:4]),
        " ".join(cleaned.split()[:2])
    ]

    for keywords in keyword_sets:
        url = f"https://newsapi.org/v2/everything?q={keywords}&language=en&sortBy=relevancy&pageSize=5&apiKey={API_KEY}"

        response = requests.get(url)

        if response.status_code != 200:
            continue

        data = response.json()
        articles = data.get("articles", [])

        if articles:
            results = []
            for article in articles:
                results.append({
                    "title": article["title"],
                    "source": article["source"]["name"],
                    "url": article["url"]
                })
            return results, len(articles)

    return [], 0


# ---------------- UI ---------------- #

st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("🧠 Fake News Detector")
st.write("AI Prediction + Real News Verification")

text = st.text_area("Enter news text:")

if st.button("Analyze"):

    if text.strip() == "":
        st.warning("Please enter some text")

    else:
        # -------- PREDICTION -------- #
        result, prob = predict_news(text)

        real = prob[0] * 100
        fake = prob[1] * 100

        if result == 0:
            st.success("✅ REAL NEWS")
        else:
            st.error("🚨 FAKE NEWS")

        # -------- METRICS -------- #
        col1, col2 = st.columns(2)
        col1.metric("Real %", f"{real:.2f}%")
        col2.metric("Fake %", f"{fake:.2f}%")

        # -------- CHART -------- #
        st.subheader("Confidence Analysis")

        chart_data = pd.DataFrame({
            "Label": ["Real", "Fake"],
            "Score": [real, fake]
        })

        st.bar_chart(chart_data.set_index("Label"))

        # -------- NEWS CHECK -------- #
        st.subheader("📰 Related News")

        results, count = check_news_api(text)

        if count == 0:
            st.warning("⚠️ No related news found (API limitation or rare headline)")
        else:
            for i, article in enumerate(results, 1):
                st.markdown(f"""
                **{i}. {article['title']}**  
                🏢 Source: *{article['source']}*  
                🔗 [Read Article]({article['url']})
                """)