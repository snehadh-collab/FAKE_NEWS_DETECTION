import streamlit as st
import pandas as pd
import nltk
import pickle

# If needed
nltk.download('stopwords')
nltk.download('wordnet')

# Load your model + vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Your cleaning function
def clean_text(text):
    return text.lower()

def predict_news(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    result = model.predict(vector)

    if result[0] == 0:
        return "REAL NEWS"
    else:
        return "FAKE NEWS"

# UI
st.title("📰 Fake News Detector")

user_input = st.text_area("Enter news text:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        prediction = predict_news(user_input)
        st.success(f"Prediction: {prediction}")