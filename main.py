import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required data (only first time)
nltk.download('stopwords')
nltk.download('wordnet')

# Load datasets
print("Loading datasets...")
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake["label"] = 1
true["label"] = 0

# Combine datasets
data = pd.concat([fake, true])

# Shuffle data
data = data.sample(frac=1).reset_index(drop=True)

print("Dataset ready. Total rows:", data.shape[0])

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')

# Cleaning function
def clean_text(text):
    text = str(text)  # safety (in case of null values)
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# Apply cleaning
print("Cleaning started...")
data["text"] = data["text"].apply(clean_text)
print("Cleaning done ✅")

# Show cleaned data
print("\nSample cleaned text:\n")
print(data["text"].head())
from sklearn.feature_extraction.text import TfidfVectorizer

print("\nConverting text to numbers (TF-IDF)...")

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(data["text"])
y = data["label"]

print("Conversion done ✅")
print("Shape of X:", X.shape)
from sklearn.model_selection import train_test_split

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Split done ✅")
from sklearn.naive_bayes import MultinomialNB

print("\nTraining model...")

model = MultinomialNB()
model.fit(X_train, y_train)

print("Model trained ✅")
from sklearn.metrics import accuracy_score

print("\nTesting model...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
def predict_news(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    result = model.predict(vector)

    if result[0] == 0:
        return "REAL NEWS"
    else:
        return "FAKE NEWS"


# Test with your own input
print("\n--- Test Prediction ---")
sample = input("Enter a news headline or text: ")
print("Prediction:", predict_news(sample))