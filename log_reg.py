import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

file_path = "spam_emails.csv"
df = pd.read_csv(file_path)

df['label'] = df['label'].str.strip().str.lower()
df = df.dropna(subset=['text', 'label']) 
df['label'] = df['label'].map({'spam': 1, 'not spam': 0})
df = df.dropna(subset=['text'])
df = df.dropna(subset=['label']) 
df['label'] = df['label'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

log_reg = LogisticRegression()
log_reg.fit(X_train_tfidf, y_train)

joblib.dump(log_reg, "logistic_regression.pkl")