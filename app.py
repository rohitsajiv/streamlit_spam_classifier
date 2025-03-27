import streamlit as st
import joblib

vectorizer = joblib.load("tfidf_vectorizer.pkl")
nb_model = joblib.load("naive_bayes.pkl")
log_reg_model = joblib.load("logistic_regression.pkl")

st.set_page_config(page_title="Email Spam Classifier", page_icon="📧", layout="centered")

st.markdown(
    """
    <style>
        body {
            background-color: #f0f2f6;
        }
        .stApp {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        }
        .title {
            color: #2c3e50;
            text-align: center;
            font-size: 32px;
            font-weight: bold;
        }
        .result {
            font-size: 24px;
            text-align: center;
            font-weight: bold;
            padding: 10px;
            border-radius: 10px;
        }
        .spam {
            background-color: #ffcccc;
            color: #c0392b;
        }
        .not-spam {
            background-color: #d4edda;
            color: #155724;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='title'>📧 Email Spam Classifier</h1>", unsafe_allow_html=True)

email_text = st.text_area("Enter the email content:", height=150)

model_choice = st.selectbox("Choose a model:", ["Naïve Bayes", "Logistic Regression"])

if st.button("Classify Email"):
    if email_text:
        email_tfidf = vectorizer.transform([email_text])
        model = nb_model if model_choice == "Naïve Bayes" else log_reg_model
        prediction = model.predict(email_tfidf)[0]

        if prediction == 0:
            result = "✅ Not Spam"
            result_class = "not-spam"
        else:
            result = "🚨 Spam Email"
            result_class = "spam"
        
        st.markdown(f"<div class='result {result_class}'>{result}</div>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter an email to classify.")
