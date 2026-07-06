import streamlit as st
import joblib
import re

# Load the pkl files
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return text



st.title("Sentiment Analysis Tool")
user_input = st.text_area("Enter your product review here:")

if st.button("Predict Sentiment"):
    if user_input:
        # Preprocess and predict
        cleaned_input = clean_text(user_input)
        vectorized_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(vectorized_input)

        print(f"User Review :",{user_input})
        st.success(f"The predicted sentiment is: {prediction[0]}")
    else:
        st.warning("Please enter some text.")