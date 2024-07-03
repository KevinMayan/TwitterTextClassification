import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

toxic_model = joblib.load('toxic_tweets_pipeline.pkl')

sentiment_model = joblib.load('twitter_sentiment_pipeline.pkl')

def classify_toxicity(text):
    prediction = toxic_model.predict([text])[0]
    categories = {0.: "Not Toxic", 1.: "Toxic"}
    return categories[prediction]

def analyze_sentiment(text):
    prediction = sentiment_model.predict([text])[0]
    categories = {-1. : "Negative", 0. : "Neutral", 1. : "Positive"}
    return categories[prediction]


def main():
    st.title('Text AI Analysis')
    st.write('Enter your text below:')

    # User input text box
    user_input = st.text_area('Input Text:', '')

    # Classify toxicity button
    if st.button('Classify Toxicity'):
        if user_input:
            toxicity = classify_toxicity(user_input)
            st.write(f'Toxicity Prediction: {toxicity}')

    # Analyze sentiment button
    if st.button('Analyze Sentiment'):
        if user_input:
            sentiment = analyze_sentiment(user_input)
            st.write(f'Sentiment Analysis (Polarity): {sentiment}')

if __name__ == '__main__':
    main()

