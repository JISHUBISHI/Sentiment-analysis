import streamlit as st
import pickle
import re

st.title('Twitter Sentiment Analysis')
def clean_text(text):
    if isinstance(text, str):
        text = text.lower() 
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text) 
        text = re.sub(r'\s+', ' ', text)  
    return text
user_tweet = st.text_input('Whats on your mind?')
with open('model.pkl', 'rb') as f:
    vectorizer, model1 = pickle.load(f)
    print("Model and vectorizer loaded successfully!")
    user_tweet = user_tweet
    user_tweet_cleaned = clean_text(user_tweet)
    user_tweet_tfidf = vectorizer.transform([user_tweet_cleaned])
    user_tweet_prediction = model1.predict(user_tweet_tfidf)
    print(f"Predicted Sentiment for your tweet: {user_tweet_prediction[0]}")



if user_tweet:
    if user_tweet_prediction == "Neutral":
        sentiment = 'Your sentient is Neutral'
    elif user_tweet_prediction == "Positive":
        sentiment = 'Your sentient is Positive'
    else :
        sentiment = 'Your sentient is Negative'
    st.write(f'Predicted Sentiment: {sentiment}')