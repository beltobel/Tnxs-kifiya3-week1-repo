import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('vader_lexicon')

def analyze_sentiment(df):
    sia = SentimentIntensityAnalyzer()
    df['sentiment'] = df['headline'].apply(lambda x: sia.polarity_scores(x))
    df['sentiment_label'] = df['sentiment'].apply(lambda x: 'positive' if x['compound'] > 0.05 else ('negative' if x['compound'] < -0.05 else 'neutral'))
    return df

def extract_keywords(df):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['headline'])
    return vectorizer.get_feature_names_out(), X.toarray()