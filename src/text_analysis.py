import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Download necessary NLTK resources
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(text):
    """
    Preprocess the input text: tokenize, remove stopwords, and lower case.
    """
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())  # Tokenize and lower case
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]  # Remove stopwords and non-alphabetic words
    return ' '.join(filtered_tokens)

def analyze_sentiment(df, column_name):
    """
    Perform sentiment analysis on a specific column of a DataFrame.
    """
    sia = SentimentIntensityAnalyzer()
    # Apply preprocessing and sentiment analysis
    df['preprocessed_text'] = df[column_name].apply(preprocess_text)
    df['sentiment'] = df['preprocessed_text'].apply(lambda x: sia.polarity_scores(x))
    df['sentiment_label'] = df['sentiment'].apply(lambda x: 'positive' if x['compound'] > 0.05 else ('negative' if x['compound'] < -0.05 else 'neutral'))
    return df

def extract_keywords(df, column_name):
    """
    Extract keywords from a specific column of a DataFrame using CountVectorizer.
    """
    vectorizer = CountVectorizer(stop_words='english')
    preprocessed_texts = df[column_name].apply(preprocess_text)
    X = vectorizer.fit_transform(preprocessed_texts)
    return vectorizer.get_feature_names_out(), X.toarray()

# Example Usage
data = {
    'headline': ['The stock market is up today.', 'Rain is expected tomorrow.', 'The product launch was a success.'],
    'description': ['Market trends indicate growth.', 'Weather forecasts predict showers.', 'The event was well-received by the public.']
}
df = pd.DataFrame(data)

# Perform sentiment analysis on the 'headline' column
df = analyze_sentiment(df, 'headline')

# Extract keywords from the 'headline' column
keywords, keyword_matrix = extract_keywords(df, 'headline')

print(df)
print("Keywords:", keywords)
