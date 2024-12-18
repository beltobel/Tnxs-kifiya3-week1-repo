import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Download required NLTK data
nltk.download('vader_lexicon')
nltk.download('stopwords')

def analyze_sentiment(df):
    """
    Analyze sentiment of headlines using SentimentIntensityAnalyzer.

    Parameters:
    - df (pd.DataFrame): DataFrame containing a 'headline' column.

    Returns:
    - pd.DataFrame: DataFrame with 'sentiment' and 'sentiment_label' columns added.
    """
    sia = SentimentIntensityAnalyzer()
    
    # Use batch processing for efficiency
    sentiments = df['headline'].apply(lambda x: sia.polarity_scores(x) if isinstance(x, str) else {'compound': 0})
    sentiment_labels = sentiments.apply(lambda x: 
        'positive' if x['compound'] > 0.05 else 
        ('negative' if x['compound'] < -0.05 else 'neutral')
    )
    
    df['sentiment'] = sentiments
    df['sentiment_label'] = sentiment_labels
    return df

def extract_keywords(df):
    """
    Extract keywords from headlines using CountVectorizer.

    Parameters:
    - df (pd.DataFrame): DataFrame containing a 'headline' column.

    Returns:
    - tuple: (list of feature names, sparse matrix of keyword frequencies)
    """
    # Get custom stop words from NLTK and extend as needed
    nltk_stop_words = stopwords.words('english')
    custom_stop_words = set(nltk_stop_words)  # Add more custom words if required
    
    vectorizer = CountVectorizer(stop_words=custom_stop_words)
    
    # Vectorize the text
    X = vectorizer.fit_transform(df['headline'].dropna())
    feature_names = vectorizer.get_feature_names_out()
    
    return feature_names, X

# Example usage:
# df = pd.DataFrame({'headline': ["Breaking news today!", "Everything is terrible.", "Just a neutral day."]})
# df = analyze_sentiment(df)
# keywords, frequencies = extract_keywords(df)
