import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    df['headline_length'] = df['headline'].apply(len)
    df['date'] = pd.to_datetime(df['date'])
    return df