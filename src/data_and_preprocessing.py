import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    df['headline_length'] = df['headline'].apply(len)
    # df['date'] = pd.to_datetime(df['date'])
    # df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    # df['date'] = pd.to_datetime(df['date'], errors='coerce')
    #Clean the date column
    df['date'] = df['date'].str.strip()
    def parse_dates(date_str):
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d", "%d-%m-%Y"):
            try:
                return pd.to_datetime(date_str, format=fmt)
            except ValueError:
                continue
            return pd.NaT


    df['date'] = df['date'].apply(parse_dates)

    return df