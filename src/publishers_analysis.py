def publisher_activity(df):
    publisher_count = df['publisher'].value_counts()
    return publisher_count

def unique_domains(df):
    df['domain'] = df['publisher'].apply(lambda x: x.split('@')[-1] if '@' in x else 'N/A')
    domain_count = df['domain'].value_counts()
    return domain_count