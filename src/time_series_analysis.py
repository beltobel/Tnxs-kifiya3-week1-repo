import matplotlib.pyplot as plt

def publication_frequency(df):
    df.set_index('date', inplace=True)
    freq = df.resample('D').size()
    plt.figure(figsize=(12, 6))
    freq.plot()
    plt.title('Publication Frequency Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Publications')
    plt.show()
    return freq