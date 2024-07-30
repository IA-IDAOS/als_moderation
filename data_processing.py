import pandas as pd

def load_data(filename):
    return pd.read_csv(filename)

def preprocess_data(df):
    df['Full Context'] = df['Services'] + ' ' + df['Problématiques'] + ' ' + df['Questions']
    return df
