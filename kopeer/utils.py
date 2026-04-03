import pandas as pd 

def normalize_df(df, columns=None):
    if columns is None:
        columns = df.columns.tolist()

    df_norm = df.copy()
    df_norm[columns] = df_norm[columns] / 10.0

    return df_norm
