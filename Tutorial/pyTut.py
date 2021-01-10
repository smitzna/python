import pandas as pd
def add1(df):
    df.iloc[:,0]=df.iloc[:,0].apply(lambda x: x+1)
    return df
