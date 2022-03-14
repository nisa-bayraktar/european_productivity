# %%
#coding utf8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
def load_data():
    args = []
    prod_df = pd.read_csv('data\productivity.csv')
    sentiment_df = pd.read_csv('data\sentiment.csv')
    hours_worked_df = pd.read_csv('data\lfsa_ewhun2_1_Data.csv')
    args += [prod_df, sentiment_df, hours_worked_df]
    return args

# %%
args = load_data()
prod_df, sentiment_df, hours_worked_df = args
print(prod_df.head())


