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
    sentiment_df.fillna(0)
    prod_df.fillna(0)
    return args

# %%
args = load_data()
prod_df, sentiment_df, hours_worked_df = args
print(prod_df)

plt.plot(np.repeat(np.array([np.linspace(2005, 2020, 16)]).transpose(), 20, 1), prod_df.iloc[0:20, 2:-1].to_numpy(dtype = np.float64).T)

#%%

np.corrcoef(prod_df.iloc[0:20, 2:-1].to_numpy(dtype = np.float64).T[:, 0:20].T)