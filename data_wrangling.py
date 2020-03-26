# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 16:06:40 2020

@author: adtor97
"""

#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

#%%
path_to_file = "C:\\Users\\USUARIO\\Desktop\\Python\\Repo\\customers_clustering\\inputs\\customers.csv"
df = pd.read_csv(path_to_file, sep = ";")
df = shuffle(df)
df = df.reset_index(drop = True)
original_columns = df.drop("ID_customer", axis = 1).columns.to_list()
print(df.head(), df.columns, df.shape)
#%%
print(df.describe())
#%%
#Analize data distribution
f, axes = plt.subplots(5, 5, figsize=(20, 20), sharex=False)
for i, feature in enumerate(df.drop("ID_customer", axis = 1).columns):
    x = df[feature]
    x = x.replace([np.inf, -np.inf], 0)
    sns.distplot(x , color="skyblue", ax=axes[i%5, i//5])
#%%
#Apply proportion transformation for behavioural variables
columns_associated = [['web_purchases',
       'app_purchases'], ['mon_thur_purchases', 'frid_sund_purchases'],
       ['lunch_purchases', 'evening_purchases'],
       ['low_ticket', 'medium_ticket' , 'big_ticket']
                       ]

for columns in columns_associated:
    if len(columns) == 1:
        column = columns[0]
        df["prop_"+column] = (df[column] / df["purchases"]).fillna(0)
    else:
        for column in columns:
            df["prop_"+column] = (df[column] / df[columns].sum(axis=1)).fillna(0)
prop_columns = [col for col in df.columns if ('prop_' in col)]
print(len(df.columns))
#%%
#Apply log transformation for skewed data

for column in df.drop(["ID_customer"], axis = 1).columns:
    new_column = "converted_" + column
    df[new_column] = np.log(df[column])    
    df[new_column] = df[new_column].replace([np.inf, -np.inf], 0)

converted_columns = [col for col in df.columns if ('converted_' in col)]
print(len(df.columns))

#%%

df.to_csv ("C:\\Users\\USUARIO\\Desktop\\Python\\Repo\\customers_clustering\\outputs\\customers_wrangled.csv", index = False )
