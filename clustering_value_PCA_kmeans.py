# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:00:19 2020

@author: adtor97 & Claudio9701
"""
#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, Normalizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
#from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
import pickle as pk

#%%
path_to_file = "C:\\Users\\USUARIO\\Desktop\\Python\\Repo\\customers_clustering\\outputs\\customers_wrangled.csv"
df = pd.read_csv(path_to_file)
df = shuffle(df)
df = df.reset_index(drop = True)
original_columns = df.drop("ID_customer", axis = 1).columns.to_list()

#%%
print(df.head(), df.columns, df.shape)
print(df.describe())
#%%
#Analize data distribution
f, axes = plt.subplots(7, 7, figsize=(30, 30), sharex=False)
for i, feature in enumerate(df.drop("ID_customer", axis = 1).columns):
    x = df[feature]
    x = x.replace([np.inf, -np.inf], 0)
    sns.distplot(x , color="skyblue", ax=axes[i%7, i//7])

#%%
#Analize PCA for new columns
PCA_selected_columns = ['converted_avg_ticket',
       'converted_total_amount', 'converted_ticket_std_dev',
       'converted_purchases', 'converted_frequency', 'converted_inactive_days']

#Choose components
n_comps = 3
pca = make_pipeline(Normalizer(), PCA(n_components = n_comps))
pca.fit(df[PCA_selected_columns])
print(pca["pca"].explained_variance_ratio_)

#%%
#Apply PCA
X_transformed = pca.fit_transform(df[PCA_selected_columns])
PCA_columns = ["PCA_" + str(i) for i in range(n_comps)]
df_transformed = pd.DataFrame(X_transformed, columns = PCA_columns)
df = pd.concat([df, df_transformed], axis = 1)
print(df.head(), df.columns, df.shape)

#%%
#Analize data distribution after log, PCA and proportions
f, axes = plt.subplots(8, 8, figsize=(30, 30), sharex=False)
for i, feature in enumerate(df.drop("ID_customer", axis = 1).columns):
    x = df[feature]
    x = x.replace([np.inf, -np.inf], 0)
    sns.distplot(x , color="skyblue", ax=axes[i%8, i//8])
#%%
#Prepare variables for kmeans

selected_columns = PCA_columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('outliers', RobustScaler()),
    ('normalize', Normalizer())
    ])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, selected_columns)
        ])

#%%
#Decide clusters numbers

inx=[]
iny=[]

#iteraciones de kmeans para guardar el valor de suma de errores y su cantidad de clusters
for i in range (1, 10):
    n_clusters = i + 1
    kmeans = Pipeline(steps = [
            ('preprocessing', preprocessor),
            ('clustering', KMeans(n_clusters))
            ])
    kmeans.fit(df[selected_columns])
    inx.append(i+1)
    iny.append(kmeans['clustering'].inertia_)

print (inx, iny)

#plotear inertia (error) y numero de cluster para decidir k
print(inx)
print (iny)
plt.plot(inx,iny)
plt.show()
plt.savefig("C:\\Users\\USUARIO\\Desktop\\Python\\Repo\\customers_clustering\\outputs\\kmeans_selection.png")
plt.clf()
plt.close()

#%%
#Clustering n = 4

# Cluster data using selected K
n_clusters = 4
kmeans_4 = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('clustering', KMeans(n_clusters))
])

kmeans_4.fit(df[selected_columns])
# Save model to inference phase
pk.dump(kmeans_4, open('C:\\Users\\USUARIO\\Desktop\\Python\\Repo\\customers_clustering\\outputs\\kmeans_4.pk', 'wb'))

#%%
y_predict_4 = kmeans_4.predict(df[selected_columns])
df["kmeans_4"]=y_predict_4

#%%
df.to_csv('C:\\Users\\USUARIO\\Desktop\\Python\\Repo\\customers_clustering\\outputs\\customers_kmeans.csv', index = False)
#%%
#Analize 4 clusters distribution
columns_analysis = ['avg_ticket', 'total_amount', 'ticket_std_dev',
       'purchases', 'frequency', 'inactive_days']

df_grouped = df[columns_analysis + ["kmeans_4"]]
df_grouped = df_grouped.groupby("kmeans_4", as_index = False).mean()
df_grouped["total_customers"] = df[["kmeans_4", "ID_customer"]].groupby("kmeans_4", as_index = False).count()["ID_customer"]
columns_grouped = df_grouped.drop("kmeans_4", axis = 1).columns.to_list()
df_grouped

#%%
f, axes = plt.subplots(3, 3, figsize=(20, 20), sharex=False)
for i, feature in enumerate(columns_grouped):
   
    x = df_grouped[feature]
    x = x.replace([np.inf, -np.inf], 0)
    sns.barplot(x = df_grouped["kmeans_4"] , y = x , color="skyblue", ax=axes[i%3, i//3])
plt.savefig("C:\\Users\\USUARIO\\Desktop\\Python\\Repo\\customers_clustering\\outputs\\kmeans_means_comparison.png")
plt.clf()
plt.close()