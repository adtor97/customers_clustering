# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:00:19 2020

@author: USUARIO
"""
#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, Normalizer, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
import pickle as pk

#%%
path_to_file = "C:\\Users\\USUARIO\\Desktop\\Python\\Repo\\customers_clustering\\inputs\\customers.xlsx"
df = pd.read_excel(path_to_file)
df = shuffle(df)
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
#Apply log transformation for skewed data

for column in df.drop(["ID_customer"], axis = 1).columns:
    new_column = "converted_" + column
    df[new_column] = np.log(df[column])    
    df[new_column] = df[new_column].replace([np.inf, -np.inf], 0)

converted_columns = [col for col in df.columns if ('converted_' in col)]

#%%
print(df.columns)
#%%
#Apply PCA to new columns
PCA_columns = ['prop_web_purchases', 'prop_app_purchases', 'prop_mon_thur_purchases',
       'prop_frid_sund_purchases', 'prop_early_purchases',
       'prop_lunch_purchases', 'prop_afterlunch_purchase',
       'prop_evening_purchase', 'prop_weekly_lunch',
       'prop_weekend_big_ticket']

#Choose components
std_clf = make_pipeline(Normalizer(), PCA(n_components=6))
std_clf.fit(df[PCA_columns])
print(std_clf["pca"].explained_variance_ratio_)

#%%
#Apply PCA
X_transformed = std_clf.fit_transform(df[PCA_columns])
df_transformed = pd.DataFrame(X_transformed)
df = pd.concat([df, df_transformed], axis = 1)
print(df.head(), df.columns, df.shape)

#%%
#Apply proportion transformation for purchases variables
columns_associated = [['web_purchases',
       'app_purchases'], ['mon_thur_purchases', 'frid_sund_purchases'],
       ['early_purchases', 'lunch_purchases', 'afterlunch_purchase',
       'evening_purchase'], ['weekly_lunch'], ['weekend_big_ticket']
                        
                        ]

for columns in columns_associated:
    if len(columns) == 1:
        column = columns[0]
        df["prop_"+column] = (df[column] / df["purchases"]).fillna(0)
    else:
        for column in columns:
            df["prop_"+column] = (df[column] / df[columns].sum(axis=1)).fillna(0)

print(len(df.columns))


#%%
#Analize data distribution after log, PCA and proportions
f, axes = plt.subplots(7, 7, figsize=(20, 20), sharex=False)
for i, feature in enumerate(df.drop("ID_customer", axis = 1).columns):
    x = df[feature]
    x = x.replace([np.inf, -np.inf], 0)
    sns.distplot(x , color="skyblue", ax=axes[i%7, i//7])
    
#%%
#Prepare variables for kmeans


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
for i in range (10):
	kmeans = KMeans(n_clusters=i+1).fit(df[selected_columns])
	inx.append(i+1)
	iny.append(kmeans.inertia_)
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
#Clustering 3, 4, 5, 6

# Cluster data using selected K
X = df[selected_columns]

n_clusters = 3
kmeans_3 = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('clustering', KMeans(n_clusters))
])

kmeans_3.fit(X)
# Save model to inference phase
pk.dump(kmeans_3, open('C:\\Users\\USUARIO\\Desktop\\Python\\Repo\\customers_clustering\\outputs\\kmeans_3.pk', 'wb'))

# Cluster data using selected K
n_clusters = 4
kmeans_4 = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('clustering', KMeans(n_clusters))
])

kmeans_4.fit(X)
# Save model to inference phase
pk.dump(kmeans_4, open('C:\\Users\\USUARIO\\Desktop\\Python\\Repo\\customers_clustering\\outputs\\kmeans_4.pk', 'wb'))

# Cluster data using selected K
n_clusters = 5
kmeans_5 = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('clustering', KMeans(n_clusters))
])

kmeans_5.fit(X)
# Save model to inference phase
pk.dump(kmeans_5, open('C:\\Users\\USUARIO\\Desktop\\Python\\Repo\\customers_clustering\\outputs\\kmeans_5.pk', 'wb'))

# Cluster data using selected K
n_clusters = 6
kmeans_6 = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('clustering', KMeans(n_clusters))
])

kmeans_6.fit(X)
# Save model to inference phase
pk.dump(kmeans_6, open('C:\\Users\\USUARIO\\Desktop\\Python\\Repo\\customers_clustering\\outputs\\kmeans_6.pk', 'wb'))

#%%
y_predict_3 = kmeans_3.predict(df[selected_columns])
y_predict_4 = kmeans_4.predict(df[selected_columns])
y_predict_5 = kmeans_5.predict(df[selected_columns])
y_predict_6 = kmeans_6.predict(df[selected_columns])

df["kmeans_clusters_3"]=y_predict_3
df["kmeans_clusters_4"]=y_predict_4
df["kmeans_clusters_5"]=y_predict_5
df["kmeans_clusters_6"]=y_predict_6

#%%
df.to_csv('C:\\Users\\USUARIO\\Desktop\\Python\\Repo\\customers_clustering\\outputs\\customers_clusterized.csv', index = False)

