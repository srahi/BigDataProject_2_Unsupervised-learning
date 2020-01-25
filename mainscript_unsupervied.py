import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline


df = pd.read_csv('dow_jones_index.data',header=0)
df.head()

df.shape

df.describe()

stocks = df['stock'].value_counts().index
stocks

df['open'] = df['open'].apply(lambda x: float(x.strip('$')))
df.head()


df = df[['stock','date','open','percent_change_price']]
df.head()

retunrs = {}
volatility = {}
for stock in stocks:
    r = (df[df['stock'] == stock]['percent_change_price'].mean())
    v = (df[df['stock'] == stock]['open'].std())
    retunrs[stock] = r
    volatility[stock] = v
    
retunrss = pd.Series(retunrs)
volatilitys = pd.Series(volatility)

dfn = pd.concat([retunrss, volatilitys ], axis =1)
dfn.columns = ['returns' , 'volatility']
dfn


ssq = []
for n in range(1,25):
    kmeans_model = KMeans(n_clusters=n, random_state=3)
    kmeans_model.fit(dfn)
    ssq.append(kmeans_model.inertia_)
plt.plot(range(1,25), ssq, marker='o')
plt.xlabel("Number of clusters")
plt.ylabel("Within-cluster SSQ")
plt.title("Scree Plot")
plt.show()


km = KMeans(n_clusters=3, random_state=243,n_init=20,max_iter=5000)
km.fit(dfn)
predicted_clusters = km.predict(dfn)
dfn['cluster'] = predicted_clusters
plt.figure(figsize=(12,10))
plt.scatter(x='returns',y='volatility',c = 'cluster',cmap = 'viridis',data =dfn,linewidths = 10)
plt.xlabel('returns')
plt.ylabel('volatility')
for i,c in enumerate(list(dfn.index)):
    x=dfn['returns'][i]
    y=dfn['volatility'][i]
    plt.text(x+0.02,y+0.02,c)
plt.show()

silhouette_score(dfn, predicted_clusters)
