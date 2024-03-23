# Applying-Hierarchical-Clustering-28
```python


import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline
import numpy as np 
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Mall_customers.csv')


# Getting annual income and spending score data.
df1 = df.iloc[:, 2:4].values


import scipy.cluster.hierarchy as shc

# Dendogram 
# x-axis represents the points 
# y-axis is the distance between the clusters 
plt.figure(figsize=(30, 7))
plt.title("Customer Dendrograms")

dend = shc.dendrogram(shc.linkage(df1, method='ward'))

from sklearn.cluster import AgglomerativeClustering

# Specify clusters
# Recursively merges pair of clusters of sample data; uses linkage distance
model = AgglomerativeClustering(n_clusters=5, linkage='ward')

# Fit and return the result of each sample's clustering assignment. 
labels_=model.fit_predict(df1)

plt.figure(figsize=(10, 7))
plt.scatter(df1[:,0], df1[:,1], c=model.labels_, cmap='rainbow')



#Observations:Within the spread, we can see that five separate clusters have been created, forming the agglomerative cluster of five clusters.
#The cluster is represented by red, green, blue, violet, and yellow.






```
