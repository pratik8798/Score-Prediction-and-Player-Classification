# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:19:13 2019

@author: Dell
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

base='F:\STC\FINAL IMPLEMENTATION\DATA\CSV'

team="India"
file_name=team+"_bowling.csv";

file=os.path.join(base,file_name)

dataset=pd.read_csv(file)
dataset=dataset[dataset.Innings!='-']
dataset=dataset[dataset.Average!='-']

#X=dataset.iloc[:,[2,3,4,6,7,8,9]]
X=dataset.iloc[:,[4,6,7,8]]

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X=sc_X.fit_transform(X)


from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)                                        #calculates wcss inertia_

plt.plot(range(1,11),wcss)
plt.title("The elbow method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

from sklearn.cluster import AgglomerativeClustering

hc=AgglomerativeClustering(n_clusters=3)
y_hc=hc.fit_predict(X)

plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=100,c='red',label='Yes')                       
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=100,c='blue',label='ok')                       
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=100,c='green',label='almost No')               

plt.scatter(hc.cluster_centers_[:,0],hc.cluster_centers_[:,1],s=300,c='yellow',label='Centroid')
plt.title("Clusters of players")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()                      


kmeans=KMeans(n_clusters=3,init="k-means++",max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(X)



plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='red',label='Yes')                       
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='blue',label='ok')                       
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='green',label='almost No')                       

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroid')
plt.title("Clusters of players")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()                      

selected=[]
data=dataset.values.tolist()
for i in range(len(dataset)):
    arr=data[i][1].split("-")
    if(int(arr[1])==2019):
        if(y_kmeans[i] == 0):
            data[i].append(y_kmeans[i])
            selected.append(data[i])
            
selected=sorted(selected,key=lambda x:(float(x[7]),float(x[6]),float(x[8])))


print("Based on bowling performances player selected")
for player in selected:
    print(player[0])


features=["Wickets","Average","Economy rate","Strike rate"]

from yellowbrick.cluster import SilhouetteVisualizer

from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=1000, n_features=12, centers=8, shuffle=True, random_state=42)

visualizer=SilhouetteVisualizer(kmeans)

visualizer.fit(X)

from yellowbrick.features import Rank1D

visualizer=Rank1D(features=features,algorithm="shapiro")

visualizer.fit(X,y_kmeans)
visualizer.transform(X)


