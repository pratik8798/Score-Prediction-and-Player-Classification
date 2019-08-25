# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 11:35:05 2019

@author: Dell
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base='F:\STC\FINAL IMPLEMENTATION\DATA\CSV'

team="India"
file = os.path.join(base, team+"_batting.csv")         

dataset=pd.read_csv(file)
dataset=dataset.replace("-",0)


#dataset.fillna(0,inplace=True)

data=dataset.iloc[:,:]
X=dataset.iloc[:,2:]

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X=sc_X.fit_transform(X)

#sc_Y=StandardScaler()
#data[:,2:]=sc_Y.fit_transform(data[:,2:])

#using elbow method to find optimal number of clusters
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

kmeans=KMeans(n_clusters=3,init="k-means++",max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(X)

plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='red',label='Not bad')                       #y_kmeans==0 matches all points belonging to cluster 1
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='blue',label='Yes')                       #y_kmeans==0 matches all points belonging to cluster 1
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='green',label='No')                       #y_kmeans==0 matches all points belonging to cluster 1

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroid')
plt.title("Clusters of players")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()                      

unique, counts = np.unique(y_kmeans, return_counts=True)
count=dict(zip(unique, counts))

'''
from sklearn.cluster import AgglomerativeClustering

hc=AgglomerativeClustering(n_clusters=3,affinity="euclidean")
y_hc=hc.fit_predict(X)
'''
selected=[]
data=dataset.values.tolist()

for i in range(len(dataset)):
    arr=data[i][1].split("-")
    if(int(arr[1])==2019):
        if(y_kmeans[i] in (0,1)):
            data[i].append(y_kmeans[i])
            selected.append(data[i])
            
selected=sorted(selected,key=lambda x:(x[10],float(x[6])),reverse=True)


print("Based on batting performances player selected(descending order")
for player in selected:
    print(player[0])


features=[
"Matches",
"Innings",
"Not outs",
"Runs",
"Average",
"Strike Rate",
"100s",
"50s"]

'''

from yellowbrick.features import Rank1D

visualizer=Rank1D(features=features,algorithm="shapiro")

visualizer.fit(X,y_kmeans)
visualizer.transform(X)
'''