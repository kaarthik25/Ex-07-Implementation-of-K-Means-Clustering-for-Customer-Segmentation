# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the required packages.</br>
2. Import the dataset to work on.</br>
3. From sklearn module import kmeans.</br>
4. Define number of clusters to be made.</br>
5. Assign the cluster values.</br>
6. Plot the cluster using matplotlib.pyplot</br>
7. End the program. 

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Kaarthikeyan.S
RegisterNumber: 212220040068 
*/
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("Mall_Customers.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.cluster import KMeans

wcss = []
for i in range(1,11):
  kmeans = KMeans(n_clusters=i,init="k-means++")
  kmeans.fit(data.iloc[:,3:])
  wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("no of clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")
km = KMeans(n_clusters=5)
km.fit(data.iloc[:,3:])
y_pred = km.predict(data.iloc[:,3:])

data["cluster"] =y_pred
df0 = data[data["cluster"]==0]
df1 = data[data["cluster"]==1]
df2 = data[data["cluster"]==2]
df3 = data[data["cluster"]==3]
df4= data[data["cluster"]==4]

plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="blue",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="pink",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="green",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="red",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="orange",label="cluster4")
plt.title("Customer Segment")
plt.legend()
```

## Output:
<img width="654" alt="7ml" src="https://user-images.githubusercontent.com/94525701/173196598-6ebb8bfd-2fa1-414e-a0ba-6abfda8bcae2.png">
<img width="654" alt="7m2" src="https://user-images.githubusercontent.com/94525701/173196604-c7c04bb8-65dc-4f83-95f7-91462b54279b.png">

<img width="435" alt="7ml3" src="https://user-images.githubusercontent.com/94525701/173196621-3597ff82-15b0-4a43-985e-f1f43a8b01d0.png">
<img width="654" alt="7ml4" src="https://user-images.githubusercontent.com/94525701/173196637-5e2411b6-b726-4766-a200-687fd31d9376.png">
<img width="748" alt="7ml5" src="https://user-images.githubusercontent.com/94525701/173196648-eed1b93f-039f-4945-9b81-0c7248c999df.png">
<img width="479" alt="7ml6" src="https://user-images.githubusercontent.com/94525701/173196657-5093be7d-f246-4714-8976-dfe142120360.png">


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
