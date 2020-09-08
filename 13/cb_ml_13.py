from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import pandas as pd


df = pd.read_csv("income.csv")
#print(df.head())

#plt.scatter(df['Age'],df['Income($)'])
#plt.show()

km = KMeans(n_clusters=3) 
y_predicted = km.fit_predict(df[['Age','Income($)']])
#print(y_predicted)

df['cluster1'] = y_predicted

#df0 = df[df.cluster1==0]
#df1 = df[df.cluster1==1]
#df2 = df[df.cluster1==2]

#plt.scatter(df0.Age,df0['Income($)'],color='green')
#plt.scatter(df1.Age,df1['Income($)'],color='red')
#plt.scatter(df2.Age,df2['Income($)'],color='blue')

#plt.xlabel("Age")
#plt.ylabel("Income($)")
#plt.legend()
#plt.show()



scaler=MinMaxScaler()
scaler.fit(df[['Income($)']])
df['Income($)']=scaler.transform(df[['Income($)']])

scaler.fit(df[['Age']])
df['Age']=scaler.transform(df[['Age']])
#print(df.head())

km1 = KMeans(n_clusters=3) 
y_predicted = km1.fit_predict(df[['Age','Income($)']])

df['cluster2']=y_predicted

print(km1.cluster_centers_)
print(df.head())


df3 = df[df.cluster2==0]
df4 = df[df.cluster2==1]
df5 = df[df.cluster2==2]

plt.scatter(df3.Age,df3['Income($)'],color='green')
plt.scatter(df4.Age,df4['Income($)'],color='red')
plt.scatter(df5.Age,df5['Income($)'],color='blue')

plt.scatter(km1.cluster_centers_[:,0],km1.cluster_centers_[:,1],color='black',marker='*',label='centroid')

plt.xlabel("Age")
plt.ylabel("Income($)")
plt.legend()
plt.show()

k_rng = range(1,10)
sse=[]

for k in k_rng:
	km2 = KMeans(n_clusters=k)
	km2.fit(df[['Age','Income($)']])
	sse.append(km2.inertia_)

plt.xlabel('k')
plt.ylabel('Sum of squered error')
plt.plot(k_rng,sse)
plt.show()
