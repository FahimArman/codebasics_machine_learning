import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from sklearn import linear_model
from sklearn.linear_model import LinearRegression

df = pd.read_csv("homeprices.csv")

print(df)

plt.xlabel('area(sqr ft)')
plt.ylabel('price(US$)')

plt.scatter(df.area, df.price,color='red',marker='+')
#plt.show()

reg = LinearRegression()
reg.fit(df[['area']],df.price)

print(reg.predict([[3300]]))

print(reg.coef_)
print(reg.intercept_)

print(reg.coef_*3300+reg.intercept_)


plt.xlabel('area', fontsize=20)
plt.ylabel('price', fontsize=20)
plt.scatter(df.area,df.price, color='red',marker='+')
plt.plot(df.area,reg.predict(df[['area']]),color='blue')
#plt.show()


d= pd.read_csv("areas.csv")
print(d.head(3))

p =  reg.predict(d)
d['prices']=p

d.to_csv("prediction.csv")
d.to_csv("predictions.csv", index=False)













