import math
import pandas as pd
from sklearn import linear_model

df = pd.read_csv("homeprices.csv")
#print(df)

median_bedrooms = math.floor(df.bedrooms.median())
#print(median_bedrooms)
df.bedrooms = df.bedrooms.fillna(median_bedrooms)
#print(df)

reg = linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']],df.price)

print(reg.coef_)
print(reg.intercept_)
print("-" * 20)
print(reg.predict([[3000,3,40]]))
print(reg.predict([[2500,4,5]]))

