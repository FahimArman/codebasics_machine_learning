import pandas as pd

##loading the data
df = pd.read_csv("homeprices.csv")
#print(df)

## change from town to number
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df.town = le.fit_transform(df.town)
#print(df)

X = df[['town','area']].values
print(X)
y = df.price
print(y)


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("town", OneHotEncoder(), [0])], remainder = 'passthrough')
X = ct.fit_transform(X)


X = X[:,1:]
print(X)


from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(X,y)

print(model.predict([[1,0,2800]]))
print(model.predict([[0,1,3400]]))

