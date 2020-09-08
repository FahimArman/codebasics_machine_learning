import pandas as pd

##loading the data
df = pd.read_csv("carprices.csv")
#print(df)

## change from town to number
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Car Model'] = le.fit_transform(df['Car Model'])
#print(df)

X = df[['Car Model','Mileage','Age(yrs)']].values
#print(X)
y = df[['Sell Price($)']].values
#print(y)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("Car Model", OneHotEncoder(), [0])], remainder = 'passthrough')
X = ct.fit_transform(X)

X = X[:,1:]
#print(X)

from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(X,y)

print(model.predict([[0,1,45000,4]]))
print(model.predict([[1,0,86000,7]]))

print("model accuracy: ", model.score(X,y))






