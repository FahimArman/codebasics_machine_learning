import pandas as pd

##loading the data
df = pd.read_csv("homeprices.csv")
#print(df)

##make the dummies
dummies = pd.get_dummies(df.town)
#print(dummies)

##merge the file with the main data and dummies
merged = pd.concat([df,dummies], axis='columns')
#print(merged)


final = merged.drop(['town','west windsor'], axis='columns')
#print(final)

## X values
X = final.drop('price', axis='columns')
#print(X)
## y values
y=final.price
#print(y)


from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(X,y)

print(model.predict([[2800,0,1]]))
print(model.predict([[3400,0,0]]))

print("model accuracy: ", model.score(X,y))

