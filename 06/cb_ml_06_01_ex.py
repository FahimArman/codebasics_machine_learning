import pandas as pd

##loading the data
df = pd.read_csv("carprices.csv")
#print(df)

##make the dummies
dummies = pd.get_dummies(df[['Car Model']])
#print(dummies)

##merge the file with the main data and dummies
merged = pd.concat([df,dummies], axis='columns')
#print(merged)

final = merged.drop(['Car Model','Car Model_Audi A5'], axis='columns')
#print(final)

## X values
X = final.drop('Sell Price($)', axis='columns')
#print(X)
## y values
y=final[['Sell Price($)']]
#print(y)


from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(X,y)

print(model.predict([[45000,4,0,1]]))
print(model.predict([[86000,7,1,0]]))


print("model accuracy: ", model.score(X,y))
