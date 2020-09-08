import pandas as pd
df = pd.read_csv('carprices.csv')
#print(df.head())


X = df[['Mileage','Age(yrs)']]
y = df[['Sell Price($)']]


from sklearn.model_selection import train_test_split

X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=20)

#print(len(X_train))
#print(len(X_test))

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train,y_train)

print("model values: \n", model.predict(X_test))
print("actual values: \n", y_test)

print(model.score(X_test,y_test))
