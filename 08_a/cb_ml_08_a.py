import pandas as pd

df = pd.read_csv("insurance_data.csv")

print(df.head())
print(df.shape)

X = df[['age']]
y = df['bought_insurance']

from sklearn.model_selection import train_test_split

X_train, X_test,y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=10)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(X_train,y_train)
print("Prediction: ",model.predict(X_test))
print("Actual Model: \n",y_test)

print("Model score : ",model.score(X_test,y_test))

print("Model Probability: \n",model.predict_proba(X_test))
print("Age 25 prediction: ",model.predict([[25]]))
print("Age 55 prediction: ",model.predict([[55]]))




