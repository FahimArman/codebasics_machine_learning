import pandas as pd 
df = pd.read_csv("titanic.csv")

#print(df.head())

df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
#print(df.head())

target = df.Survived
inputs = df.drop('Survived',axis='columns')

#print(inputs.head())

dummies = pd.get_dummies(inputs.Sex)
#print(dummies.head(3))

inputs = pd.concat([inputs,dummies],axis='columns')
#print(inputs.head(3))

inputs.drop('Sex', axis='columns', inplace = True)
print(inputs.head(10))

from math import floor
print(inputs.columns[inputs.isna().any()])
inputs.Age = inputs.Age.fillna(floor(inputs.Age.mean()))

print(inputs.head(10))



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(inputs,target,test_size=0.2)

print(len(inputs))
print(len(X_train))
print(len(X_test))



from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))





