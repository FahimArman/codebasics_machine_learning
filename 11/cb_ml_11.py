import pandas as pd

from sklearn.datasets import load_digits

digits = load_digits()
print(dir(digits))

df = pd.DataFrame(digits.data)
df['target']=digits.target
#print(df.head())

X = df.drop(['target'],axis='columns')
y = df.target
#print(X.head())


from sklearn.model_selection import train_test_split

X_train,X_test, y_train,y_test = train_test_split(X,y,test_size= 0.2)

print(len(X_train)," : ", len(X_test))



from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train,y_train)

print(model.score(X_test,y_test))

