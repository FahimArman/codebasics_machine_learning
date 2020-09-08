import pandas as pd

from sklearn.datasets import load_iris

iris = load_iris()

#print(dir(iris))
#print(iris.feature_names)

df = pd.DataFrame(iris.data,columns = iris.feature_names)
df['target'] = iris.target
#print(df.head())
#print(iris.target_names)

df['flower_name'] = df.target.apply(lambda x:iris.target_names[x])
#print(df.head())


df0=df[df.target==0]
df1=df[df.target==1]
df2=df[df.target==2]

#print(df2.head())

X = df.drop(['target','flower_name'], axis='columns')
print(X.head())
y = df.target

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

print(len(X_train)," ",len(X_test))


from sklearn.svm import SVC
model = SVC()

model.fit(X_train,y_train)

print(model.score(X_test,y_test))






