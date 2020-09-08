from sklearn.datasets import load_iris

iris = load_iris()
#print(dir(iris))

#print(list(iris.feature_names))
#print(iris.data)
#print(iris.DESCR)
#print(iris.target)
#print(list(iris.target_names))


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.data,iris.target, test_size=0.2, random_state=10)

print(len(X_train))
print(len(X_test))


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(X_train,y_train)

print("Score: ",model.score(X_test,y_test))

print("prediction: ",model.predict([iris.data[67]]))
print("Actual number: ",[iris.target[67]])




