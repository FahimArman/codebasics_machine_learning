from sklearn.datasets import load_digits

digits = load_digits()
print(dir(digits))

#print(digits.data[0])
#print(digits.target[0])
#print(digits.data[0:5])
#print(digits.target[0:5])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target, test_size=0.2, random_state=10)

#print(len(X_train))
#print(len(X_test))

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(X_train,y_train)

print("Score: ",model.score(X_test,y_test))

print("prediction: ",model.predict([digits.data[67]]))
print("Actual number: ",[digits.target[67]])


#Confusion matrix
y_predicted = model.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_predicted)
print(cm)

import seaborn as sn
import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
