import pandas as pd

df = pd.read_csv("HR_comma_sep.csv")
#print(df.head())
print(df.shape)

from matplotlib import pyplot as plt
#X= df[['satisfaction_level']]
#X= df[['last_evaluation']]
#X= df[['number_project']]
#X= df[['average_montly_hours']] #may be
X= df[['time_spend_company']]
y= df['left']

#plt.scatter(X,y,marker='+', color= 'red')
#plt.show()

from sklearn.model_selection import train_test_split

X_train, X_test,y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=10)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(X_train,y_train)
model.predict(X_test)
#print("Actual Model: \n",y_test)

print("Model score : ",model.score(X_test,y_test))




