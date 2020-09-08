import pandas as pd

df = pd.read_csv("titanic.csv")
#print(df.head())

#selected_columns = df[["col1","col2"]]
selected_columns = df[['Pclass','Sex','Age','Fare']]
inputs = selected_columns.copy()
#print(inputs.head())
target = df['Survived']

from sklearn.preprocessing import LabelEncoder
le_sex=LabelEncoder()

inputs['sex_n'] = le_sex.fit_transform(inputs['Sex'])
#print(inputs.head())

inputs_n= inputs.drop(['Sex'], axis='columns')
print(inputs_n.head())

from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(inputs_n,target)

#print(model.predict([[2,0,1]]))

print("score: ",model.score(inputs_n,target))
