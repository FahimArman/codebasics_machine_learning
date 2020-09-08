import pandas as pd

df = pd.read_csv("spam.csv")
df['spam']= df['Category'].apply(lambda x:1 if x =='spam' else 0)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df.Message,df.spam,test_size=0.2)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline

clf = Pipeline([
		('vectorize',CountVectorizer()),
		('nb',MultinomialNB())
])

clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))



