import pandas as pd

df = pd.read_csv("spam.csv")
#print(df.head(3))


#print(df.groupby('Category').describe())
df['spam']= df['Category'].apply(lambda x:1 if x =='spam' else 0)
#print(df.head(3))

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(df.Message,df.spam,test_size=0.2)

from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
X_train_count = v.fit_transform(X_train.values)
#print(X_train_count.toarray()[:3])


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_count,y_train)

X_test_count = v.transform(X_test)
print("score: ",model.score(X_test_count,y_test))

