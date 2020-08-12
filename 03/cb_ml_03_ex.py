from word2number import w2n
import math
from sklearn import linear_model
import pandas as pd


df = pd.read_csv("hiring.csv")
df.columns = [c.replace(' ', '_') for c in df.columns]
df.columns = [c.replace('$', '_') for c in df.columns]
df.columns = [c.replace('(', '_') for c in df.columns]
df.columns = [c.replace(')', '_') for c in df.columns]
print(df)


avg_test_score = math.floor(df.test_score_out_of_10_.median())
#median_test_score = math.floor(d['test_score(out of 10)'].mean())
print("Avg test score: ",avg_test_score)

df.test_score_out_of_10_=df.test_score_out_of_10_.fillna(avg_test_score)
#d['test_score(out of 10)'] = d['test_score(out of 10)'].fillna(median_test_score)

df.experience = df.experience.fillna("zero")
print(df)

df.experience = [w2n.word_to_num(value) for value in df.experience]
#df.experience = df.experience.apply(w2n.word_to_num)
print(df)


reg=linear_model.LinearRegression()
reg.fit(df[['experience','test_score_out_of_10_','interview_score_out_of_10_']],df.salary___)

print("Expected salary: ", reg.predict([[2,9,6]]))
print("Expected salary: ", reg.predict([[12,10,10]]))

