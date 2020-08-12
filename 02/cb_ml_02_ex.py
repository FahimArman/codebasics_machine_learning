import pandas as pd
from sklearn import linear_model

df = pd.read_csv("canada_per_capita_income.csv")
df.columns = [c.replace(' ', '_') for c in df.columns]
df.columns = [c.replace('$', '_') for c in df.columns]
df.columns = [c.replace('(', '_') for c in df.columns]
df.columns = [c.replace(')', '_') for c in df.columns]
print(df.head(3))


reg = linear_model.LinearRegression()
reg.fit(df[['year']],df.per_capita_income__US__)

print(reg.predict([[2020]]))




