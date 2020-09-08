#sklearn joblib

from joblib import dump, load
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("homeprices.csv")

model = LinearRegression()
model.fit(df[['area']],df.price)

print(model.predict([[3300]]))
print("-" * 20)

dump(model,'model.joblib')
mj=load('model.joblib')

print(mj.predict([[3300]]))


