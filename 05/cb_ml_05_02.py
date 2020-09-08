#sklearn joblib

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("homeprices.csv")

model = LinearRegression()
model.fit(df[['area']],df.price)

print(model.predict([[3300]]))
print("-" * 20)

joblib.dump(model,'model_joblib')
mj=joblib.load('model_joblib')

print(mj.predict([[3300]]))


