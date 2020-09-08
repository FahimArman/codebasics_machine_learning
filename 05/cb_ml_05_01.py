#pickle

import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("homeprices.csv")

model = LinearRegression()
model.fit(df[['area']],df.price)

print(model.predict([[3300]]))
print("-" * 20)


with open('model_pickle','wb') as f:
	pickle.dump(model,f)

with open('model_pickle','rb') as f:
	mp = pickle.load(f)

print(mp.predict([[3300]]))
