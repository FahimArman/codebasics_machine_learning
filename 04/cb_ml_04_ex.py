import numpy as np
import pandas as pd

def gradient_descent(x,y):
	m_curr = 0
	b_curr = 0
	iterations = 100
	n = len(x)
	learning_rate = 0.0002
	
	for i in range(iterations):
		y_predicted = m_curr * x + b_curr
		
		cost = (1/n) * sum([val**2 for val in y-y_predicted])
		
		mb = -(2/n)*sum(x*(y-y_predicted))
		db = -(2/n)*sum(y-y_predicted)
		
		m_curr = m_curr - (learning_rate*mb)
		b_curr = b_curr - (learning_rate*db)
		
		print("m:{},b:{}, cost:{}, i:{}".format(m_curr,b_curr,cost,i))



df=pd.read_csv("test_scores.csv")
x = df.math
y= df.cs
print(df)
print("-" * 20)
gradient_descent(x,y)
