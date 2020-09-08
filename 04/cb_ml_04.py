import numpy as np

def gradient_descent(x,y):
	m_curr = 0
	b_curr = 0
	iterations = 1000
	n = len(x)
	learning_rate = 0.08
	
	for i in range(iterations):
		y_predicted = m_curr * x + b_curr
		
		cost = (1/n) * sum([val**2 for val in y-y_predicted])
		
		mb = -(2/n)*sum(x*(y-y_predicted))
		db = -(2/n)*sum(y-y_predicted)
		
		m_curr = m_curr - (learning_rate*mb)
		b_curr = b_curr - (learning_rate*db)
		
		print("m:{},b:{}, cost:{}, i:{}".format(m_curr,b_curr,cost,i))

x=np.array([1,2,3,4,5])
y=np.array([5,7,9,11,13])

gradient_descent(x,y)
