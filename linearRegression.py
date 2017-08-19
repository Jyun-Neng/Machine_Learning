import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def readfile(data_path):
	df = pd.read_csv(data_path, header=0)
	Input = df['Input']
	Output = df['Output']
	m1 = []
	for x in Input:
		m1.append(1)
	return Input, Output, m1

def linearRegression(Input, Output, m1):	 
	m = np.matrix([m1, Input], dtype=int)
	m = m.T
	y = np.asmatrix(Output,dtype=int)
	y = y.T

	B = (m.T * m).I * (m.T * y)		# B = (m^t * m)^(-1) * (m^t * y) is the least square matrix
	B = np.asmatrix(B, dtype=int)	# B stores w0 and w1

	input_size = np.size(Input)
	input_range = Input.values
	input_range = np.sort(input_range)
	max_num = input_range[input_size-1]
	min_num = input_range[0]

	lin_x = []
	lin_y = []
	for x in range(min_num,max_num+1):
		lin_x.append(x)
		X = np.matrix([1, x])
		lin_y.append(int(X * B))	# y = w0 + w1 * x
	return lin_x, lin_y, B
	

data_path = './data.csv'
Input, Output, m1 =  readfile(data_path)
lin_x, lin_y, B = linearRegression(Input, Output, m1)

print(B)
print(lin_x)
print(lin_y) 


#======================plot============================

fig, a1 = plt.subplots()
a1.plot(Input, Output, 'ro')	# plot original data
a1.plot(lin_x, lin_y, label='linear regression')	# plot the linear regression line.
a1.set_xlabel('Input')
a1.set_ylabel('Output')
plt.show()
