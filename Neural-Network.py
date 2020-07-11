# import sys
# import numpy as np
# import matplotlib
# print("Python: ", sys.version)
# print("Numpy:", np.__version__)
# print("Matplotlib: ", matplotlib.__version__)
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
nnfs.init()	

X=[ [0.2,0.8,-0.5,1.0],
	[0.5,-0.91,0.26,-0.5],
	[-0.26,-0.27,0.17,0.87]]
X, y=spiral_data(100,3)
class Layer_Dense:
	def __init__(self,n_inputs,n_neurons):
		self.weight=0.1*np.random.randn(n_inputs,n_neurons)
		self.biases=np.zeros((1,n_neurons))
	def forward(self, inputs):
		self.output=np.dot(inputs,self.weight)+self.biases

class Activation_ReLu:
	def forward(self, inputs):
		self.outputs=np.maximum(0, inputs)
layer1=Layer_Dense(2,5)
# layer2=Layer_Dense(5,2)
activation = Activation_ReLu()
layer1.forward(X)
# layer2.forward(layer1.output)
activation.forward(layer1.output)

print(activation.outputs)
plt.scatter(X[:,0],X[:,1])
plt.show()
plt.scatter(X[:,0],X[:,1], c=y, cmap="brg")
plt.show()

'''
a=[]
for i, j in zip(weights,biases):
	m=0
	for k,l in zip(inputs, i):
		m+=k*l
	m+=j
	a.append(m)
print(a)
'''	
