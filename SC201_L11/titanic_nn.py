"""
File: titanic_nn.py
Name: 
-----------------------------
This file demonstrates how to use batch
gradient descent to update weights by numpy 
array. The training process should be way
faster than stochastic gradient descent
(You should see a smooth training curve as well)
-----------------------------
W1.shape = (6, 3)
W2.shape = (3, 1)
X.shape = (6, m)
Y.shape = (1, m)
-----------------------------
If you correctly implement this NN, you should see the following acc:
0.8151260504201681
"""


import numpy as np


TRAIN = 'titanic_data/train.csv'     # This is the filename of interest
NUM_EPOCHS = 10000                   # This constant controls the total number of epochs
ALPHA = 0.02                         # This constant controls the learning rate α
N1 = 3                               # This constant controls the number of neurons in Layer 1
N2 = 1                               # This constant controls the number of neurons in Layer 2


def main():
	X_train, Y = data_preprocessing()
	_, m = X_train.shape
	print('Y.shape', Y.shape)
	print('X.shape', X_train.shape)
	# ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
	X = normalize(X_train)
	####################################
	#                                  #
	#              TODO:               #
	#                                  #
	####################################


def normalize(X):
	"""
	:param X: numpy_array, the dimension is (n, m)
	:return: numpy_array, the values are normalized, where the dimension is still (n, m)
	"""
	min_array = np.min(X, axis=1, keepdims=True)
	max_array = np.max(X, axis=1, keepdims=True)
	return (X-min_array)/(max_array-min_array)


def two_layer_nn(X, Y):
	"""
	:param X: numpy_array, the array holding all the training data
	:param Y: numpy_array, the array holding all the ture labels in X
	:return W1, W2, B1, B2: numpy_array, the array holding all the parameters 
	"""
	n, m = X.shape
	np.random.seed(1)

	# Initialize W1

	# Initialize W2

	# Initialize B1

	# Initialize B2
	
	for epoch in range(NUM_EPOCHS):
		pass
		# Forward Pass
		# TODO:

		# Backward Pass
		# TODO:

		# Updates all the weights and biases
		# TODO:
	return


def data_preprocessing(mode='train'):
	"""
	:param mode: str, indicating if it's training mode or testing mode
	:return: Tuple(numpy_array, numpy_array), the first one is X, the other one is Y
	"""
	data_lst = []
	label_lst = []
	first_data = True
	if mode == 'train':
		with open(TRAIN, 'r') as f:
			for line in f:
				data = line.split(',')
				# ['0PassengerId', '1Survived', '2Pclass', '3Last Name', '4First Name', '5Sex', '6Age', '7SibSp', '8Parch', '9Ticket', '10Fare', '11Cabin', '12Embarked']
				if first_data:
					first_data = False
					continue
				if not data[6]:
					continue
				label = [int(data[1])]
				if data[5] == 'male':
					sex = 1
				else:
					sex = 0
				# ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
				passenger_lst = [int(data[2]), sex, float(data[6]), int(data[7]), int(data[8]), float(data[10])]
				data_lst.append(passenger_lst)
				label_lst.append(label)
	else:
		pass
	return np.array(data_lst).T, np.array(label_lst).T


if __name__ == '__main__':
	main()
