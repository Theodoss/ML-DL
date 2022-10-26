"""
File: titanic_batch_gradient_descent.py
Name: 
-----------------------------
This file demonstrates how to use batch
gradient descent to update weights by numpy 
array. The training process should be way
faster than stochastic gradient descent
(You should see a smooth training curve as well)
-----------------------------
w1.shape = (n, 1)
X.shape = (n, m)
H.shape = (1, m)
Y.shape = (1, m)
"""


import time
import numpy as np


TRAIN = 'titanic_data/train.csv'
NUM_EPOCHS = 60000
ALPHA = 0.05


def main():
	start = time.time()
	X_train, Y = data_preprocessing()
	print('Y.shape', Y.shape)
	print('X.shape', X_train.shape)
	# ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
	n, m = X_train.shape
	X = normalize(X_train)
	###############################
	#                             #
	#            TODO:            #
	#                             #
	###############################
	end = time.time()
	print('Total run time (Batch-GD):', end-start)


def normalize(X):
	"""
	:param X: numpy_array, the dimension is (n, m)
	:return: numpy_array, the values are normalized, where the dimension is still (n, m)
	"""
	pass


def batch_gradient_descent(X, Y):
	"""
	:param X: numpy_array, the array holding all the training data
	:param Y: numpy_array, the array holding all the ture labels in X
	:return: numpy_array, the trained weights with dimension (n, m)
	"""
	np.random.seed(0)
	# Initialize w and b
	####################
	#                  #
	#       TODO:      #
	#                  #
	####################

	# Start Training
	####################
	#                  #
	#       TODO:      #
	#                  #
	####################


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
