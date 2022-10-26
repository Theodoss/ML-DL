"""
File: titanic_level1.py
Name: 
----------------------------------
This file builds a machine learning algorithm from scratch 
by Python codes. We'll be using 'with open' to read in dataset,
store data into a Python dict, and finally train the model and 
test it on kaggle. This model is the most flexible one among all
levels. You should do hyperparameter tuning and find the best model.
"""

import math
from collections import defaultdict
TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'


def data_preprocess(filename: str, data: dict, mode='Train', training_data=None):
	"""
	:param filename: str, the filename to be processed
	:param data: an empty Python dictionary
	:param mode: str, indicating the mode we are using
	:param training_data: dict[str: list], key is the column name, value is its data
						  (You will only use this when mode == 'Test')
	:return data: dict[str: list], key is the column name, value is its data
	"""
	############################
	#                          #
	#          TODO:           #
	#                          #
	############################
	with open (filename, 'r') as f:
		head = True
		for line in f:
			line = line.strip()
			line = line.split(',')
			# 表頭構建
			if head == True:
				data["ID"] = []
				data["Pclass"] = []
				data["Sex"] = []
				data["Age"] = []
				data["SibSp"] = []
				data["Parch"] = []
				data["Fare"] = []
				data["Embarked"] = []
				head = False
			else:
			# if mode ='Test':
				pclass = 0
				sex = 0
				age = 0
				sibsp = 0
				parch = 0
				fare = 0
				embarked = 0
				null_marker = False
				for i in range(len(line)):
					if i == 0:
						id = line[0]
					if i == 2:
						if line[i] != "":
							pclass = line[i]
						else:
							null_marker = True
					elif i == 5:
						if line[i] != "":
							sex = line[i]
							sex = 1 if sex == 'Male'else 0
						else:
							null_marker = True
					elif i == 6:
						if line[i] != "":
							age = line[i]
						else:
							null_marker = True
					elif i == 7:
						if line[i] != "":
							sibsp = line[i]
						else:
							null_marker = True
					elif i == 8:
						if line[i] != "":
							parch = line[i]
						else:
							null_marker = True
					elif i == 10:
						if line[i] != "":
							fare = line[i]
						else:
							null_marker = True
					elif i == 12:
						if line[i] != "":
							embarked = line[i]
							if embarked == 'S':
								embarked = 0
							elif embarked == 'C':
								embarked = 1
							else:
								embarked = 2
						else:
							null_marker = True

				if not null_marker:
					data["ID"].append(int(id))
					data["Pclass"].append(int(pclass))
					data["Sex"].append(int(sex))
					data["Age"].append(float(age))
					data["SibSp"].append(int(sibsp))
					data["Parch"].append(int(parch))
					data["Fare"].append(float(fare))
					data["Embarked"].append(int(embarked))

	print(data["ID"])
	print(len(data["Pclass"]))
	return data


def one_hot_encoding(data: dict, feature: str):
	"""
	:param data: dict[str, list], key is the column name, value is its data
	:param feature: str, the column name of interest
	:return data: dict[str, list], remove the feature column and add its one-hot encoding features
	"""
	############################
	#                          #
	#          TODO:           #
	#                          #
	############################
	return data


def normalize(data: dict):
	"""
	:param data: dict[str, list], key is the column name, value is its data
	:return data: dict[str, list], key is the column name, value is its normalized data
	"""
	############################
	#                          #
	#          TODO:           #
	#                          #
	############################
	return data


def learnPredictor(inputs: dict, labels: list, degree: int, num_epochs: int, alpha: float):
	"""
	:param inputs: dict[str, list], key is the column name, value is its data
	:param labels: list[int], indicating the true label for each data
	:param degree: int, degree of polynomial features
	:param num_epochs: int, the number of epochs for training
	:param alpha: float, known as step size or learning rate
	:return weights: dict[str, float], feature name and its weight
	"""
	# Step 1 : Initialize weights
	weights = {}  # feature => weight
	keys = list(inputs.keys())
	if degree == 1:
		for i in range(len(keys)):
			weights[keys[i]] = 0
	elif degree == 2:
		for i in range(len(keys)):
			weights[keys[i]] = 0
		for i in range(len(keys)):
			for j in range(i, len(keys)):
				weights[keys[i] + keys[j]] = 0
	# Step 2 : Start training
	# TODO:
	# Step 3 : Feature Extract
	# TODO:
	# Step 4 : Update weights
	# TODO:
	return weights

data_preprocess("titanic_data/train.csv",{})