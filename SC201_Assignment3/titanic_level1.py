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
import statistics
import util
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
				if mode == "Train":
					data["Survived"] = []
				data["Pclass"] = []
				data["Sex"] = []
				data["Age"] = []
				data["SibSp"] = []
				data["Parch"] = []
				data["Fare"] = []
				data["Embarked"] = []
				head = False
			else:
				pclass = ""
				sex = ""
				age = ""
				sibsp = ""
				parch = ""
				fare = ""
				embarked = ""
				survived = ""
				# 資料擷取與處理
				for i in range(len(line)):
					# 控制 Test /Train 資料行數不一樣
					test_data_num = 1 if mode == 'Test' else 0
					if i == 1 and mode == 'Train':
						survived = line[i]
					if i == 2 - test_data_num:
						pclass = line[i]
						if mode == 'Test' and pclass == '':
							pclass = round(sum(training_data["Pclass"])/len(training_data["Pclass"]),3)
					elif i == 5 - test_data_num:
						sex = 1 if line[i] == 'male'else 0
						if mode == 'Test' and sex == '':
							sex = round(sum(training_data["Sex"])/len(training_data["Sex"]),3)
					elif i == 6 - test_data_num:
						age = line[i]
						if mode == 'Test' and age == '':
							age = round(sum(training_data["Age"])/len(training_data["Age"]),3)
					elif i == 7 - test_data_num:
						sibsp = line[i]
						if mode == 'Test' and sibsp == '':
							sibsp = round(sum(training_data["SibSp"])/len(training_data["SibSp"]),3)
					elif i == 8- test_data_num:
						parch = line[i]
						if mode == 'Test' and parch == '':
							parch = round(sum(training_data["Parch"])/len(training_data["Parch"]),3)
					elif i == 10 - test_data_num:
						fare = line[i]
						if mode == 'Test' and fare == '':
							fare = round(sum(training_data["Fare"])/len(training_data["Fare"]),3)
					elif i == 12 - test_data_num:
						if line[i] == 'S':
							embarked = 0
						elif line[i] == 'C':
							embarked = 1
						elif line[i] == 'Q':
							embarked = 2
						elif embarked == 'Test' and line[i] == '':
							embarked = round(sum(training_data["Embarked"])/len(training_data["Embarked"]),3)

				# 逐列資料檢查並填入dict
				if mode == "Train":
					if pclass != "" and sex != "" and age != "" and sibsp != "" and parch != "" and fare != "" and embarked != "":
						data["Survived"].append(int(survived))
						data["Pclass"].append(int(pclass))
						data["Sex"].append(int(sex))
						data["Age"].append(float(age))
						data["SibSp"].append(int(sibsp))
						data["Parch"].append(int(parch))
						data["Fare"].append(float(fare))
						data["Embarked"].append(int(embarked))
				else:
					data["Pclass"].append(int(pclass))
					data["Sex"].append(int(sex))
					data["Age"].append(float(age))
					data["SibSp"].append(int(sibsp))
					data["Parch"].append(int(parch))
					data["Fare"].append(float(fare))
					data["Embarked"].append(int(embarked))
					# print(survived, pclass, sex, age,sibsp, parch, fare, embarked)
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
	sex_0 = []
	sex_1 = []
	for sex in data["Sex"]:
		if sex == 0:
			sex_0.append(1)
			sex_1.append(0)
		else:
			sex_0.append(0)
			sex_1.append(1)
	data["Sex_0"] = sex_0
	data["Sex_1"] = sex_1
	if 'Sex' in data:
		del data['Sex']

	embarked_0 = []
	embarked_1 = []
	embarked_2 = []
	for embarked in data["Embarked"]:
		if embarked == 0:
			embarked_0.append(1)
			embarked_1.append(0)
			embarked_2.append(0)
		elif embarked == 1 :
			embarked_0.append(0)
			embarked_1.append(1)
			embarked_2.append(0)
		else:
			embarked_0.append(0)
			embarked_1.append(0)
			embarked_2.append(1)
	data["Embarked_0"] = embarked_0
	data["Embarked_1"] = embarked_1
	data["Embarked_2"] = embarked_2
	if 'Embarked' in data:
		del data['Embarked']

	pclass_0 = []
	pclass_1 = []
	pclass_2 = []
	for pclass in data["Pclass"]:
		if pclass == 1:
			pclass_0.append(1)
			pclass_1.append(0)
			pclass_2.append(0)
		elif pclass == 2:
			pclass_0.append(0)
			pclass_1.append(1)
			pclass_2.append(0)
		else:
			pclass_0.append(0)
			pclass_1.append(0)
			pclass_2.append(1)
	data["Pclass_0"] = pclass_0
	data["Pclass_1"] = pclass_1
	data["Pclass_2"] = pclass_2
	if 'Pclass' in data:
		del data['Pclass']

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
	for each_data in data:
		data_min = min(data[each_data])
		data_max = max(data[each_data])
		for i in range(len(data[each_data])):
			data[each_data][i] = (data[each_data][i]-data_min)/(data_max-data_min)
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
	for epoch in range(num_epochs):
		if degree == 1:
			feature = []
			for i in range(len(inputs[0])):
				feature = []
				for keys in inputs:
					feature.append(inputs[keys][i])
				h = 1/(1 + math.exp(-(util.dotProduct(weights, feature))))
				weights = util.increment(weights, -alpha*(h-labels[i]), feature)


		# elif degree == 2:


	# Step 3 : Feature Extract
	# TODO:
	# Step 4 : Update weights
	# TODO:
	return weights

data_preprocess("titanic_data/train.csv",{})
