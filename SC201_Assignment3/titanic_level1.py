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
    with open(filename, 'r') as f:
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
                            pclass = round(sum(training_data["Pclass"]) / len(training_data["Pclass"]), 3)
                    elif i == 5 - test_data_num:
                        sex = 1 if line[i] == 'male' else 0
                        if mode == 'Test' and sex == '':
                            sex = round(sum(training_data["Sex"]) / len(training_data["Sex"]), 3)
                    elif i == 6 - test_data_num:
                        age = line[i]
                        if mode == 'Test' and age == '':
                            age = round(sum(training_data["Age"]) / len(training_data["Age"]), 3)
                    elif i == 7 - test_data_num:
                        sibsp = line[i]
                        if mode == 'Test' and sibsp == '':
                            sibsp = round(sum(training_data["SibSp"]) / len(training_data["SibSp"]), 3)
                    elif i == 8 - test_data_num:
                        parch = line[i]
                        if mode == 'Test' and parch == '':
                            parch = round(sum(training_data["Parch"]) / len(training_data["Parch"]), 3)
                    elif i == 10 - test_data_num:
                        fare = line[i]
                        if mode == 'Test' and fare == '':
                            fare = round(sum(training_data["Fare"]) / len(training_data["Fare"]), 3)
                    elif i == 12 - test_data_num:
                        if line[i] == 'S':
                            embarked = 0
                        elif line[i] == 'C':
                            embarked = 1
                        elif line[i] == 'Q':
                            embarked = 2
                        elif embarked == 'Test' and line[i] == '':
                            embarked = round(sum(training_data["Embarked"]) / len(training_data["Embarked"]), 3)

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
    unique_value = []
    new_list = []
    # 找出value中獨特value的數量
    for word in data[feature]:
        if word not in unique_value:
            unique_value.append(word)
    unique_value = sorted(unique_value)
    # 依數量建構list
    for i in range(len(unique_value)):
        new_list.append([])

    # 加入序列
    for word in data[feature]:
        index = unique_value.index(word)
        for i in range(len(new_list)):
            if i == index:
                new_list[i].append(1)
            else:
                new_list[i].append(0)

    # 合併序列
    for i in range(len(new_list)):
        data[feature + "_" + str(i)] = new_list[i]

    # 刪除舊序列
    del data[feature]

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
            data[each_data][i] = (data[each_data][i] - data_min) / (data_max - data_min)
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
            for i in range(len(inputs[keys[0]])):
                # Step 3 : Feature Extract
                # TODO:
                feature = {}
                for key in inputs:
                    feature[key] = inputs[key][i]
                # Step 4 : Update weights
                # TODO:
                k = util.dotProduct(weights, feature)
                h = 1 / (1 + math.exp(-k))
                y = labels[i]
                util.increment(weights, -alpha * (h - y), feature)

        elif degree == 2:
            for i in range(len(inputs[keys[0]])):
                # Step 3 : Feature Extract
                # TODO:
                feature = {}
                # Degree 1 feature
                for key in inputs:
                    feature[key] = inputs[key][i]
                # Degree 2 feature
                degree2_feature = {}
                for key1 in feature:
                    for key2 in feature:
                        # if key2 + key1 not in degree2_feature:
                        degree2_feature[key1 + key2] = feature[key1] * feature[key2]
                feature.update(degree2_feature)
                # Step 4 : Update weights
                # TODO:
                k = util.dotProduct(weights, feature)
                h = 1 / (1 + math.exp(-k))
                y = labels[i]
                util.increment(weights, -alpha * (h - y), feature)

    return weights


data_preprocess("titanic_data/train.csv", {})